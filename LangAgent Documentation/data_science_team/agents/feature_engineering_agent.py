# LangAgent/data_science_team/agents/feature_engineering.py


# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import io
import pandas as pd

from langagent.data_science_team.templates.agent_templates import(
    node_func_execute_agent_code_on_data, 
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_explain_agent_code, 
    create_coding_agent_graph
)
from langagent.data_science_team.tools.parsers import PythonOutputParser
from langagent.data_science_team.tools.regex import relocate_imports_inside_function, add_comments_to_top
from langagent.data_science_team.tools.metadata import get_dataframe_summary
from langagent.data_science_team.tools.logging import log_function

# Setup
AGENT_NAME = "feature_engineering_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# * Feature Engineering Agent

def make_feature_engineering_agent(model, log=False, log_path=None, overwrite = True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False):
    """
    Creates a feature engineering agent that can be run on a dataset. The agent applies various feature engineering
    techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,
    and generating polynomial features. The agent takes in a dataset and user instructions and outputs a Python
    function for feature engineering. It also logs the code generated and any errors that occur.
    
    The agent is instructed to apply the following feature engineering techniques:
    
    - Remove string or categorical features with unique values equal to the size of the dataset
    - Remove constant features with the same value in all rows
    - High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
    - Encoding categorical variables using OneHotEncoding
    - Numeric features should be left untransformed
    - Create datetime-based features if datetime columns are present
    - If a target variable is provided:
        - If a categorical target variable is provided, encode it using LabelEncoding
        - All other target variables should be converted to numeric and unscaled
    - Convert any boolean True/False values to 1/0
    - Return a single data frame containing the transformed features and target variable, if one is provided.
    - Any specific instructions provided by the user

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to "logs/".
    overwrite : bool, optional
        Whether or not to overwrite the log file if it already exists. If False, a unique file name will be created. 
        Defaults to True.
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.

    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from langagent.data_science_team.agents import feature_engineering_agent

    llm = ChatOpenAI(model="gpt-4o")

    feature_engineering_agent = make_feature_engineering_agent(llm)

    df = pd.read_csv("https://github.com/knowusuboaky/LangAgent/tree/main/data/churn_data.csv")

    response = feature_engineering_agent.invoke({
        "user_instructions": None,
        "target_variable": "Churn",
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['data_engineered'])
    ```

    Returns
    -------
    app : langchain.graphs.StateGraph
        The feature engineering agent as a state graph.
    """
    llm = model

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_engineered: dict
        target_variable: str
        all_datasets_summary: str
        feature_engineer_function: str
        feature_engineer_function_path: str
        feature_engineer_function_name: str
        feature_engineer_error: str
        max_retries: int
        retry_count: int

    def recommend_feature_engineering_steps(state: GraphState):
        """
        Recommend a series of feature engineering steps based on the input data.
        These recommended steps will be appended to the user_instructions.
        """
        print("---FEATURE ENGINEERING AGENT----")
        print("    * RECOMMEND FEATURE ENGINEERING STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Expert. Given the following information about the data, 
            recommend a series of numbered steps to take to engineer features. 
            The steps should be tailored to the data characteristics and should be helpful 
            for a feature engineering agent that will be implemented.
            
            General Steps:
            Things that should be considered in the feature engineering steps:
            
            * Convert features to the appropriate data types based on their sample data values
            * Remove string or categorical features with unique values equal to the size of the dataset
            * Remove constant features with the same value in all rows
            * High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
            * Encoding categorical variables using OneHotEncoding
            * Numeric features should be left untransformed
            * Create datetime-based features if datetime columns are present
            * If a target variable is provided:
                * If a categorical target variable is provided, encode it using LabelEncoding
                * All other target variables should be converted to numeric and unscaled
            * Convert any Boolean (True/False) values to integer (1/0) values. This should be performed after one-hot encoding.
            
            Custom Steps:
            * Analyze the data to determine if any additional feature engineering steps are needed.
            * Recommend steps that are specific to the data provided. Include why these steps are necessary or beneficial.
            * If no additional steps are needed, simply state that no additional steps are required.
            
            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
            
            User instructions:
            {user_instructions}
            
            Previously Recommended Steps (if any):
            {recommended_steps}
            
            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return the steps as a numbered list (no code, just the steps).
            
            Avoid these:
            1. Do not include steps to save files.
            """,
            input_variables=["user_instructions", "recommended_steps", "all_datasets_summary"]
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        all_datasets_summary = get_dataframe_summary([df])
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str
        }) 
        
        return {
            "recommended_steps": "\n\n# Recommended Feature Engineering Steps:\n" + recommended_steps.content.strip(),
            "all_datasets_summary": all_datasets_summary_str
        }
    
    
    # Human Review   
    prompt_text_human_review = "Are the following feature engineering instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_feature_engineering_steps", "explain_feature_engineering_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= 'explain_feature_engineering_code',
                no_goto="recommend_feature_engineering_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="feature_engineer_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_feature_engineering_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_feature_engineering_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="feature_engineer_function", 
            )
    
    def create_feature_engineering_code(state: GraphState):
        if bypass_recommended_steps:
            print("---FEATURE ENGINEERING AGENT----")
        print("    * CREATE FEATURE ENGINEERING CODE")

        feature_engineering_prompt = PromptTemplate(
            template="""
            
            You are a Feature Engineering Agent. Your job is to create a feature_engineer() function that can be run on the data provided using the following recommended steps.
            
            Recommended Steps:
            {recommended_steps}
            
            Use this information about the data to help determine how to feature engineer the data:
            
            Target Variable (if provided): {target_variable}
            
            Below are summaries of all datasets provided. Use this information about the data to help determine how to feature engineer the data:
            {all_datasets_summary}
            
            You can use Pandas, Numpy, and Scikit Learn libraries to feature engineer the data.
            
            Return Python code in ```python``` format with a single function definition, feature_engineer(data_raw), including all imports inside the function.

            Return code to provide the feature engineering function:
            
            def feature_engineer(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_engineered
            
            Best Practices and Error Preventions:
            - Handle missing values in numeric and categorical features before transformations.
            - Avoid creating highly correlated features unless explicitly instructed.
            - Convert Boolean to integer values (0/1) after one-hot encoding unless otherwise instructed.
            
            Avoid the following errors:
            
            - name 'OneHotEncoder' is not defined
            
            - Shape of passed values is (7043, 48), indices imply (7043, 47)
            
            - name 'numeric_features' is not defined
            
            - name 'categorical_features' is not defined


            """,
            input_variables=["recommeded_steps", "target_variable", "all_datasets_summary"]
        )

        feature_engineering_agent = feature_engineering_prompt | llm | PythonOutputParser()

        response = feature_engineering_agent.invoke({
            "recommended_steps": state.get("recommended_steps"),
            "target_variable": state.get("target_variable"),
            "all_datasets_summary": state.get("all_datasets_summary"),
        })
        
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        # For logging: store the code generated
        file_path, file_name = log_function(
            response=response,
            file_name="feature_engineering_agent.py",
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )

        return {
            "feature_engineer_function": response,
            "feature_engineer_function_path": file_path,
            "feature_engineer_function_name": file_name
        }

    

    def execute_feature_engineering_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_engineered",
            error_key="feature_engineer_error",
            code_snippet_key="feature_engineer_function",
            agent_function_name="feature_engineer",
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during feature engineering: "
        )

    def fix_feature_engineering_code(state: GraphState):
        feature_engineer_prompt = """
        You are a Feature Engineering Agent. Your job is to fix the feature_engineer() function that currently contains errors.
        
        Provide only the corrected function definition.
        
        Broken code:
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="feature_engineer_function",
            error_key="feature_engineer_error",
            llm=llm,
            prompt_template=feature_engineer_prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("feature_engineer_function_path"),
        )

    def explain_feature_engineering_code(state: GraphState):
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="feature_engineer_function",
            result_key="messages",
            error_key="feature_engineer_error",
            llm=llm,
            role=AGENT_NAME,
            explanation_prompt_template="""
            Explain the feature engineering steps performed by this function. Keep the explanation clear and concise.\n\n# Feature Engineering Agent:\n\n{code}
            """,
            success_prefix="# Feature Engineering Agent:\n\n ",
            error_message="The Feature Engineering Agent encountered an error during feature engineering. Data could not be explained."
        )

    # Create the graph
    node_functions = {
        "recommend_feature_engineering_steps": recommend_feature_engineering_steps,
        "human_review": human_review,
        "create_feature_engineering_code": create_feature_engineering_code,
        "execute_feature_engineering_code": execute_feature_engineering_code,
        "fix_feature_engineering_code": fix_feature_engineering_code,
        "explain_feature_engineering_code": explain_feature_engineering_code
    }
    
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_feature_engineering_steps",
        create_code_node_name="create_feature_engineering_code",
        execute_code_node_name="execute_feature_engineering_code",
        fix_code_node_name="fix_feature_engineering_code",
        explain_code_node_name="explain_feature_engineering_code",
        error_key="feature_engineer_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
    )

    return app
