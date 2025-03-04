# LangAgent/data_science_team/agents/data_wrangling.py


# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Union
import operator
import os
import io
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

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

# Setup Logging Path
AGENT_NAME = "data_wrangling_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

def make_data_wrangling_agent(model, log=False, log_path=None, overwrite = True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False):
    """
    Creates a data wrangling agent that can be run on one or more datasets. The agent can be
    instructed to perform common data wrangling steps such as:
    
    - Joining or merging multiple datasets
    - Reshaping data (pivoting, melting)
    - Aggregating data via groupby operations
    - Encoding categorical variables (one-hot, label encoding)
    - Creating computed features (e.g., ratio of two columns)
    - Ensuring consistent data types
    - Dropping or rearranging columns

    The agent takes in one or more datasets (passed as a dictionary or list of dictionaries if working on multiple dictionaries), user instructions,
    and outputs a python function that can be used to wrangle the data. If multiple datasets
    are provided, the agent should combine or transform them according to user instructions.

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
        Whether or not to use human in the loop. If True, adds an interrupt and human-in-the-loop 
        step that asks the user to review the data wrangling instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.

    Example
    -------
    ``` python
    from langchain_openai import ChatOpenAI
    import pandas as pd
    
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C'],
        'value': [10, 20, 15, 5]
    })
    
    llm = ChatOpenAI(model="gpt-4o")
    
    data_wrangling_agent = make_data_wrangling_agent(llm)

    response = data_wrangling_agent.invoke({
        "user_instructions": "Calculate the sum and mean of 'value' by 'category'.",
        "data_raw": df.to_dict(),
        "max_retries":3, 
        "retry_count":0
    })
    pd.DataFrame(response['data_wrangled'])
    ```
    
    Returns
    -------
    app : langchain.graphs.StateGraph
        The data wrangling agent as a state graph.
    """
    llm = model
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    
    
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        # data_raw should be a dict for a single dataset or a list of dicts for multiple datasets
        data_raw: Union[dict, list]
        data_wrangled: dict
        all_datasets_summary: str
        data_wrangler_function: str
        data_wrangler_function_path: str
        data_wrangler_function_name: str
        data_wrangler_error: str
        max_retries: int
        retry_count: int

    def recommend_wrangling_steps(state: GraphState):
        print("---DATA WRANGLING AGENT----")
        print("    * RECOMMEND WRANGLING STEPS")

        data_raw = state.get("data_raw")

        if isinstance(data_raw, dict):
            # Single dataset scenario
            primary_dataset_name = "main"
            datasets = {primary_dataset_name: data_raw}
        elif isinstance(data_raw, list) and all(isinstance(item, dict) for item in data_raw):
            # Multiple datasets scenario
            datasets = {f"dataset_{i}": d for i, d in enumerate(data_raw, start=1)}
            primary_dataset_name = "dataset_1"
        else:
            raise ValueError("data_raw must be a dict or a list of dicts.")

        # Convert all datasets to DataFrames for inspection
        dataframes = {name: pd.DataFrame.from_dict(d) for name, d in datasets.items()}

        # Create a summary for all datasets
        # We'll include a short sample and info for each dataset
        all_datasets_summary = get_dataframe_summary(dataframes)

        # Join all datasets summaries into one big text block
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        # Prepare the prompt:
        # We now include summaries for all datasets, not just the primary dataset.
        # The LLM can then use all this info to recommend steps that consider merging/joining.
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Wrangling Expert. Given the following data (one or multiple datasets) and user instructions, 
            recommend a series of numbered steps to wrangle the data based on a user's needs. 
            
            You can use any common data wrangling techniques such as joining, reshaping, aggregating, encoding, etc. 
            
            If multiple datasets are provided, you may need to recommend how to merge or join them. 
            
            Also consider any special transformations requested by the user. If the user instructions 
            say to do something else or not to do certain steps, follow those instructions.
            
            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return your recommended steps as a numbered point list, explaining briefly why each step is needed.
            
            Avoid these:
            1. Do not include steps to save files.
            """,
            input_variables=["user_instructions", "recommended_steps", "all_datasets_summary"]
        )

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str,
        }) 

        return {
            "recommended_steps": "\n\n# Recommended Wrangling Steps:\n" + recommended_steps.content.strip(),
            "all_datasets_summary": all_datasets_summary_str,
        }

    
    def create_data_wrangler_code(state: GraphState):
        if bypass_recommended_steps:
            print("---DATA WRANGLING AGENT----")
        print("    * CREATE DATA WRANGLER CODE")
        
        data_wrangling_prompt = PromptTemplate(
            template="""
            You are a Data Wrangling Coding Agent. Your job is to create a data_wrangler() function that can be run on the provided data. 
            
            Follow these recommended steps:
            {recommended_steps}
            
            If multiple datasets are provided, you may need to merge or join them. Make sure to handle that scenario based on the recommended steps and user instructions.
            
            Below are summaries of all datasets provided. If more than one dataset is provided, you may need to merge or join them.:
            {all_datasets_summary}
            
            Return Python code in ```python``` format with a single function definition, data_wrangler(), that includes all imports inside the function. And returns a single pandas data frame.

            ```python
            def data_wrangler(data_list):
                '''
                Wrangle the data provided in data.
                
                data_list: A list of one or more pandas data frames containing the raw data to be wrangled.
                '''
                import pandas as pd
                import numpy as np
                # Implement the wrangling steps here
                
                # Return a single DataFrame 
                return data_wrangled
            ```
            
            Avoid Errors:
            1. If the incoming data is not a list. Convert it to a list first. 
            2. Do not specify data types inside the function arguments.
            
            Make sure to explain any non-trivial steps with inline comments. Follow user instructions. Comment code thoroughly.
            
            
            """,
            input_variables=["recommended_steps", "all_datasets_summary"]
        )

        data_wrangling_agent = data_wrangling_prompt | llm | PythonOutputParser()

        response = data_wrangling_agent.invoke({
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": state.get("all_datasets_summary")
        })
        
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)
        
        # For logging: store the code generated
        file_path, file_name = log_function(
            response=response,
            file_name="data_wrangling_agent.py",
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )

        return {
            "data_wrangler_function" : response,
            "data_wrangler_function_path": file_path,
            "data_wrangler_function_name": file_name
        }

    
    def human_review(state: GraphState) -> Command[Literal["recommend_wrangling_steps", "create_data_wrangler_code"]]:
        return node_func_human_review(
            state=state,
            prompt_text="Are the following data wrangling steps correct? (Answer 'yes' or provide modifications)\n{steps}",
            yes_goto="create_data_wrangler_code",
            no_goto="recommend_wrangling_steps",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_steps"            
        )
        
    # Human Review   
    prompt_text_human_review = "Are the following data wrangling instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_wrangling_steps", "explain_data_wrangler_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= 'explain_data_wrangler_code',
                no_goto="recommend_wrangling_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_wrangler_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_wrangling_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_wrangling_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_wrangler_function", 
            )

    
    def execute_data_wrangler_code(state: GraphState):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_wrangled",
            error_key="data_wrangler_error",
            code_snippet_key="data_wrangler_function",
            agent_function_name="data_wrangler",
            # pre_processing=pre_processing,
            post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during data wrangling: "
        )
        
    def fix_data_wrangler_code(state: GraphState):
        data_wrangler_prompt = """
        You are a Data Wrangling Agent. Your job is to create a data_wrangler() function that can be run on the data provided. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for data_wrangler().
        
        Return Python code in ```python``` format with a single function definition, data_wrangler(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_wrangler_function",
            error_key="data_wrangler_error",
            llm=llm,  
            prompt_template=data_wrangler_prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_wrangler_function_path"),
        )
    
    def explain_data_wrangler_code(state: GraphState):        
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="data_wrangler_function",
            result_key="messages",
            error_key="data_wrangler_error",
            llm=llm,  
            role=AGENT_NAME,
            explanation_prompt_template="""
            Explain the data wrangling steps that the data wrangling agent performed in this function. 
            Keep the summary succinct and to the point.\n\n# Data Wrangling Agent:\n\n{code}
            """,
            success_prefix="# Data Wrangling Agent:\n\n ",
            error_message="The Data Wrangling Agent encountered an error during data wrangling. Data could not be explained."
        )
        
    # Define the graph
    node_functions = {
        "recommend_wrangling_steps": recommend_wrangling_steps,
        "human_review": human_review,
        "create_data_wrangler_code": create_data_wrangler_code,
        "execute_data_wrangler_code": execute_data_wrangler_code,
        "fix_data_wrangler_code": fix_data_wrangler_code,
        "explain_data_wrangler_code": explain_data_wrangler_code
    }
    
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_wrangling_steps",
        create_code_node_name="create_data_wrangler_code",
        execute_code_node_name="execute_data_wrangler_code",
        fix_code_node_name="fix_data_wrangler_code",
        explain_code_node_name="explain_data_wrangler_code",
        error_key="data_wrangler_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
    )
        
    return app




