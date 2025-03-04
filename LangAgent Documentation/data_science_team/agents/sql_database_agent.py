# LangAgent/data_science_team/agents/sql database.py


from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import io
import pandas as pd
import sqlalchemy as sql

from langagent.data_science_team.templates.agent_templates import(
    node_func_execute_agent_from_sql_connection,
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_explain_agent_code, 
    create_coding_agent_graph
)
from langagent.data_science_team.tools.parsers import PythonOutputParser, SQLOutputParser  
from langagent.data_science_team.tools.regex import relocate_imports_inside_function, add_comments_to_top
from langagent.data_science_team.tools.metadata import get_database_metadata
from langagent.data_science_team.tools.logging import log_function

# Setup
AGENT_NAME = "sql_database_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


def make_sql_database_agent(model, connection, n_samples = 10, log=False, log_path=None, overwrite = True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False):
    """
    Creates a SQL Database Agent that can recommend SQL steps and generate SQL code to query a database. 
    
    Parameters
    ----------
    model : ChatOpenAI
        The language model to use for the agent.
    connection : sqlalchemy.engine.base.Engine
        The connection to the SQL database.
    n_samples : int, optional
        The number of samples to retrieve for each column, by default 10. 
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether to log the generated code, by default False
    log_path : str, optional
        The path to the log directory, by default None
    overwrite : bool, optional
        Whether to overwrite the existing log file, by default True
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    
    Returns
    -------
    app : langchain.graphs.StateGraph
        The data cleaning agent as a state graph.
        
    Examples
    --------
    ```python
    from langagent.data_science_team.agents import make_sql_database_agent
    import sqlalchemy as sql
    from langchain_openai import ChatOpenAI

    sql_engine = sql.create_engine("sqlite:///data/database-sql-transactions/leads_scored.db")

    conn = sql_engine.connect()

    llm = ChatOpenAI(model="gpt-4o-mini")
    
    sql_agent = make_sql_database_agent(
        model=llm, 
        connection=conn
    )

    sql_agent

    response = sql_agent.invoke({
        "user_instructions": "List the tables in the database",
        "max_retries":3, 
        "retry_count":0
    })
    ```
    
    """
    
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection
    
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
        data_sql: dict
        all_sql_database_summary: str
        sql_query_code: str
        sql_database_function: str
        sql_database_function_path: str
        sql_database_function_name: str
        sql_database_error: str
        max_retries: int
        retry_count: int
    
    def recommend_sql_steps(state: GraphState):
        
        print("---SQL DATABASE AGENT---")
        print("    * RECOMMEND SQL QUERY STEPS")
        
        
        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a SQL Database Instructions Expert. Given the following information about the SQL database, 
            recommend a series of numbered steps to take to collect the data and process it according to user instructions. 
            The steps should be tailored to the SQL database characteristics and should be helpful 
            for a sql database coding agent that will write the SQL code.
            
            IMPORTANT INSTRUCTIONS:
            - Take into account the user instructions and the previously recommended steps.
            - If no user instructions are provided, just return the steps needed to understand the database.
            - Take into account the database dialect and the tables and columns in the database.
            - Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            
            
            User instructions / Question:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of the database metadata and the SQL tables:
            {all_sql_database_summary}

            Return the steps as a numbered point list (no code, just the steps).
            
            Consider these:
            
            1. Consider the database dialect and the tables and columns in the database.
            
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include steps to modify existing tables, create new tables or modify the database schema.
            3. Do not include steps that alter the existing data in the database.
            4. Make sure not to include unsafe code that could cause data loss or corruption or SQL injections.
            
            """,
            input_variables=["user_instructions", "recommended_steps", "all_sql_database_summary"]
        )
        
        # Create a connection if needed
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        # Get the database metadata
        all_sql_database_summary = get_database_metadata(conn, n_samples=n_samples)
        
        steps_agent = recommend_steps_prompt | llm
        
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_sql_database_summary": all_sql_database_summary
        })
        
        return {
            "recommended_steps": "\n\n# Recommended SQL Database Steps:\n" + recommended_steps.content.strip(),
            "all_sql_database_summary": all_sql_database_summary
        }
        
    def create_sql_query_code(state: GraphState):
        if bypass_recommended_steps:
            print("---SQL DATABASE AGENT---")
        print("    * CREATE SQL QUERY CODE")
        
        # Prompt to get the SQL code from the LLM
        sql_query_code_prompt = PromptTemplate(
            template="""
            You are a SQL Database Coding Expert. Given the following information about the SQL database, 
            write the SQL code to collect the data and process it according to user instructions. 
            The code should be tailored to the SQL database characteristics and should take into account user instructions, recommended steps, database and table characteristics.
            
            IMPORTANT INSTRUCTIONS:
            - Do not use a LIMIT clause unless a user specifies a limit to be returned.
            - Return SQL in ```sql ``` format.
            - Only return a single query if possible.
            - Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            - Pay attention to the SQL dialect from the database summary metadata. Write the SQL code according to the dialect specified.
            
            
            User instructions / Question:
            {user_instructions}

            Recommended Steps:
            {recommended_steps}

            Below are summaries of the database metadata and the SQL tables:
            {all_sql_database_summary}

            Return:
            - The SQL code in ```sql ``` format to collect the data and process it according to the user instructions.
            
            Avoid these:
            - Do not include steps to save files.
            - Do not include steps to modify existing tables, create new tables or modify the database schema.
            - Make sure not to alter the existing data in the database.
            - Make sure not to include unsafe code that could cause data loss or corruption.
            
            """,
            input_variables=["user_instructions", "recommended_steps", "all_sql_database_summary"]
        )
        
        # Create a connection if needed
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        # Get the database metadata
        all_sql_database_summary = get_database_metadata(conn, n_samples=n_samples)
        
        sql_query_code_agent = sql_query_code_prompt | llm | SQLOutputParser()
        
        sql_query_code = sql_query_code_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_sql_database_summary": all_sql_database_summary
        })
        
        print("    * CREATE PYTHON FUNCTION TO RUN SQL CODE")
        
        response = f"""
def sql_database_pipeline(connection):
    import pandas as pd
    import sqlalchemy as sql
    
    # Create a connection if needed
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    sql_query = '''
    {sql_query_code}
    '''
    
    return pd.read_sql(sql_query, connection)
        """
        
        response = add_comments_to_top(response, AGENT_NAME)
            
        # For logging: store the code generated
        file_path, file_name = log_function(
            response=response,
            file_name="sql_database_agent.py",
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )
        
        return {
            "sql_query_code": sql_query_code,
            "sql_database_function": response,
            "sql_database_function_path": file_path,
            "sql_database_function_name": file_name
        }
    
    # Human Review   
    prompt_text_human_review = "Are the following SQL agent instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_sql_steps", "explain_sql_database_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= 'explain_sql_database_code',
                no_goto="recommend_sql_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="sql_database_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_sql_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_sql_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="sql_database_function", 
            )
    
    def execute_sql_database_code(state: GraphState):
        
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        return node_func_execute_agent_from_sql_connection(
            state=state,
            connection=conn,
            result_key="data_sql",
            error_key="sql_database_error",
            code_snippet_key="sql_database_function",
            agent_function_name="sql_database_pipeline",
            post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during executing the sql database pipeline: "
        )
    
    def fix_sql_database_code(state: GraphState):
        prompt = """
        You are a SQL Database Agent code fixer. Your job is to create a sql_database_pipeline(connection) function that can be run on a sql connection. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for sql_database_pipeline().
        
        Return Python code in ```python``` format with a single function definition, sql_database_pipeline(connection), that includes all imports inside the function. The connection object is a SQLAlchemy connection object. Don't specify the class of the connection object, just use it as an argument to the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="sql_database_function",
            error_key="sql_database_error",
            llm=llm,  
            prompt_template=prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("sql_database_function_path", None),
        )
        
    def explain_sql_database_code(state: GraphState):
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="sql_database_function",
            result_key="messages",
            error_key="sql_database_error",
            llm=llm,
            role=AGENT_NAME,
            explanation_prompt_template="""
            Explain the SQL steps that the SQL Database agent performed in this function. 
            Keep the summary succinct and to the point.\n\n# SQL Database Agent:\n\n{code}
            """,
            success_prefix="# SQL Database Agent:\n\n",  
            error_message="The SQL Database Agent encountered an error during SQL Query Analysis. No SQL function explanation is returned."
        )
        
    # Create the graph
    node_functions = {
        "recommend_sql_steps": recommend_sql_steps,
        "human_review": human_review,
        "create_sql_query_code": create_sql_query_code,
        "execute_sql_database_code": execute_sql_database_code,
        "fix_sql_database_code": fix_sql_database_code,
        "explain_sql_database_code": explain_sql_database_code
    }
    
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_sql_steps",
        create_code_node_name="create_sql_query_code",
        execute_code_node_name="execute_sql_database_code",
        fix_code_node_name="fix_sql_database_code",
        explain_code_node_name="explain_sql_database_code",
        error_key="sql_database_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
    )
        
    return app
        
             
            
        
        
        
        
    