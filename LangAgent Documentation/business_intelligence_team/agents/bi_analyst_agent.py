# LangAgent/analysis_team/agents/bi_analyst.py


# * LIBRARIES


from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.output_parsers import BaseOutputParser

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, HumanMessage


from langgraph.graph import END, StateGraph

import os
import yaml
import ast
import json
import re

from pprint import pprint
from typing import Annotated, TypedDict

import pandas as pd
import sqlalchemy as sql

import plotly as pl
import plotly.express as px
from plotly.graph_objects import Figure
import plotly.io as pio

from IPython.display import Image
import matplotlib.pyplot as plt
import time


def create_bi_analyst(db_path, llm):    
    
    PATH_DB = db_path
    
    # * Routing Preprocessor Agent

    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL database agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL translator agent 
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the SQL query. If unknown, not specified or 'None' is found, then select 'table'.  
        
        Return JSON with 'formatted_user_question_sql_only' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {initial_question}
        """,
        input_variables=["initial_question"]
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    # * SQL Agent

    db = SQLDatabase.from_uri(PATH_DB)

    def extract_sql_code(text):
        sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
        if sql_code_match:
            sql_code = sql_code_match.group(1).strip()
            #print(f"Extracted SQL Code: {sql_code}")

            return sql_code
        else:
            sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
            if sql_code_match:
                sql_code = sql_code_match.group(1).strip()
                return sql_code
            else:
                return None
        
    class SQLOutputParser(BaseOutputParser):
        def parse(self, text: str):
            sql_code = extract_sql_code(text)
            if sql_code is not None:
                return sql_code
            else:
                # Assume ```sql wasn't used
                return text

    prompt_sqlite = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
        
        Do not use a LIMIT clause with {top_k} unless a user specifies a limit to be returned.
        
        Return SQL in ```sql ``` format.
        
        Only return a single query if possible.
        
        Never query for all columns from a table unless the user specifies it. You must query only the columns that are needed to answer the question unless the user specifies it. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        
        Pay attention to use date(\'now\') function to get the current date, if the question involves "today".
            
        Only use the following tables:
        {table_info}
        
        Question: {input}'
        """
    )

    sql_generator = (
        create_sql_query_chain(
            llm = llm,
            db = db,
            k = int(1e7),
            prompt = prompt_sqlite
        ) 
        | SQLOutputParser() # NEW SQLCodeExtactor
    )

    # * Dataframe Conversion
        
    sql_engine = sql.create_engine(PATH_DB)

    conn = sql_engine.connect()

    # * Chart Instructor

    prompt_chart_instructions = PromptTemplate(
        template="""
        You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting using the `matplotlib` library. 
        
        You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart using `matplotlib`.

        USER QUESTION: {question}
        
        DATA: {data}
        
        Formulate "chart generator instructions" by informing the chart generator of what type of `matplotlib` plot to use (e.g. bar, line, scatter, etc.) to best represent the data.
        
        Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.

        Return your instructions in the following format:
        CHART GENERATOR INSTRUCTIONS: FILL IN THE INSTRUCTIONS HERE
        """,
        input_variables=['question', 'data']
    )

    chart_instructor = prompt_chart_instructions | llm | StrOutputParser()


    # * Chart Generator

    repl = PythonREPL()

    @tool
    def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."]
    ):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    prompt_chart_generator = PromptTemplate(
        template = """
        You are an expert in creating beautiful and informative data visualizations using the `matplotlib` python library. You must use `matplotlib` to produce plots.
        
        Your job is to produce Python code to generate visually appealing visualizations.
        
        Create the Python code to produce the requested visualization given the plot requested from the original user question and the input data. 
        
        The input data will be provided as a dictionary and will need to be converted to a pandas data frame before creating the visualization.
        
        The output of the `matplotlib` chart should be displayed using `plt.show()`.

        Make sure to add: `import matplotlib.pyplot as plt`
        Make sure to print the figure details after generating the plot.
        Make sure to import `pandas` for handling the data.

        Here's an example of creating a beautiful plot using `matplotlib`:
        
        ```python
        import matplotlib.pyplot as plt
        import pandas as pd

        # Create a sample DataFrame
        data = {data}
        df = pd.DataFrame(data)

        # Create a plot with custom figure size and improved aesthetics
        plt.figure(figsize=(14, 7))  # Increase figure size for better readability
        df.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Example plot with improved color scheme

        # Add a title and labels with increased font sizes
        plt.title('Your Chart Title', fontsize=18, pad=20)  # Add padding for better spacing
        plt.xlabel('X-axis Label', fontsize=14)
        plt.ylabel('Y-axis Label', fontsize=14)

        # Customize ticks and rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        # Add gridlines for better readability
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Show the plot with tight layout for better spacing
        plt.tight_layout()
        plt.show()
        ```

        CHART INSTRUCTIONS: {chart_instructions}
        INPUT DATA: {data}
        
        Important Notes on creating the chart code:
        - Ensure the chart is visually appealing with a larger figure size (e.g., 14x7), gridlines, and clear font sizes.
        - Add a margin using `plt.tight_layout()` to ensure all elements are well-spaced and readable.
        - Ensure proper alignment of x-axis labels by rotating them if necessary.
        - Make sure to use `plt.show()` to display the chart.
        """,
        input_variables=["chart_instructions", "data"]
    )

    tools = [python_repl]

    chart_generator = prompt_chart_generator.partial(tool_names=", ".join([tool.name for tool in tools])) | llm.bind_tools(tools)
    
    # * Summarizer
    
    summarizer_prompt = PromptTemplate(
        template="""
        You are an expert in summarizing data analysis results, focusing on providing key insights and delivering explanations at the highest possible level for business professionals to easily understand. 
        Your goal is to help the business comprehend the most critical insights from the analysis. 
        Be concise in your explanation and focus on explaining the key findings clearly. 
        Avoid using bullet points and markdown headers.

        You have access to the company's database, which may contain tables related to sales, customers, products, transactions, and other business-relevant information. 
        You also have the ability to write SQL, produce data tables, and create charts for analysis.

        You are given the following details regarding the analysis performed:

        - user_question: The initial question asked by the user regarding the data.
        - formatted_user_question_sql_only: A processed version of the user question provided to the SQL expert
        - sql_query: The SQL query that was created to answer the user's question.
        - data: The results of the sql query when run on the database

        ANALYSIS RESULTS FOR SUMMARIZATION: {results}
        """,
        input_variables=["results"]
    )

    summarizer = summarizer_prompt | llm | StrOutputParser()


    # * LANGGRAPH
    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        user_question: str
        formatted_user_question_sql_only: str
        sql_query : str
        data: dict
        routing_preprocessor_decision: str
        chart_generator_instructions: str
        chart_code: str
        chart_plotly_json: dict
        chart_plotly_error: bool
        summary: str
        
    def preprocess_routing(state):
        print("---ROUTER---")
        question = state.get("user_question")

        
        # Chart Routing and SQL Prep
        response = routing_preprocessor.invoke({"initial_question": question})
        
        time.sleep(1)

        formatted_user_question_sql_only = response['formatted_user_question_sql_only']
        
        routing_preprocessor_decision = response['routing_preprocessor_decision']
        
        return {
            "formatted_user_question_sql_only": formatted_user_question_sql_only,
            "routing_preprocessor_decision": routing_preprocessor_decision
        }

    def generate_sql(state):
        print("---GENERATE SQL---")
        question = state.get("formatted_user_question_sql_only")
        
        # Handle case when formatted_user_question_sql_only is None:
        if question is None:
            question = state.get("user_question")
        
        time.sleep(1)
        
        # Generate SQL
        sql_query = sql_generator.invoke({"question": question})
        
        # Add print statement to check the output from the SQL generator
        #print(f"SQL Query Generated: {sql_query}")

        return {"sql_query": sql_query}

    def convert_dataframe(state):
        print("---CONVERT DATA FRAME---")

        sql_query = state.get("sql_query")
        
        time.sleep(1)
        
        # Remove trailing ' that gpt-3.5-turbo sometimes leaves
        sql_query = sql_query.rstrip("'")

        # Add this line to print the SQL query before execution
        #print(f"Generated SQL Query: {sql_query}")
        
        df = pd.read_sql(sql_query, conn)
        
        return {"data": dict(df)}

    def decide_chart_or_table(state):
        print("---DECIDE CHART OR TABLE---")
        return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"

    def instruct_chart_generator(state):
        print("---INSTRUCT CHART GENERATOR---")
        
        question = state.get("user_question")
        
        data = state.get("data")
        
        time.sleep(1)
        
        chart_generator_instructions = chart_instructor.invoke({"question": question, "data": data})
        
        return {"chart_generator_instructions": chart_generator_instructions}

    def generate_chart(state):
        print("---GENERATE CHART---")
        
        chart_instructions = state.get("chart_generator_instructions")
        
        data = state.get("data")
        
        time.sleep(1)
        
        response = chart_generator.invoke({"chart_instructions": chart_instructions, "data": data})
        
        # Fix - if invalid tool calls
        try:
            code = dict(response)['tool_calls'][0]['args']['code']
        except: 
            code = dict(response)['invalid_tool_calls'][0]['args']
        
        #repl.run(code)
            
        return {
            "chart_code": code, 
            #"chart_plotly_json": result, 
        }

    def summarize_results(state):
        print("---SUMMARIZE RESULTS----")
        
        result = summarizer.invoke({"results": dict(state)})
        time.sleep(1)
        
        return {"summary": result}
            
    def state_printer(state):
        """print the state"""
        print("---STATE PRINTER---")
        #print(f"User Question: {state['user_question']}")
        #print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
        #print(f"SQL Query: \n{state['sql_query']}\n")
        #print(f"Data: \n{pd.DataFrame(state['data'])}\n")
        #print(f"Chart or Table: {state['routing_preprocessor_decision']}")
        
        #if state['routing_preprocessor_decision'] == "chart":
            #print(f"Chart Code: \n{pprint(state['chart_code'])}")
            #repl.run(state['chart_code'])

        #summary = state.get('summary', 'No summary available')
        #print(f"Summarize Results: {summary}")

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("convert_dataframe", convert_dataframe)
    workflow.add_node("instruct_chart_generator", instruct_chart_generator)
    workflow.add_node("generate_chart", generate_chart)
    workflow.add_node("summarizer", summarize_results)
    workflow.add_node("state_printer", state_printer)

    workflow.set_entry_point("preprocess_routing")
    workflow.add_edge("preprocess_routing", "generate_sql")
    workflow.add_edge("generate_sql", "convert_dataframe")

    workflow.add_conditional_edges(
        "convert_dataframe", 
        decide_chart_or_table,
        {
            # Result : Step Name To Go To
            "chart":"instruct_chart_generator", # Path Chart
            "table":"summarizer" # Summarizer
        }
    )

    workflow.add_edge("instruct_chart_generator", "generate_chart")
    workflow.add_edge("generate_chart", "summarizer")
    workflow.add_edge("summarizer", "state_printer")
    workflow.add_edge("state_printer", END)

    app = workflow.compile()
    
    return app
