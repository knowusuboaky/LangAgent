# LangAgent/trading_team/agents/sentiments_analyst.py


# * LIBRARIES

import os
import re
import json
import math
import traceback
from typing import Annotated, Any, Dict, List, Sequence, TypedDict, Callable, Dict, Type, Optional

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, BaseOutputParser
from langchain.prompts import PromptTemplate
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, END
import warnings
import yaml

warnings.filterwarnings("ignore")

# Langgraph Workflow Function
def create_coding_agent_graph(
    GraphState: Type,
    node_functions: Dict[str, Callable],
    recommended_steps_node_name: str,
    create_code_node_name: str,
    execute_code_node_name: str,
    fix_code_node_name: str,
    error_key: str = "error",
    max_retries_key: str = "max_retries",
    retry_count_key: str = "retry_count",
    checkpointer: Optional[Callable] = None,
):
    """
    Creates a state graph workflow for the coding agent.

    Parameters:
        GraphState: The type representing the graph state structure.
        node_functions: A dictionary mapping node names to their corresponding functions.
        recommended_steps_node_name: The name of the initial node for recommended steps.
        create_code_node_name: The name of the node for creating code.
        execute_code_node_name: The name of the node for executing code.
        fix_code_node_name: The name of the node for fixing code issues.
        error_key: Key in the state indicating the presence of an error.
        max_retries_key: Key for the maximum number of retries allowed.
        retry_count_key: Key for tracking retry attempts.
        checkpointer: Optional; a callable for checkpointing the workflow.

    Returns:
        A compiled state graph workflow.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node(recommended_steps_node_name, node_functions[recommended_steps_node_name])
    workflow.add_node(fix_code_node_name, node_functions[fix_code_node_name])
    workflow.add_node(create_code_node_name, node_functions[create_code_node_name])
    workflow.add_node(execute_code_node_name, node_functions[execute_code_node_name])
    workflow.add_node("_end_", lambda state: state)  # Define the END node

    # Set entry point
    workflow.set_entry_point(recommended_steps_node_name)

    # Define a helper to check if we have an error & can still retry
    def error_and_can_retry(state):
        return (
            state.get(error_key) is not None
            and state.get(retry_count_key) is not None
            and state.get(max_retries_key) is not None
            and state[retry_count_key] < state[max_retries_key]
        )
    
    # Add edges
    workflow.add_edge(recommended_steps_node_name, fix_code_node_name)

    workflow.add_conditional_edges(
        fix_code_node_name,
        lambda s: "fix_code" if error_and_can_retry(s) else "_end_",
        {
            "fix_code": recommended_steps_node_name,
            "_end_": END,
        },
    )  # Loop back to parse_user_instructions


    workflow.add_edge(fix_code_node_name, create_code_node_name)  # Directly to create_code_node_name
    workflow.add_edge(create_code_node_name, execute_code_node_name)
    workflow.add_edge(execute_code_node_name, "_end_")

    # Compile the workflow
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()

# Python Parser for output standardization  
class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):        
        def extract_python_code(text):
            python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
            if python_code_match:
                python_code = python_code_match.group(1).strip()
                return python_code
            else:
                python_code_match = re.search(r"python(.*?)'", text, re.DOTALL)
                if python_code_match:
                    python_code = python_code_match.group(1).strip()
                    return python_code
                else:
                    return None
        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        else:
            # Assume ```sql wasn't used
            return text

# Fetch company news from the API
def get_company_news(
    ticker: str,
    api_key: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch company news from the Financial Datasets API.

    Args:
        ticker (str): The stock ticker (e.g., "AAPL").
        api_key (str): Your API key for authentication.
        end_date (str): End date for the news in 'YYYY-MM-DD' format.
        start_date (Optional[str]): Start date for the news in 'YYYY-MM-DD' format.
        limit (int): Number of news items to fetch (max 1000).

    Returns:
        List[Dict]: A list of news items in dictionary format.
    """
    # Add your API key to the headers
    headers = {"X-API-KEY": api_key}

    # Build the query URL
    url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={end_date}"
    if start_date:
        url += f"&start_date={start_date}"
    url += f"&limit={limit}"

    print(f"Fetching news from URL: {url}")  # Debugging print

    # Make the API request
    response = requests.get(url, headers=headers)

    # Check for a successful response
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

    # Parse and return the news from the response
    news = response.json().get("news", [])
    return news

# Fetch insider trades metrics from the API.
def get_insider_trades(
    api_key: str,
    ticker: str,
    end_date: str,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch insider trades from API."""

    # fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key
    
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_lte={end_date}"
        f"&limit={limit}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    insider_trades = data.get("insider_trades")
    if not insider_trades:
        raise ValueError("No insider trades returned")
   
    return insider_trades[:limit]



# Setup
def make_sentiments_agent(
    model,
    api_key: Optional[str] = None,
):
    """
    Creates a sentiment analysis agent to evaluate a company's market sentiment and generates trading signals.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use for reasoning and explanations.
    api_key : str, optional
        API key for accessing financial datasets. Defaults to None.
 
    Returns
    -------
    function
        The sentiment agent function.
    """

    # Define GraphState for the agent
    class GraphState(TypedDict):
        """
        Defines the state structure for the valuation analysis agent.
        """

        # Messages and User Instructions
        messages: Annotated[Sequence[BaseMessage], operator.add]  # Messages for LLMs
        user_instructions: str  # Original instructions
        adjusted_instructions: Optional[str]  # Instructions adjusted by retries
        relative_time: Optional[str]  # Relative time from User's input instructions

        # Analysis Output
        reasoning: Optional[str]  # Reasoning or context built by the agent
        signal: Optional[str]  # Signal output from the analysis (e.g., Very Bullish, Bearish)
        confidence: Optional[float]  # Confidence score for the signal
        error: Optional[str]  # Error messages, if any

        # API and External Service Information
        api_key: Optional[str]  # API key for accessing external financial services

        # Parsed User Instructions
        tickers: Optional[List[str]]  # List of tickers to analyze
        period: Optional[str]  # Reporting period (e.g., ttm, quarterly, annual)
        start_date: Optional[str]  # Start date of the analysis period
        end_date: Optional[str]  # End date of the analysis period
        limit: Optional[int]  # Number of periods to fetch

        # Financial Data Retrieved from API
        insider_trades: Optional[List[Dict[str, any]]]  # List of insiders_trades dictionaries
        insider_trades_data: Optional[Dict[str, List[Dict[str, any]]]]  # Dictionary of insiders trades for each ticker

        # Sensitivity Analysis
        sensitivity_ranges: Optional[Dict[str, List[float]]]  # Ranges for sensitivity analysis

        # Validation Flags
        is_valid_ticker: Optional[bool]  # Whether the ticker is valid and matches the user's input
        are_dates_valid: Optional[bool]  # Whether start_date and end_date are valid and consistent

        # Validation for Output and Cross-Checks
        parsed_ticker_matches_user: Optional[bool]  # If parsed ticker matches the user's question
        parsed_dates_match_user: Optional[bool]  # If parsed dates match the user's question
        parsed_instruction_valid: Optional[bool]  # If the parsed instruction is considered valid

        # Additional Execution Information
        retry_count: Optional[int]  # Count of retries for fixing instructions or errors
        code_snippet: Optional[str]  # Snippet of code for fixing parse instructions

        # Analysis Results
        analysis_results: Optional[Dict[str, Dict[str, Any]]]  # Detailed analysis for each ticker
        overall_results: Optional[Dict[str, Dict[str, Any]]]  # Signal and confidence for each ticker
        reasoning_reports: Optional[Dict[str, str]]  # Detailed reasoning for each ticker
        reasoning_dict: Optional[Dict[str, Dict[str, Dict[str, Any]]]]  # Structured reasoning for each ticker
        overall_summary: Optional[str]  # Overall textual summary of the analysis
        report: Optional[Dict[str, str]]  # Complete combined report for all tickers



    def parse_user_instructions(state: Dict) -> Dict:
        """
        Parses user instructions from the state and updates it with actionable details.
        Handles relative time frames like 'last 3 years'.

        Parameters:
            state (Dict): The current state containing user instructions.

        Returns:
            Dict: Updated state with parsed details (ticker, period, start_date, end_date).
        """
        print("    * PARSE USER INSTRUCTIONS")

        try:
            # Extract user instructions and relative_time from the state
            #user_instructions = state.get("user_instructions")
            # Use adjusted_instructions if available
            user_instructions = state.get("adjusted_instructions", state.get("user_instructions"))
            relative_time = state.get("relative_time")  # Expected as an input
            if not user_instructions:
                raise ValueError("User instructions are missing in the state.")

            # Automatically extract relative time from user instructions
            relative_patterns = {
                r"last (\d+) year[s]?": lambda match: f"year_{match.group(1)}",
                r"last one year": lambda _: "year_1",
                r"last (\d+) month[s]?": lambda match: f"month_{match.group(1)}",
                r"last one month": lambda _: "month_1",
            }

            relative_time = None
            for pattern, resolver in relative_patterns.items():
                match = re.search(pattern, user_instructions, re.IGNORECASE)
                if match:
                    relative_time = resolver(match)
                    break

            # Prompt template for parsing instructions
            parse_prompt = PromptTemplate(
                template="""
                You are an expert financial analysis assistant, skilled in interpreting user instructions and extracting actionable details.

                Based on the provided user instructions, extract the following details in JSON format:
                - **tickers**: A list of stock ticker symbols. If the user provides multiple companies or tickers, identify and return all as a list (e.g., "Apple, Microsoft" → ["AAPL", "MSFT"], "Apple (AAPL) and Microsoft (MSFT)" → ["AAPL", "MSFT"] or "AAPL and MSFT" → ["AAPL", "MSFT"]). Ensure accuracy.
                - **end_date**: The end date for the analysis period, formatted as "YYYY-MM-DD."
                    - Extract if the user specifies a date range like:
                    - "from 2024-05-01 to 2024-12-31" → **end_date** = "2024-12-31".
                    - "to May 2024" → **end_date** = "2024-05-31".
                    - "to January 2023" → **end_date** = "2023-01-31".
                    - "to 2023" → **start_date** = "2023-01-31".
                    - If the user specifies phrases like "to date" or "to today," interpret it as the current date.
                    - If the user specifies phrases 
                    like "last year", "last one year", "last 1 year", "last two years", "last 2 years", use the format "{relative_time}" (e.g., "year_1" for "last 1 year" or "last year" or "last one year")
                    - If the user specifies phrases like "last month", "last one month", "last 1 month", "last two months", "last 2 months", use the format "{relative_time}" (e.g., "month_1" for "last 1 month" or "last month" or "last one month")
                    - If no end date is explicitly provided, use the keyword "latest."
                - **start_date**: The start date for the analysis period, formatted as "YYYY-MM-DD."
                    - Extract if the user specifies a date range like:
                    - "from 2024-05-01 to 2024-12-31" → **start_date** = "2024-05-01".
                    - "from May 2024 to date" → **start_date** = "2024-05-01".
                    - "from January 2023" → **start_date** = "2023-01-01".
                    - "from 2023" → **start_date** = "2023-01-01".
                    - If not explicitly mentioned, default to `null`.
                - **period**: The reporting period specified by the user (e.g., "ttm," "annual," or "quarterly").
                    - If not explicitly mentioned:
                    - Use "quarterly" if the range between `start_date` and `end_date` is one year or more.
                    - Use "ttm" if the range is less than a year.
                - **limit**: The number of periods or statements to fetch.
                    - If explicitly mentioned, extract from instructions like "last 3 statements" or "most recent 5 quarters."
                    - If not explicitly mentioned, calculate dynamically based on the `start_date`, `end_date`, and `period`.

                Ensure the output is in the following JSON format:
                {{
                    "ticker": <string>,
                    "start_date": <string or null>,
                    "end_date": <string>,
                    "period": <string>,
                    "limit": <integer>
                }}

                ### Example 1:
                --------------
                Input:
                "Analyze my portfolio of $10,000 in cash, with $3,000 in Apple (AAPL) and $4,000 in Microsoft (MSFT), focusing on long-term investment goals. Using the last two statements, evaluate the portfolio's performance and conduct a TTM analysis of AAPL and MSFT from October 1, 2024, to December 31, 2024, to inform long-term strategies."

                Output:
                {{
                    "ticker": ["AAPL", "MSFT"],
                    "start_date": "2024-10-01",
                    "end_date": "2024-12-31",
                    "period": "ttm",
                    "limit": 2,
                }}

                ### Example 2:
                --------------
                Input:
                "Analyze Apple and Tesla (AAPL, TSLA) from January 1, 2023, to December 31, 2023. Provide quarterly performance analysis for the portfolio."

                Output:
                {{
                    "ticker": ["AAPL", "TSLA"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "period": "quarterly",
                    "limit": 4
                }}

                ### Example 3:
                --------------
                Input:
                "Provide a trailing twelve months (TTM) analysis for Amazon (AMZN) and Netflix (NFLX) for the last one year."

                Output:
                {{
                    "ticker": ["AMZN", "NFLX"],
                    "start_date": None,
                    "end_date": "year_1",
                    "period": "ttm",
                    "limit": 2
                }}

                ### Example 4:
                --------------
                Input:
                "Analyze Tesla (TSLA) for the last 6 months."

                Output:
                {{
                    "ticker": ["TSLA"],
                    "start_date": None,
                    "end_date": "month_6",
                    "period": "ttm",
                    "limit": 2
                }}



                ### User Instructions:
                {user_instructions}

                Ensure the JSON is valid and complete.
                """,
                input_variables=["user_instructions", "relative_time"]
            )

            # Parse the user instructions
            try:
                parsed_preprocessor = parse_prompt | model | JsonOutputParser()
                details = parsed_preprocessor.invoke({"user_instructions": user_instructions, "relative_time": relative_time})
            except Exception as e:
                raise ValueError(f"Failed to parse user instructions: {str(e)}")

            # Extract parsed details
            tickers = details.get("tickers", [])
            start_date = details.get("start_date")  # Can be None
            end_date = details.get("end_date", "latest")  # Defaults to 'latest' if not provided
            period = details.get("period")  # Use parsed period if provided
            limit = details.get("limit", 1)  # Default limit

            # Log details for debugging
            print(f"Parsed Ticker: {tickers}")
            print(f"Start Date: {start_date}")
            print(f"End Date: {end_date}")
            print(f"Period: {period}")
            print(f"Limit: {limit}")

            # Fallback or verification step: Extract tickers using regex if not found or for validation
            if not tickers:
                ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")  # Matches stock tickers like AAPL, TSLA
                tickers = re.findall(ticker_pattern, user_instructions)
                print(f"Extracted tickers via regex: {tickers}")

            if not tickers:
                raise ValueError("No valid tickers found in the user instructions.")
                
            # Patterns for resolving dates
            date_patterns = {
                r"year_(\d+)": lambda match: (datetime.now() - timedelta(days=int(match.group(1)) * 365)).strftime("%Y-%m-%d"),
                r"month_(\d+)": lambda match: (datetime.now() - timedelta(days=int(match.group(1)) * 30)).strftime("%Y-%m-%d"),
                r"(\d+)_startmonth": lambda match: f"{(datetime.now().year - 1):04d}-{int(match.group(1)):02d}-01" 
                if 1 <= int(match.group(1)) <= 12 else ValueError("Invalid month for startmonth"),
                r"(\d+)_endmonth": lambda match: f"{(datetime.now().year - 1):04d}-{int(match.group(1)):02d}-{(datetime.now().replace(month=int(match.group(1)), day=1) + timedelta(days=31)).replace(day=1) - timedelta(days=1):%d}" 
                if 1 <= int(match.group(1)) <= 12 else ValueError("Invalid month for endmonth"),
            }

            if relative_time: 
                # Set default end_date to today
                start_date = None
                for pattern, resolver in date_patterns.items():
                    match = re.search(pattern, relative_time, re.IGNORECASE)
                    if match:
                        start_date = resolver(match)
                        break
        
                end_date = (datetime.now()).strftime("%Y-%m-%d")
            
            # Resolve 'latest' for end_date
            if end_date == "latest":
                end_date = (datetime.now()).strftime('%Y-%m-%d')  # Today's date

            # Resolve relative time
            if relative_time:
                if "year_" in relative_time:
                    num_years = int(relative_time.split("_")[1])
                    start_date = (datetime.now() - timedelta(days=num_years * 365)).strftime('%Y-%m-%d')
                elif "month_" in relative_time:
                    num_months = int(relative_time.split("_")[1])
                    start_date = (datetime.now() - timedelta(days=num_months * 30)).strftime('%Y-%m-%d')

            # Determine default period if not explicitly mentioned
            if start_date and not period:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")

                # Calculate total months between start and end
                total_months = (end.year - start.year) * 12 + (end.month - start.month)

                # Use "quarterly" for ranges of a year or more, otherwise use "ttm"
                if total_months >= 12:
                    period = "quarterly"
                else:
                    period = "ttm"


            # Log details for debugging
            print(f"Parsed Ticker: {tickers}")
            print(f"Start Date: {start_date}")
            print(f"End Date: {end_date}")
            print(f"Period: {period}")
            print(f"Limit: {limit}")

            # Update state with parsed details
            state.update({
                "tickers": tickers,
                "period": period,
                "start_date": start_date,
                "limit": limit,
                "end_date": end_date,
            })

        
        except ValueError as ve:
            error_message = (
                f"ValueError: {str(ve)}. We encountered an issue processing your input. Please provide clear and complete instructions, including: "
                "a valid stock ticker (e.g., 'AAPL' for Apple), a specific date range (e.g., 'from YYYY-MM-DD to YYYY-MM-DD'), "
                "an end date (e.g., 'to date' or a specific date like 'YYYY-MM-DD'), or a relative time frame (e.g., 'last 3 years' or 'last 6 months'). "
                "Additionally, specify the number of documents or periods to analyze if applicable (e.g., 'limit of 5 quarterly statements'). "
                "This will help ensure accurate and efficient processing of your request."
            )
            state["reasoning"] = error_message
            print(error_message)

        except Exception as e:
            # Log detailed reasoning and update the state with error
            error_message = (
                f"We encountered an issue processing your input. Please provide clear and complete instructions, including: "
                "a valid stock ticker (e.g., 'AAPL' for Apple), a specific date range (e.g., 'from YYYY-MM-DD to YYYY-MM-DD'), "
                "an end date (e.g., 'to date' or a specific date like 'YYYY-MM-DD'), or a relative time frame (e.g., 'last 3 years' or 'last 6 months'). "
                "Additionally, specify the number of documents or periods to analyze if applicable (e.g., 'limit of 5 quarterly statements'). "
                "This will help ensure accurate and efficient processing of your request."
            )
            state["reasoning"] = error_message
            print(error_message)
            raise  
           
        return state    
  

    def fix_parse_instructions_code(state: GraphState):
        """
        Fixes the parse_user_instructions() function based on errors or logical issues in the state.
        Ensures to only return the function definition for parse_user_instructions() if:
        - The answer is incorrect.
        - The end date is before the start date.
        - There is a mismatch between the extracted dates or ticker and the user's question.
        - An error is detected.

        Passes without changes if:
        - `end_date` exists but `start_date` does not (valid case).
        - The extracted information matches the user's question.
        """
        print("    * FIX PARSE INSTRUCTIONS CODE")


        # Keys for state management
        code_snippet_key = "code_snippet"
        error_key = "error"
        retry_count_key = "retry_count"
        retry_limit = 3
        
        #print("      retry_count:" + str(state.get(retry_count_key)))

        # Ensure retry count does not exceed limits
        retry_count = state.get(retry_count_key, 0)
        if retry_count >= retry_limit:
            raise Exception("Maximum retries reached. Unable to fix the code.")

        # Extract state details
        code_snippet = state.get(code_snippet_key, "")
        error_message = state.get(error_key, "")
        start_date = state.get("start_date")
        end_date = state.get("end_date")
        ticker = state.get("ticker")
        user_question = state.get("user_instructions", "")

        # Check if there's an error or logical inconsistency
        answer_incorrect = error_message or (start_date and end_date and start_date > end_date)

        # Validate if the dates and ticker correspond to the user's question
        if user_question:
            import re

            # Extract patterns for validation
            ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")  # Matches stock tickers like AAPL, TSLA
            date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")  # Matches dates in YYYY-MM-DD format

            # Check if ticker matches the user's input
            user_ticker = ticker_pattern.search(user_question)
            ticker_mismatch = user_ticker and ticker != user_ticker.group(0)

            # Check if dates align with the user's input
            user_dates = date_pattern.findall(user_question)
            if user_dates:
                user_start_date = user_dates[0]
                user_end_date = user_dates[1] if len(user_dates) > 1 else None
                date_mismatch = (
                    (start_date and user_start_date and start_date != user_start_date) or
                    (end_date and user_end_date and end_date != user_end_date)
                )
            else:
                date_mismatch = False

            # If there's a mismatch, mark as incorrect
            answer_incorrect = answer_incorrect or ticker_mismatch or date_mismatch

        if not answer_incorrect:
            # If no issues detected, do not attempt to fix
            return state

        # Prompt template for LLM
        prompt_template = """
        You are a Python expert. The following function `parse_user_instructions()` is broken and needs fixing.
        Please fix the function and ensure the following conditions are met:
        1. The function accurately parses the user's question and extracts:
            - `ticker`: Stock ticker symbol.
            - `start_date` and `end_date`: Date range for analysis.
        2. The extracted `ticker` and dates (`start_date`, `end_date`) must match the user's question.
        3. The function ensures `end_date` is not earlier than `start_date`.
        4. If `end_date` exists but `start_date` does not, treat it as valid.
        5. If there are errors or inconsistencies, they are resolved in the fixed function.

        Return ONLY the fixed `parse_user_instructions()` function definition.

        Current Function:
        {code_snippet}

        Error Message (if any):
        {error}
        """

        # Format the prompt
        prompt = prompt_template.format(
            code_snippet=code_snippet,
            error=error_message
        )

        # Generate the fix using the model
        try:
            response = model | PythonOutputParser().invoke(prompt)
            state["adjusted_instructions"] = response.strip()  # Write to adjusted_instructions

            # Update the state with the fixed code and reset error
            state[retry_count_key] = retry_count + 1  # Increment retry count
            state[error_key] = None  # Clear error
        except Exception as e:
            state[error_key] = f"Error fixing code: {str(e)}"
        return state

                
    def fetch_insider_trades_data(state: GraphState):
        """
        Fetch financial data based on the user's instructions.
        """
        print("    * FETCH INSIDER TRADES DATA")

        tickers = state.get("tickers")
        period = state.get("period", "ttm")
        limit = state.get("limit")
        end_date = state.get("end_date")
        api_key = state.get("financial_data_api_key")  # Access API key from state

        
        limit = 1000 # Maximum limit for insider traders

        insider_trades_data = {}
        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")

                # Get the insider trades
                insider_trades = get_insider_trades(
                    api_key=api_key,
                    ticker=ticker,
                    end_date=end_date,
                    limit=limit,
                )

                insider_trades_data[ticker] = insider_trades
                print(f"Data for {ticker}: {insider_trades}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                insider_trades_data[ticker] = None

        state["insider_trades_data"] = insider_trades_data
        
        print("\nAll fetched data:")
        for ticker, data in insider_trades_data.items():
            print(f"{ticker}: {data}")
        
        return state
        try: 
            state["insider_trades"] = insider_trades
            return state
        except Exception as e:
            state["error"] = f"Error fetching insider trades data: {str(e)}"
            #print(f"Error: {state['error']}")
            return state



    def generate_analysis(
        state: GraphState
    ):
        """
        Generate a sentiment analysis based on insiders trading.

        Returns:
            dict: Updated state with overall signal, confidence, and reasoning report.
        """

        print("    * GENERATE ANALYSIS")

        insider_trades_data = state.get("insider_trades_data")
        api_key = state.get("api_key")
        end_date = state.get("end_date")
        start_date = state.get("start_date", None)

        reasoning_reports = {}
        overall_results = {}
        reasoning_dict = {}  # Initialize reasoning_dict for structured analysis


        for ticker, insider_trades in insider_trades_data.items():
            if insider_trades is None:
                #reasoning_reports[ticker] = f"Error: No data available for {ticker}."
                #overall_results[ticker] = {"signal": None, "confidence": 0}
                continue

            try:
                # Get the signals from the insider trades
                transaction_shares = pd.Series(
                    [t.get("transaction_shares") for t in insider_trades]
                ).dropna()
                bearish_condition = transaction_shares < 0
                insider_signals = np.where(bearish_condition, "bearish", "bullish").tolist()

                # Count signals
                insider_bullish_signals = insider_signals.count("bullish")
                insider_bearish_signals = insider_signals.count("bearish")

                # Get the company news
                company_news = get_company_news(api_key=api_key, ticker = ticker, start_date = start_date, end_date = end_date, limit=100)

                # Get the sentiment from the company news
                sentiment = pd.Series([n["sentiment"] for n in company_news]).dropna()
                news_signals = np.where(
                    sentiment == "negative", "bearish", 
                    np.where(sentiment == "positive", "bullish", "neutral")
                ).tolist()

                # Combine signals from both sources with weights
                insider_weight = 0.3
                news_weight = 0.7

                # Calculate weighted signal counts
                weighted_bullish_signals = (
                    insider_signals.count("bullish") * insider_weight +
                    news_signals.count("bullish") * news_weight
                )
                weighted_bearish_signals = (
                    insider_signals.count("bearish") * insider_weight +
                    news_signals.count("bearish") * news_weight
                )

                total_signals = weighted_bullish_signals + weighted_bearish_signals

                # Determine overall signal
                if total_signals == 0:
                    overall_signal = "Neutral"
                    confidence = 0
                else:
                    bullish_percentage = weighted_bullish_signals / total_signals
                    bearish_percentage = weighted_bearish_signals / total_signals


                    if bullish_percentage > 0.75:
                        overall_signal = "Very Bullish"
                    elif bearish_percentage > 0.75:
                        overall_signal = "Very Bearish"
                    elif weighted_bullish_signals > weighted_bearish_signals:
                        overall_signal = "Bullish"
                    elif weighted_bearish_signals > weighted_bullish_signals:
                        overall_signal = "Bearish"
                    else:
                        overall_signal = "Neutral"

                    # Calculate confidence
                    confidence = round(max(bullish_percentage, bearish_percentage) * 100, 2)


                # Generate reasoning report
                #ticker = state.get("ticker", "Unknown Ticker")  # Fallback to "Unknown Ticker" if not provided

                # Generate reasoning report
                reasoning_report = f"""## **Sentiment Analysis Report for {ticker}**\n\n"""
                reasoning_report += f"""### **Overall Signal**: {overall_signal}\n"""
                reasoning_report += f"""### **Confidence**: {confidence}%\n\n"""
                reasoning_report += f"""### **Insider Trades Details**\n"""
                reasoning_report += f"""- Insider Bullish Signals: {insider_bullish_signals}\n"""
                reasoning_report += f"""- Insider Bearish Signals: {insider_bearish_signals}\n\n"""
                reasoning_report += f"""### **Company News Details**\n"""
                reasoning_report += f"""- News Bullish Signals: {news_signals.count('bullish')}\n"""
                reasoning_report += f"""- News Bearish Signals: {news_signals.count('bearish')}\n\n"""
                reasoning_report += f"""### **Analysis Details**\n"""
                reasoning_report += f"""- Total Signals Analyzed: {total_signals}\n"""
                reasoning_report += f"""- Weighted Bullish Signals: {weighted_bullish_signals}\n"""
                reasoning_report += f"""- Weighted Bearish Signals: {weighted_bearish_signals}\n"""
                reasoning_report += f"""- Confidence is derived from the proportion of dominant signals.\n"""


                # Store the analysis for the ticker
                overall_results[ticker] = {"signal": overall_signal, "confidence": confidence}
                reasoning_reports[ticker] = reasoning_report

                # Add structured data to reasoning_dict
                reasoning_dict[ticker] = {
                        "Overall Signal": overall_signal,
                        "Confidence": confidence,
                        "Insider Trades Details": {
                            "Bullish Signals": insider_bullish_signals,
                            "Bearish Signals": insider_bearish_signals,
                        },
                        "Company News Details": {
                            "Bullish Signals": news_signals.count("bullish"),
                            "Bearish Signals": news_signals.count("bearish"),
                        },
                        "Analysis Details": {
                            "Total Signals Analyzed": total_signals,
                            "Weighted Bullish Signals": weighted_bullish_signals,
                            "Weighted Bearish Signals": weighted_bearish_signals,
                        },
                    }
                
                
            except Exception as e:
                print(f"Error in generate_analysis for {ticker}: {e}")
                continue

        # Update state with results for all tickers
        full_report = ""
        for ticker, analysis_report in reasoning_reports.items():
            full_report += analysis_report + "\n" + "-" * 80 + "\n"


        state["report"] = full_report  # Save the consolidated report in the state
        state["overall_results"] = overall_results
        state["reasoning_reports"] = reasoning_reports
        state["reasoning_reports"] = reasoning_reports
        state["reasoning_dict"] = reasoning_dict
        return state


    # Define the graph
    node_functions = {
        "parse_user_instructions": parse_user_instructions,
        "fix_parse_instructions_code": fix_parse_instructions_code,  # Ensure this node comes after parsing
        "fetch_insider_trades_data": fetch_insider_trades_data,
        "generate_analysis": generate_analysis,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="parse_user_instructions",
        create_code_node_name="fetch_insider_trades_data",
        execute_code_node_name="generate_analysis",
        fix_code_node_name="fix_parse_instructions_code",  # Comes after parsing
    )

    return app