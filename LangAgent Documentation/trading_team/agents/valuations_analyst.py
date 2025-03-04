# LangAgent/trading_team/agents/valuations_analyst.py


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

# Fetch financial metrics from the API.
def get_financial_metrics(
    api_key: str,
    ticker: str,
    report_period: str,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial metrics from the API."""
    headers = {"X-API-KEY": os.environ.get("api_key")}

    
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    financial_metrics = data.get("financial_metrics")
    if not financial_metrics:
        raise ValueError("No financial metrics returned")
    return financial_metrics

# Fetch cash flow statements from the API.
def search_line_items(
    api_key: str,
    ticker: str,
    report_period: str,
    line_items: List[str],
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch cash flow statements from the API."""
    headers = {"X-API-KEY": os.environ.get("api_key")}


    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "report_period": report_period,
        "period": period,
        "limit": limit
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    search_results = data.get("search_results")
    if not search_results:
        raise ValueError("No search results returned")
    return search_results

# Fetch market cap from the API.
def get_market_cap(
    api_key: str,
    ticker: str,
) -> List[Dict[str, Any]]:
    """Fetch market cap from the API."""
    headers = {"X-API-KEY": os.environ.get("api_key")}


    url = (
        f'https://api.financialdatasets.ai/company/facts'
        f'?ticker={ticker}'
    )

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    company_facts = data.get('company_facts')
    if not company_facts:
        raise ValueError("No company facts returned")
    return company_facts.get('market_cap')

def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income
                    + Depreciation/Amortization
                    - Capital Expenditures
                    - Working Capital Changes

    Args:
        net_income: Annual net income
        depreciation: Annual depreciation and amortization
        capex: Annual capital expenditures
        working_capital_change: Annual change in working capital
        growth_rate: Expected growth rate
        required_return: Required rate of return (Buffett typically uses 15%)
        margin_of_safety: Margin of safety to apply to final value
        num_years: Number of years to project

    Returns:
        float: Intrinsic value with margin of safety
    """
    if not all(
        [
            isinstance(x, (int, float))
            for x in [net_income, depreciation, capex, working_capital_change]
        ]
    ):
        return 0

    # Calculate initial owner earnings
    owner_earnings = net_income + depreciation - capex - working_capital_change

    if owner_earnings <= 0:
        return 0

    # Project future owner earnings
    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)

    # Calculate terminal value (using perpetuity growth formula)
    terminal_growth = min(growth_rate, 0.03)  # Cap terminal growth at 3%
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (
        required_return - terminal_growth
    )
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

    # Sum all values and apply margin of safety
    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

    return value_with_safety_margin

def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    Use this function to calculate the intrinsic value of a stock.
    """
    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    terminal_value = (
        cash_flows[-1]
        * (1 + terminal_growth_rate)
        / (discount_rate - terminal_growth_rate)
    )
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value

def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    return current_working_capital - previous_working_capital



# Setup
def make_valuations_agent(
    model,
    api_key: Optional[str] = None,
    dcf_assumptions: Optional[Dict[str, float]] = None,
    owner_earnings_assumptions: Optional[Dict[str, float]] = None,
    sensitivity_ranges: Optional[Dict[str, List[float]]] = None
):
    """
    Creates a valuation analysis agent to evaluate a company's financial metrics and generate trading signals.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use for reasoning and explanations.
    api_key : str, optional
        API key for accessing financial datasets. Defaults to None.
    dcf_assumptions : Optional[Dict[str, float]], optional
        Assumptions for DCF valuation. Defaults to:
            - discount_rate (float): Discount rate for DCF (default 0.10).
            - terminal_growth_rate (float): Terminal growth rate for DCF (default 0.03).
            - num_years (int): Number of projection years for DCF (default 5).
    owner_earnings_assumptions : Optional[Dict[str, float]], optional
        Assumptions for Owner Earnings valuation. Defaults to:
            - required_return (float): Required return for Owner Earnings valuation (default 0.15).
            - margin_of_safety (float): Margin of safety for Owner Earnings valuation (default 0.25).

    Returns
    -------
    function
        The valuation agent function.
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
        report: Optional[str]  # Consolidated report for all tickers
        overall_results: Optional[Dict[str, Dict[str, Any]]]  # Overall results per ticker (signal, confidence)
        reasoning_reports: Optional[Dict[str, str]]  # Reasoning reports for each ticker
        reasoning_dict: Optional[Dict[str, Dict[str, Dict[str, Any]]]]  # Structured reasoning for each ticker

        # API and External Service Information
        api_key: Optional[str]  # API key for accessing external financial services

        # Parsed User Instructions
        tickers: Optional[List[str]]  # List of tickers to analyze
        period: Optional[str]  # Reporting period (e.g., ttm, quarterly, annual)
        start_date: Optional[str]  # Start date of the analysis period
        end_date: Optional[str]  # End date of the analysis period
        limit: Optional[int]  # Number of periods to fetch

        # Financial Data Retrieved from API
        financial_metrics: Optional[List[Dict[str, Any]]]  # List of financial metric dictionaries
        financial_line_items: Optional[List[Dict[str, Any]]]  # Specific financial line items (e.g., free cash flow, net income)
        market_cap: Optional[float]  # Market capitalization of the company
        financial_metrics_data: Optional[Dict[str, List[Dict[str, Any]]]]  # Financial metrics per ticker
        financial_line_items_data: Optional[Dict[str, List[Dict[str, Any]]]]  # Financial line items per ticker
        market_cap_data: Optional[Dict[str, float]]  # Market cap data per ticker

        # Assumptions for Valuation Methods
        dcf_assumptions: Optional[Dict[str, float]]  # Assumptions for DCF valuation
        owner_earnings_assumptions: Optional[Dict[str, float]]  # Assumptions for Owner Earnings valuation

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

            # Calculate `limit` if `start_date` is provided and `end_date` is resolved
            if start_date:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")

                # Calculate total months between start and end
                total_months = (end.year - start.year) * 12 + (end.month - start.month)

                # Apply the formula for limit based on the period
                if period == "quarterly":
                    limit = total_months // 3  # 3 months per quarter
                elif period == "ttm":
                    limit = max(1, total_months)  # One "ttm" per 12 months
                elif period == "annual":
                    limit = max(1, total_months // 12)  # 12 months per year

            # Validate parsed details
            if not tickers:
                raise ValueError("No valid tickers found in the user instructions.")
            if limit < 1:
                raise ValueError("Limit must be a positive integer.")

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
                

    def fetch_financial_data(state: GraphState):
        """
        Fetch financial data based on the user's instructions.
        """
        print("    * FETCH FINANCIAL DATA")
        #print(f"State received in fetch_financial_data: {state}")  # Log the entire state

        tickers = state.get("tickers")
        period = state.get("period", "ttm")
        limit = state.get("limit", 1)
        end_date = state.get("end_date")
        api_key = state.get("financial_data_api_key")  # Access API key from state

        print(f"Starting data fetch for Ticker: {tickers}, Period: {period}, Limit: {limit}, End Date: {end_date}")

        financial_metrics_data = {}
        financial_line_items_data = {}
        market_cap_data = {}

        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")

                # Fetch financial metrics
                financial_metrics = get_financial_metrics(
                    api_key=api_key,
                    ticker=ticker,
                    report_period=end_date,
                    period=period,
                    limit=limit
                )
                financial_metrics_data[ticker] = financial_metrics

                # Fetch financial line items
                financial_line_items = search_line_items(
                    api_key=api_key,
                    ticker=ticker,
                    report_period=end_date,
                    line_items=["free_cash_flow", "net_income", "depreciation_and_amortization", "capital_expenditure", "working_capital"],
                    period=period,
                    limit=limit
                )
                financial_line_items_data[ticker] = financial_line_items

                # Fetch market cap
                market_cap = get_market_cap(api_key=api_key, ticker=ticker)
                market_cap_data[ticker] = market_cap

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

        # Update state with fetched data
        state["financial_metrics_data"] = financial_metrics_data
        state["financial_line_items_data"] = financial_line_items_data
        state["market_cap_data"] = market_cap_data

        return state
    

    def generate_analysis(
        state: GraphState, 
        dcf_assumptions: Optional[Dict[str, float]] = None,
        owner_earnings_assumptions: Optional[Dict[str, float]] = None,
        sensitivity_ranges: Optional[Dict[str, List[float]]] = None
    ):
        """
        Generate a valuation analysis based on financial metrics.

        This function evaluates valuation gaps using DCF and Owner Earnings methods and provides
        an overall signal with confidence and reasoning. Optionally, performs sensitivity analysis
        if sensitivity ranges are provided.

        Parameters:
            state (GraphState): The current state containing financial metrics and user instructions.
                - ticker (str): The ticker of the company.
                - period (str): The period of data to fetch (e.g., 'quarterly', 'ttm').
                - limit (int): The number of periods to fetch.
            dcf_assumptions (Optional[Dict[str, float]]): Assumptions for DCF valuation. Defaults to:
                - discount_rate (float): Discount rate for DCF (default 0.10).
                - terminal_growth_rate (float): Terminal growth rate for DCF (default 0.03).
                - num_years (int): Number of projection years for DCF (default 5).
            owner_earnings_assumptions (Optional[Dict[str, float]]): Assumptions for Owner Earnings valuation. Defaults to:
                - required_return (float): Required return for Owner Earnings valuation (default 0.15).
                - margin_of_safety (float): Margin of safety for Owner Earnings valuation (default 0.25).
            sensitivity_ranges (Optional[Dict[str, List[float]]]): Ranges for sensitivity analysis. Defaults to:
                - Example:
                    {
                        "growth_rate": [0.02, 0.05, 0.08],
                        "discount_rate": [0.08, 0.10, 0.12],
                        "required_return": [0.135, 0.15, 0.165]
                    }

        Returns:
            dict: Updated state with overall signal, confidence, and reasoning report.
        """

        print("    * GENERATE ANALYSIS")


        # Initialize result containers
        reasoning_reports = {}
        overall_results = {}
        reasoning_dict = {}  # Initialize reasoning_dict for structured analysis


        # Assign default assumptions if None
        dcf_assumptions = dcf_assumptions or {
            "discount_rate": 0.10,
            "terminal_growth_rate": 0.03,
            "num_years": 5,
        }
        owner_earnings_assumptions = owner_earnings_assumptions or {
            "required_return": 0.15,
            "margin_of_safety": 0.25,
        }

        # Default sensitivity ranges if None
        sensitivity_ranges = sensitivity_ranges or {
            "growth_rate": [0.02, 0.05, 0.08],  # Growth rate sensitivity (2%, 5%, 8%)
            "discount_rate": [0.08, 0.10, 0.12],  # Discount rate sensitivity (8%, 10%, 12%)
            "required_return": [0.135, 0.15, 0.165],  # Required return sensitivity (13.5%, 15%, 16.5%)
        }

        #print("Sensitivity Ranges:", sensitivity_ranges)

        # Check if sensitivity analysis is needed
        perform_sensitivity = sensitivity_ranges is not None

        # Ensure sensitivity ranges are complete
        if perform_sensitivity:
            required_keys = {"growth_rate", "discount_rate", "required_return"}
            missing_keys = required_keys - sensitivity_ranges.keys()
            if missing_keys:
                raise ValueError(f"Missing sensitivity ranges for: {', '.join(missing_keys)}")

        # Process each ticker individually
        for ticker in state.get("tickers", []):
            try:
                # Extract ticker-specific data
                financial_metrics = state.get("financial_metrics_data", {}).get(ticker)
                financial_line_items = state.get("financial_line_items_data", {}).get(ticker)
                market_cap = state.get("market_cap_data", {}).get(ticker)

                if not financial_metrics or not financial_line_items or market_cap is None:
                    raise ValueError(f"Missing data for ticker {ticker}")

                # Sort metrics by calendar_date to ensure proper order
                financial_metrics = sorted(financial_metrics, key=lambda x: x["calendar_date"], reverse=True)

                # Use the most recent period for point-in-time metrics
                latest_period = financial_metrics[0]

                # Calculate TTM averages or aggregates for cumulative metrics
                ttm_free_cash_flow = sum(m.get("free_cash_flow_per_share", 0) for m in financial_metrics)
                avg_earnings_growth = sum(m.get("earnings_growth", 0) for m in financial_metrics) / len(financial_metrics)

                # Fetch specific financial line items (current and previous data based on period and limit)
                # Convert report periods to datetime objects and find the latest and second latest periods
                latest_item = None
                previous_item = None
                latest_date = None
                second_latest_date = None

                for item in financial_line_items:
                    current_date = datetime.strptime(item['report_period'], '%Y-%m-%d')
                    if latest_date is None or current_date > latest_date:
                        previous_item = latest_item
                        second_latest_date = latest_date
                        latest_item = item
                        latest_date = current_date
                    elif second_latest_date is None or current_date > second_latest_date:
                        previous_item = item
                        second_latest_date = current_date

                # Print the results
                #print("Current Financial Line Item:")
                #print(latest_item)
                current_financial_line_item = latest_item  # financial_line_items[0]

                #print("\nPrevious Financial Line Item:")
                #print(previous_item)
                previous_financial_line_item = previous_item # financial_line_items[1]
            
                # Calculate working capital change
                working_capital_change = (
                    current_financial_line_item.get("working_capital", 0)
                    - previous_financial_line_item.get("working_capital", 0)
                )

                # Owner Earnings Valuation
                owner_earnings_value = calculate_owner_earnings_value(
                    net_income=current_financial_line_item.get("net_income"),
                    depreciation=current_financial_line_item.get("depreciation_and_amortization"),
                    capex=current_financial_line_item.get("capital_expenditure"),
                    working_capital_change=working_capital_change,
                    growth_rate=avg_earnings_growth,  # Use multi-quarter average growth rate # Base case: Use average growth rate
                    required_return=owner_earnings_assumptions["required_return"],
                    margin_of_safety=owner_earnings_assumptions["margin_of_safety"],
                )

                # DCF Valuation
                dcf_value = calculate_intrinsic_value(
                    free_cash_flow=ttm_free_cash_flow,  # Use TTM free cash flow
                    growth_rate=avg_earnings_growth,  # Use multi-quarter average growth rate # Base case: Use average growth rate
                    discount_rate=dcf_assumptions["discount_rate"],
                    terminal_growth_rate=dcf_assumptions["terminal_growth_rate"],
                    num_years=dcf_assumptions["num_years"],
                )

                # Calculate valuation gaps
                dcf_gap = (dcf_value - market_cap) / market_cap
                owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
                valuation_gap = (dcf_gap + owner_earnings_gap) / 2

                # Perform sensitivity analysis for DCF
                sensitivity_results_dcf = []
                sensitivity_results_oe = []
                if perform_sensitivity:
                    for growth_rate in sensitivity_ranges.get("growth_rate", []):
                        for discount_rate in sensitivity_ranges.get("discount_rate", []):
                            try:
                                intrinsic_value = calculate_intrinsic_value(
                                    free_cash_flow=ttm_free_cash_flow,
                                    growth_rate=growth_rate,
                                    discount_rate=discount_rate,
                                    terminal_growth_rate=dcf_assumptions["terminal_growth_rate"],
                                    num_years=dcf_assumptions["num_years"],
                                )
                                sensitivity_results_dcf.append({
                                    "growth_rate": growth_rate,
                                    "discount_rate": discount_rate,
                                    "intrinsic_value": intrinsic_value,
                                })
                                #print(f"DCF Sensitivity - Growth Rate: {growth_rate:.2%}, Discount Rate: {discount_rate:.2%}, Intrinsic Value: ${intrinsic_value:,.2f}")
                            except Exception as e:
                                print(f"Error in DCF sensitivity calculation: {e}")

                    # Perform sensitivity analysis for Owner Earnings
                    for growth_rate in sensitivity_ranges.get("growth_rate", []):
                        for required_return in sensitivity_ranges.get("required_return", []):
                            try:
                                owner_earnings_value = calculate_owner_earnings_value(
                                    net_income=current_financial_line_item.get("net_income"),
                                    depreciation=current_financial_line_item.get("depreciation_and_amortization"),
                                    capex=current_financial_line_item.get("capital_expenditure"),
                                    working_capital_change=working_capital_change,
                                    growth_rate=growth_rate,
                                    required_return=required_return,
                                    margin_of_safety=owner_earnings_assumptions["margin_of_safety"],
                                )
                                sensitivity_results_oe.append({
                                    "growth_rate": growth_rate,
                                    "required_return": required_return,
                                    "owner_earnings_value": owner_earnings_value,
                                })
                                #print(f"OE Sensitivity - Growth Rate: {growth_rate:.2%}, Required Return: {required_return:.2%}, Intrinsic Value: ${owner_earnings_value:,.2f}")
                            except Exception as e:
                                print(f"Error in Owner Earnings sensitivity calculation: {e}")

                # Determine signal
                if valuation_gap > 0.30:  # More than 30% undervalued
                    signal = "Very Bullish"
                elif 0.15 < valuation_gap <= 0.30:  # Between 15% and 30% undervalued
                    signal = "Bullish"
                elif -0.15 <= valuation_gap <= 0.15:  # Within ±15% of fair value
                    signal = "Neutral"
                elif -0.30 <= valuation_gap < -0.15:  # Between 15% and 30% overvalued
                    signal = "Bearish"
                else:  # More than 30% overvalued
                    signal = "Very Bearish"

                # Create reasoning
                reasoning = {
                    "dcf_analysis": {
                        "signal": "Bullish" if dcf_gap > 0.15 else "Bearish" if dcf_gap < -0.15 else "Neutral",
                        "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}",
                    },
                    "owner_earnings_analysis": {
                        "signal": "Bullish" if owner_earnings_gap > 0.15 else "Bearish" if owner_earnings_gap < -0.15 else "Neutral",
                        "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}",
                    },
                }

                # Confidence calculation
                confidence = round(abs(valuation_gap), 2) * 100

                # Generate reasoning report
                #ticker = state.get("ticker", "Unknown Ticker")  # Fallback to "Unknown Ticker" if not provided
                
                reasoning_report = f"""## **Valuation Analysis Report for {ticker}**\n\n"""
                reasoning_report += f"""### **Overall Signal**: {signal}\n"""
                reasoning_report += f"""### **Confidence**: {confidence}%\n\n"""
                reasoning_report += f"""### **Analysis Details**\n"""
                reasoning_report += f"""- **DCF Analysis**: {reasoning['dcf_analysis']['details']}\n"""
                reasoning_report += f"""- **Owner Earnings Analysis**: {reasoning['owner_earnings_analysis']['details']}\n"""

                # Add sensitivity analysis results if enabled in the state
                if perform_sensitivity:
                    # Sensitivity Analysis for DCF
                    reasoning_report += f"""\n### **Sensitivity Analysis (DCF)**\n"""
                    reasoning_report += f"""| Growth Rate | Discount Rate | Intrinsic Value ($) |\n"""
                    reasoning_report += f"""|-------------|---------------|---------------------|\n"""
                    for result in sensitivity_results_dcf:
                        reasoning_report += (
                            f"| {result['growth_rate']:.2%} | {result['discount_rate']:.2%} | ${result['intrinsic_value']:,.2f} |\n"
                        )

                    # Sensitivity Analysis for Owner Earnings
                    reasoning_report += f"""\n### **Sensitivity Analysis (Owner Earnings)**\n"""
                    reasoning_report += f"""| Growth Rate | Required Return | Intrinsic Value ($) |\n"""
                    reasoning_report += f"""|-------------|-----------------|---------------------|\n"""
                    for result in sensitivity_results_oe:
                        reasoning_report += (
                            f"| {result['growth_rate']:.2%} | {result['required_return']:.2%} | ${result['owner_earnings_value']:,.2f} |\n"
                        )

                # Store the analysis for the ticker
                overall_results[ticker] = {"signal": signal, "confidence": confidence}
                reasoning_reports[ticker] = reasoning_report

                # Add structured data to reasoning_dict
                reasoning_dict[ticker] = {
                    "Overall Signal": signal,
                    "Confidence": confidence,
                    "Analysis Details": {
                        "DCF Analysis": reasoning["dcf_analysis"]["details"],
                        "Owner Earnings Analysis": reasoning["owner_earnings_analysis"]["details"],
                    },
                    "Sensitivity Analysis": {
                        "DCF": [
                            {
                                "Growth Rate": result["growth_rate"],
                                "Discount Rate": result["discount_rate"],
                                "Intrinsic Value ($)": result["intrinsic_value"],
                            }
                            for result in sensitivity_results_dcf
                        ],
                        "Owner Earnings": [
                            {
                                "Growth Rate": result["growth_rate"],
                                "Required Return": result["required_return"],
                                "Intrinsic Value ($)": result["owner_earnings_value"],
                            }
                            for result in sensitivity_results_oe
                        ],
                    } if perform_sensitivity else None,
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
        state["reasoning_dict"] = reasoning_dict

        return state

    # Define the graph
    node_functions = {
        "parse_user_instructions": parse_user_instructions,
        "fix_parse_instructions_code": fix_parse_instructions_code,  # Ensure this node comes after parsing
        "fetch_financial_data": fetch_financial_data,
        "generate_analysis": generate_analysis,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="parse_user_instructions",
        create_code_node_name="fetch_financial_data",
        execute_code_node_name="generate_analysis",
        fix_code_node_name="fix_parse_instructions_code",  # Comes after parsing
    )

    return app
    