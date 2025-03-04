# LangAgent/trading_team/agents/fundamentals_analyst.py


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
    fix_code_node_name: Optional[str] = None,
    explain_code_node_name: Optional[str] = None,
    error_key: str = "error",
    max_retries_key: str = "max_retries",
    retry_count_key: str = "retry_count",
    human_in_the_loop: bool = False,
    human_review_node_name: str = "human_review",
    checkpointer: Optional[Callable] = None,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = False,
):
    workflow = StateGraph(GraphState)

    # Add nodes
    if not bypass_recommended_steps:
        workflow.add_node(recommended_steps_node_name, node_functions[recommended_steps_node_name])
    workflow.add_node(create_code_node_name, node_functions[create_code_node_name])
    workflow.add_node(execute_code_node_name, node_functions[execute_code_node_name])
    workflow.add_node("END", lambda state: state)  # Define the END node

    if fix_code_node_name:
        workflow.add_node(fix_code_node_name, node_functions[fix_code_node_name])
    if explain_code_node_name and not bypass_explain_code:
        workflow.add_node(explain_code_node_name, node_functions[explain_code_node_name])

    # Set entry point
    entry_point = create_code_node_name if bypass_recommended_steps else recommended_steps_node_name
    workflow.set_entry_point(entry_point)

    # Add edges
    if not bypass_recommended_steps:
        next_node = human_review_node_name if human_in_the_loop else create_code_node_name
        workflow.add_edge(recommended_steps_node_name, next_node)
    workflow.add_edge(create_code_node_name, execute_code_node_name)

    if fix_code_node_name:
        workflow.add_conditional_edges(
            execute_code_node_name,
            lambda state: fix_code_node_name if state.get(error_key) else (explain_code_node_name if explain_code_node_name else "END"),
            {
                fix_code_node_name: fix_code_node_name,
                explain_code_node_name: explain_code_node_name,
                "END": "END",
            }
        )
        workflow.add_edge(fix_code_node_name, execute_code_node_name)

    # If no fix or explain nodes, execute_code_node_name -> END
    if not fix_code_node_name and (not explain_code_node_name or bypass_explain_code):
        workflow.add_edge(execute_code_node_name, "END")

    if explain_code_node_name and not bypass_explain_code:
        workflow.add_edge(explain_code_node_name, "END")

    if human_in_the_loop:
        workflow.add_node(human_review_node_name, node_functions[human_review_node_name])
        workflow.add_edge(human_review_node_name, create_code_node_name)

    # Compile the workflow
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()

# Financial API Functions
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



# Setup
def make_fundamentals_agent(
    model,
    industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
    api_key: Optional[str] = None,
):
    """
    Creates a fundamentals analysis agent to evaluate a company's financial metrics and generate trading signals.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use for reasoning and explanations.
    industry_benchmarks : dict, optional
        A dictionary of industry-specific benchmarks for financial metrics. These benchmarks provide reference values
        against which a company's financial metrics can be compared. The dictionary is structured into categories 
        of metrics, each containing specific benchmarks.

        Structure:
        {
            "profitability": {
                "return_on_equity": float,  # Target return on equity, e.g., 0.15 for 15%
                "net_margin": float,        # Target net margin, e.g., 0.20 for 20%
                "operating_margin": float,  # Target operating margin, e.g., 0.15 for 15%
            },
            "growth": {
                "revenue_growth": float,    # Target revenue growth rate, e.g., 0.10 for 10%
                "earnings_growth": float,   # Target earnings growth rate, e.g., 0.10 for 10%
                "book_value_growth": float, # Target book value growth rate, e.g., 0.10 for 10%
            },
            "financial_health": {
                "current_ratio": float,               # Minimum current ratio, e.g., 1.5
                "debt_to_equity": float,              # Maximum debt-to-equity ratio, e.g., 0.5
                "free_cash_flow_to_eps_ratio": float, # Minimum free cash flow to EPS ratio, e.g., 0.8
            },
            "market_value": {
                "peg_ratio": float,                      # Maximum PEG ratio, e.g., 1.0
                "enterprise_value_to_ebitda_ratio": float, # Maximum EV/EBITDA ratio, e.g., 10
            },
            "price_ratios": {
                "price_to_earnings_ratio": float, # Maximum P/E ratio, e.g., 25
                "price_to_book_ratio": float,     # Maximum P/B ratio, e.g., 3
                "price_to_sales_ratio": float,    # Maximum P/S ratio, e.g., 5
            },
        }

        Example:
        industry_benchmarks = {
            "profitability": {
                "return_on_equity": 0.15,
                "net_margin": 0.20,
                "operating_margin": 0.15,
            },
            "growth": {
                "revenue_growth": 0.10,
                "earnings_growth": 0.10,
                "book_value_growth": 0.10,
            },
            "financial_health": {
                "current_ratio": 1.5,
                "debt_to_equity": 0.5,
                "free_cash_flow_to_eps_ratio": 0.8,
            },
            "market_value": {
                "peg_ratio": 1.0,
                "enterprise_value_to_ebitda_ratio": 10,
            },
            "price_ratios": {
                "price_to_earnings_ratio": 25,
                "price_to_book_ratio": 3,
                "price_to_sales_ratio": 5,
            },
        }

        Defaults to None.
    api_key : str, optional
        API key for accessing financial datasets. Defaults to None.

    Returns
    -------
    function
        The fundamentals agent function.
    """
    # Define GraphState for the agent
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]  # Messages for LLMs
        user_instructions: str  # User's input instructions
        reasoning: Optional[str]  # Reasoning or context built by the agent
        signal: Optional[str]  # Signal output from the analysis (e.g., bullish, bearish)
        confidence: Optional[float]  # Confidence score for the signal
        error: Optional[str]  # Error messages, if any
        api_key: Optional[str]  # API key for accessing external services

        # Parsed instructions
        tickers: Optional[List[str]]  # List of tickers to analyze
        period: Optional[str]  # Reporting period (e.g., ttm, annual)
        start_date: Optional[str]  # Start date of the analysis period
        end_date: Optional[str]  # End date of the analysis period
        limit: Optional[int]

        # Financial metrics retrieved from the API
        financial_metrics: Optional[List[Dict[str, any]]]  # List of financial metric dictionaries
        financial_metrics_data: Optional[Dict[str, List[Dict[str, any]]]]  # Dictionary of financial metric for each ticker
        fundamentals_report: Optional[str]  # Markdown format of the analysis report

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


    def fetch_financial_data(state: GraphState):
        """
        Fetch financial data based on the user's instructions.
        """
        print("    * FETCH FINANCIAL DATA")
        #print(f"State received in fetch_financial_data: {state}")  # Log the entire state

        tickers = state.get("tickers")
        period = state.get("period", "ttm")
        limit = state.get("limit")
        end_date = state.get("end_date")
        api_key = state.get("financial_data_api_key")  # Access API key from state


        #print(f"Ticker: {ticker}, Period: {period}, Limit: {limit}, End Date: {end_date}")
        financial_metrics_data = {}
        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")

                # Fetch financial data
                financial_metrics = get_financial_metrics(
                    api_key=api_key,
                    ticker=ticker,
                    period=period,
                    limit=limit,
                    report_period=end_date
                )
                financial_metrics_data[ticker] = financial_metrics
                print(f"Data for {ticker}: {financial_metrics}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                financial_metrics_data[ticker] = None

        state["financial_metrics_data"] = financial_metrics_data
        
        print("\nAll fetched data:")
        for ticker, data in financial_metrics_data.items():
            print(f"{ticker}: {data}")
        
        return state


                
    def generate_analysis(state: GraphState, industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Generate an analysis based on financial metrics, using industry benchmarks if provided or Z-scores otherwise.

        This function evaluates multiple categories of financial health, including profitability, growth, financial health,
        market value, and price ratios. Each category is analyzed based on provided benchmarks or Z-scores, with results
        summarized into an overall signal and detailed reasoning.

        Parameters:
            state (GraphState): The current state containing financial metrics and user instructions.
            industry_benchmarks (Optional[Dict[str, Dict[str, float]]]): Benchmarks for specific metrics by industry.

        Returns:
            dict: Updated state with overall signal, confidence, and reasoning report.
        """
        print("    * GENERATE ANALYSIS")

        financial_metrics_data = state.get("financial_metrics_data")
        if not financial_metrics_data:
            state["overall_signal"] = "neutral"
            state["confidence"] = 0
            state["reasoning_reports"] = {}
            print("No financial metrics provided.")
            return state

        reasoning_reports = {}
        overall_results = {}
        reasoning_dict = {}  # New dictionary for structured analysis results


        for ticker, metrics in financial_metrics_data.items():
            if metrics is None:
                #reasoning_reports[ticker] = f"Error: No data available for {ticker}."
                #overall_results[ticker] = {"signal": None, "confidence": 0}
                continue
        
            # Use default benchmarks if none are provided
            benchmarks = industry_benchmarks or {
                "profitability": {
                    "return_on_equity": 0.15,
                    "net_margin": 0.20,
                    "operating_margin": 0.15,
                },
                "growth": {
                    "revenue_growth": 0.10,
                    "earnings_growth": 0.10,
                    "book_value_growth": 0.10,
                },
                "financial_health": {
                    "current_ratio": 1.5,
                    "debt_to_equity": 0.5,
                    "free_cash_flow_to_eps_ratio": 0.8,
                },
                "market_value": {
                    "peg_ratio": 1.0,
                    "enterprise_value_to_ebitda_ratio": 10,
                },
                "price_ratios": {
                    "price_to_earnings_ratio": 25,
                    "price_to_book_ratio": 3,
                    "price_to_sales_ratio": 5,
                },
            }


            # Compute Z-scores
            z_scores = []
            try:
                if isinstance(metrics, list):  # Metrics over time
                    metrics_by_key = {key: [entry[key] for entry in metrics if key in entry and isinstance(entry[key], (int, float))] for key in metrics[0].keys()}
                    mean = {key: np.mean(values) for key, values in metrics_by_key.items() if values}
                    std_dev = {key: np.std(values) for key, values in metrics_by_key.items() if values}
                    z_scores = [
                        {key: round((entry[key] - mean[key]) / std_dev[key], 2) if key in mean and std_dev[key] != 0 else None for key in entry.keys()}
                        for entry in metrics
                    ]
                elif isinstance(metrics, dict):  # Single point in time
                    mean = {key: metrics[key] for key in metrics if isinstance(metrics[key], (int, float))}
                    std_dev = {key: 1 for key in mean}  # Standard deviation is 1 to avoid division
                    z_scores = [{key: 0 for key in metrics}]  # Z-scores are 0 since there's no variation
                else:
                    raise ValueError("Invalid metrics format")
            except Exception as e:
                print(f"Error calculating Z-scores: {e}")
                z_scores = [{} for _ in metrics] if isinstance(metrics, list) else [{}]


            # Prepare summary tables and calculate signals for each category
            signals_profitability = "Neutral"
            signals_growth = "Neutral"
            signals_financial_health = "Neutral"
            signals_price_ratios = "Neutral"
            signals_market_value = "Neutral"

            profitability_table = []
            growth_table = []
            financial_health_table = []
            market_value_table = []
            price_ratios_table = []

            # Initialize counts for benchmark categories
            above_excellent_count = 0
            excellent_count = 0
            satisfactory_count = 0
            below_average_count = 0
            critical_count = 0

            def categorize_benchmark(value, benchmark):
                if isinstance(value, (int, float)) and isinstance(benchmark, (int, float)):
                    difference = value - benchmark
                    if difference > 0.2 * benchmark:  # 20% or more above benchmark
                        return "Above Excellent"
                    elif 0 < difference <= 0.2 * benchmark:  # Slightly above benchmark
                        return "Excellent"
                    elif abs(difference) <= 0.05 * benchmark:  # Within ±5% of benchmark
                        return "Satisfactory"
                    elif -0.2 * benchmark <= difference < -0.05 * benchmark:  # 5%-20% below benchmark
                        return "Below Average"
                    elif difference < -0.2 * benchmark:  # 20% or more below benchmark
                        return "Critical"
                return "N/A"  # For invalid or non-numeric values

            for period_data, z_score_data in zip(metrics if isinstance(metrics, list) else [metrics], z_scores):
                for category, category_metrics in benchmarks.items():
                    very_bullish_count = 0
                    bullish_count = 0
                    neutral_count = 0
                    bearish_count = 0
                    very_bearish_count = 0

                    above_excellent_count = 0
                    excellent_count = 0
                    satisfactory_count = 0
                    below_average_count = 0
                    critical_count = 0

                    for metric, benchmark in category_metrics.items():
                        value = period_data.get(metric, "N/A")
                        z_score = z_score_data.get(metric, "N/A")
                        classification = "N/A"

                        # Format the period as YYYY-MM
                        raw_date = period_data.get("calendar_date", "N/A")
                        if raw_date != "N/A":
                            try:
                                year, month, _ = map(int, raw_date.split("-"))
                                formatted_period = f"{year}-{month:02d}"  # Ensure month is zero-padded
                                period_data["calendar_date"] = formatted_period  # Update the period in the data
                            except ValueError:
                                formatted_period = "N/A"  # Handle cases where date parsing fails
                        else:
                            formatted_period = "N/A"


                        # Determine classification based on Z-Score
                        if isinstance(z_score, (int, float)):
                            if z_score > 2:
                                classification = "Very High"
                                very_bullish_count += 1
                            elif 1 < z_score <= 2:
                                classification = "High"
                                bullish_count += 1
                            elif -1 <= z_score <= 1:
                                classification = "Average"
                                neutral_count += 1
                            elif -2 <= z_score < -1:
                                classification = "Low"
                                bearish_count += 1
                            elif z_score < -2:
                                classification = "Very Low"
                                very_bearish_count += 1

                        # Determine benchmark category
                        benchmark_status = categorize_benchmark(value, benchmark)
                        if benchmark_status == "Above Excellent":
                            above_excellent_count += 1
                        elif benchmark_status == "Excellent":
                            excellent_count += 1
                        elif benchmark_status == "Satisfactory":
                            satisfactory_count += 1
                        elif benchmark_status == "Below Average":
                            below_average_count += 1
                        elif benchmark_status == "Critical":
                            critical_count += 1

                        row = {
                            "Period": period_data.get("calendar_date", "N/A"),  # Use formatted period (e.g., "2024-Q3")
                            "Metric Name": metric,
                            "Value": value,
                            "Benchmark": benchmark_status,
                            "Z-Score": z_score,
                            "Classification": classification,
                        }

                        if category == "profitability":
                            profitability_table.append(row)
                            profitability_table = [r for r in profitability_table if r.get("Value") not in ["N/A", None]]
                        elif category == "growth":
                            growth_table.append(row)
                            growth_table = [r for r in growth_table if r.get("Value") not in ["N/A", None]]
                        elif category == "financial_health":
                            financial_health_table.append(row)
                            financial_health_table = [r for r in financial_health_table if r.get("Value") not in ["N/A", None]]
                        elif category == "market_value":
                            market_value_table.append(row)
                            market_value_table = [r for r in market_value_table if r.get("Value") not in ["N/A", None]]
                        elif category == "price_ratios":
                            price_ratios_table.append(row)
                            price_ratios_table = [r for r in price_ratios_table if r.get("Value") not in ["N/A", None]]

                    # Define tables for categories
                    tables = {
                        "profitability": profitability_table,
                        "growth": growth_table,
                        "financial_health": financial_health_table,
                        "market_value": market_value_table,
                        "price_ratios": price_ratios_table,
                    }

                    # Determine signal for the category using both Z-Scores and benchmark comparison
                    category_signal = (
                        "Very Bullish" if very_bullish_count + above_excellent_count > bullish_count + excellent_count + neutral_count + satisfactory_count + bearish_count + below_average_count + very_bearish_count + critical_count
                        else "Bullish" if bullish_count + excellent_count > very_bullish_count + above_excellent_count + neutral_count + satisfactory_count + bearish_count + below_average_count + very_bearish_count + critical_count
                        else "Very Bearish" if very_bearish_count + critical_count > bearish_count + below_average_count + neutral_count + satisfactory_count + bullish_count + excellent_count + very_bullish_count + above_excellent_count
                        else "Bearish" if bearish_count + below_average_count > very_bearish_count + critical_count + neutral_count + satisfactory_count + bullish_count + excellent_count + very_bullish_count + above_excellent_count
                        else "Neutral"
                    )


                    # Determine signal for the category using both Z-Scores and benchmark comparison
                    total_count = (
                        very_bullish_count + bullish_count + neutral_count +
                        bearish_count + very_bearish_count + above_excellent_count +
                        excellent_count + satisfactory_count + below_average_count +
                        critical_count
                    )

                    if total_count == 0:
                        category_signal = "Neutral"
                    else:
                        signal_score = (
                            2 * (very_bullish_count + above_excellent_count) +  # Strong positive weight
                            1 * (bullish_count + excellent_count) -            # Moderate positive weight
                            1 * (bearish_count + below_average_count) -        # Moderate negative weight
                            2 * (very_bearish_count + critical_count)          # Strong negative weight
                        )

                        signal_score /= total_count  # Normalize the score

                        # Define thresholds for signal determination
                        if signal_score > 0.5:
                            category_signal = "Very Bullish"
                        elif 0.2 < signal_score <= 0.5:
                            category_signal = "Bullish"
                        elif -0.2 <= signal_score <= 0.2:
                            category_signal = "Neutral"
                        elif -0.5 <= signal_score < -0.2:
                            category_signal = "Bearish"
                        else:
                            category_signal = "Very Bearish"


                    if category == "profitability":
                        signals_profitability = category_signal
                    elif category == "growth":
                        signals_growth = category_signal
                    elif category == "financial_health":
                        signals_financial_health = category_signal
                    elif category == "market_value":
                        signals_market_value = category_signal
                    elif category == "price_ratios":
                        signals_price_ratios = category_signal
            

            # Initialize category signals
            category_signals = []

            # Add each signal to the list dynamically
            if signals_profitability:
                category_signals.append(signals_profitability)
            if signals_growth:
                category_signals.append(signals_growth)
            if signals_financial_health:
                category_signals.append(signals_financial_health)
            if signals_price_ratios:
                category_signals.append(signals_price_ratios)
            if signals_market_value:
                category_signals.append(signals_market_value)


            # Determine overall signal
            very_bullish_signals = category_signals.count("Very Bullish")
            bullish_signals = category_signals.count("Bullish")
            neutral_signals = category_signals.count("Neutral")
            bearish_signals = category_signals.count("Bearish")
            very_bearish_signals = category_signals.count("Very Bearish")

            # Combine bullish and bearish signals with weights
            bullish_score = 2 * very_bullish_signals + bullish_signals
            bearish_score = 2 * very_bearish_signals + bearish_signals

            if bullish_score > bearish_score:
                overall_signal = "Very Bullish" if bullish_score - bearish_score >= 2 else "Bullish"
            elif bearish_score > bullish_score:
                overall_signal = "Very Bearish" if bearish_score - bullish_score >= 2 else "Bearish"
            else:
                overall_signal = "Neutral"

            
            # Calculate confidence level
            bullish_family_score = very_bullish_signals + bullish_signals
            bearish_family_score = very_bearish_signals + bearish_signals

            total_signals = len(category_signals)
            dominant_signals = min(max(bullish_family_score, bearish_family_score), total_signals)  # Ensure dominant_signals is not higher than total_signals
            confidence = round(dominant_signals / total_signals, 2) * 100

            # Initialize confidence_score with the initial confidence value
            confidence_score = confidence

            # Adjust confidence for neutral signals
            if neutral_signals > total_signals / 2:
                confidence_score = round(confidence * 0.8, 2)  # Reduce confidence if "neutral" dominates


            # Generate reasoning report
            #ticker = state.get("ticker", "Unknown Ticker")  # Fallback to "Unknown Ticker" if not provided

            # Start the reasoning report
            reasoning_report = f"""## **Fundamentals Analysis Report for {ticker}**\n\n"""

            # Add overall signal and confidence
            reasoning_report += f"""### **Overall Signal**: {overall_signal}\n"""
            reasoning_report += f"""### **Confidence**: {confidence_score}%\n\n"""

            # Add category signals
            reasoning_report += f"""### **Category Signals**\n"""
            reasoning_report += f"""- **Profitability**: {signals_profitability}\n"""
            reasoning_report += f"""- **Growth**: {signals_growth}\n"""
            reasoning_report += f"""- **Financial Health**: {signals_financial_health}\n"""
            reasoning_report += f"""- **Price Ratios**: {signals_price_ratios}\n"""
            reasoning_report += f"""- **Market Value**: {signals_market_value}\n\n"""

            # Add detailed tables for each category
            for category, table in tables.items():
                reasoning_report += f"#### **{category.capitalize()}**\n"
                if table:
                    periods = sorted(set(row["Period"] for row in table))
                    for period in periods:
                        reasoning_report += f"##### Period: {period}\n"
                        reasoning_report += "\n".join(
                            f"- **{row['Metric Name']}**: Value = {row['Value']}, "
                            f"Benchmark = {row['Benchmark']}, "
                            f"Z-Score = {row['Z-Score']:.2f}, "
                            f"Classification = {row['Classification']}"
                            for row in table if row["Period"] == period
                        ) + "\n\n"
                else:
                    reasoning_report += "No data available.\n\n"

            # Store the analysis for the ticker
            overall_results[ticker] = {"signal": overall_signal, "confidence": confidence}
            reasoning_reports[ticker] = reasoning_report


            # Initialize category signals dict for reasoning dict only
            category_signals_dict = {
                "Profitability": signals_profitability,
                "Growth": signals_growth,
                "Financial Health": signals_financial_health,
                "Price Ratios": signals_price_ratios,
                "Market Value": signals_market_value,
            }

            # Add detailed tables for each category to reasoning_dict
            detailed_tables_dict = {}  # To store detailed tables for reasoning_dict

            for category, table in tables.items():
                category_details = []  # List to store period-specific details
                if table:
                    periods = sorted(set(row["Period"] for row in table))
                    for period in periods:
                        period_details = {
                            "Period": period,
                            "Metrics": [
                                {
                                    "Metric Name": row["Metric Name"],
                                    "Value": row["Value"],
                                    "Benchmark": row["Benchmark"],
                                    "Z-Score": round(row["Z-Score"], 2) if isinstance(row["Z-Score"], (int, float)) else "N/A",
                                    "Classification": row["Classification"],
                                }
                                for row in table if row["Period"] == period
                            ],
                        }
                        category_details.append(period_details)
                else:
                    category_details.append({"Period": "N/A", "Metrics": "No data available"})

                detailed_tables_dict[category.capitalize()] = category_details


            # Add structured data to reasoning_dict
            reasoning_dict[ticker] = {
                "Overall Signal": overall_signal,
                "Confidence": confidence_score,
                "Category Signals": category_signals_dict,
                "Category Details": detailed_tables_dict,  # Include detailed tables here
            }

        # Update state with results for all tickers
        full_report = ""
        for ticker, analysis_report in reasoning_reports.items():
            full_report += analysis_report + "\n" + "-" * 80 + "\n"


        state["report"] = full_report  # Save the consolidated report in the state
        state["overall_results"] = overall_results
        state["reasoning_reports"] = reasoning_reports
        state["reasoning_dict"] = reasoning_dict  # Add the reasoning_dict to the state

        return state


    # Define the graph
    node_functions = {
        "parse_user_instructions": parse_user_instructions,
        "fetch_financial_data": fetch_financial_data,
        "generate_analysis": generate_analysis
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="parse_user_instructions",
        create_code_node_name="fetch_financial_data",
        execute_code_node_name="generate_analysis",

    )

    return app


