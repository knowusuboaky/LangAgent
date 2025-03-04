# LangAgent/trading_team/agents/risk_manager.py


# * LIBRARIES

import os
import re
import json
import traceback
import math
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

# Fetch price data from API
def get_prices(api_key: str, ticker: str, start_date: str, end_date: str) -> list[Dict[str, Any]]:

    # fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )

    # Parse the response JSON directly
    data = response.json()
    prices = data.get("prices", [])

    if not prices:
        raise ValueError("No price data returned")

    # Cache the results as a list of dictionaries
    return prices

def prices_to_df(prices: list[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    # Directly use the dictionary values
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

# Setup
def make_risk_manager_agent(
    model,
    api_key: Optional[str] = None,
    ):
    """
    Creates a risk management agent to control position sizing based on real-world risk factors for multiple tickers.

    Returns
    -------
    function
        The risk management agent function.
    """

    # Define the GraphState for the workflow
    class GraphState(TypedDict):
        """
        Defines the state structure for the risk management workflow.

        Attributes
        ----------
        data : Dict[str, Any]
            Contains all input and intermediate data required for the workflow. 
            Includes portfolio details, ticker symbols, analysis configuration, 
            and any other contextual data needed.
        errors : Optional[List[str]]
            A list of error messages encountered during the workflow. This helps
            in debugging and identifying issues with specific operations.
        result : Optional[Dict[str, Any]]
            Stores the final risk analysis results for all tickers. The results 
            include key metrics such as remaining position limits, current prices, 
            and reasoning behind the calculations.
        metadata : Optional[Dict[str, Any]]
            Contains additional metadata about the workflow, such as execution 
            timestamps, agent configurations, or flags for enabling/disabling 
            specific features.
        """
        user_instructions: str
        data: Dict[str, Any]
        errors: Optional[List[str]]
        result: Optional[Dict[str, Any]]
        metadata: Optional[Dict[str, Any]]
        logs: Optional[List[str]]
        api_key: Optional[str]
        report: Optional[str]  
        reasoning_dict: Optional[Dict[str, Dict[str, Dict[str, Any]]]]  # Structured reasoning for each ticker

    def parse_user_instructions(state: Dict):
        """
        Parses user instructions and generates structured `data` for further processing.

        Parameters
        ----------
        state : dict
            The input state containing user instructions and initial setup data.

        Returns
        -------
        dict
            The updated state with parsed structured data added to the `state["data"]`.
        """
        print("    * PARSE USER INSTRUCTIONS")

        try:
            # Extract user instructions from the state
            user_instructions = state.get("user_instructions")
            if not user_instructions:
                raise ValueError("User instructions are missing in the state.")

            # Construct the prompt as a string
            prompt = f"""
            You are an expert financial assistant skilled at interpreting user instructions into structured data for portfolio analysis.

            Parse the user's instructions into the following JSON format:
            {{
                "data": {{
                    "portfolio": {{
                        "cash": <float>,  # User's available cash in USD (default: 10000 if not mentioned)
                        "cost_basis": {{  # Cost basis for each stock (assume zero if not mentioned)
                            "TICKER1": <float>,  # Example: {{"AAPL": 3000}}
                            "TICKER2": <float>
                        }}
                    }},
                    "tickers": [<string>],  # List of tickers to analyze (convert company names to tickers if needed)
                    "start_date": <string>,  # Start date for the analysis (YYYY-MM-DD)
                    "end_date": <string>  # End date for the analysis (YYYY-MM-DD)
                }}
            }}

            ### Example Instructions and Outputs:
            Input: "I have $10,000 in cash, with $3,000 already invested in Apple and $4,000 in Microsoft. Analyze my portfolio from December 1, 2024, to December 31, 2024."
            Output:
            {{
                "data": {{
                    "portfolio": {{
                        "cash": 10000,
                        "cost_basis": {{"AAPL": 3000, "MSFT": 4000}}
                    }},
                    "tickers": ["AAPL", "MSFT"],
                    "start_date": "2024-12-01",
                    "end_date": "2024-12-31"
                }}
            }}

            ### User Instructions:
            {user_instructions}

            Ensure the output is in valid JSON format and follows the rules above.
            """

            # Pass the prompt as a HumanMessage to the model
            response = model.invoke([HumanMessage(content=prompt)])

            # Extract JSON block from the response
            json_match = re.search(r"```json(.*?)```", response.content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON block found in the model response.")

            # Extract and parse the JSON content
            json_text = json_match.group(1).strip()

            # Debugging step: print the raw JSON text
            print("Raw JSON content:", json_text)

            # Replace single quotes with double quotes for JSON compliance
            json_text = json_text.replace("'", '"')

            # Parse JSON
            parsed_result = json.loads(json_text)
            state["data"] = parsed_result.get("data", {})

        except ValueError as ve:
            error_message = f"ValueError: {str(ve)}. Could not parse user instructions."
            state["errors"] = [error_message]
            print(error_message)
            raise

        except json.JSONDecodeError as jde:
            error_message = f"JSONDecodeError: {str(jde)}. The response is not valid JSON."
            state["errors"] = [error_message]
            print(error_message)
            raise

        except Exception as e:
            error_message = f"Error: {str(e)}. Parsing failed."
            state["errors"] = [error_message]
            print(error_message)
            raise

        return state


    # Define individual steps in the workflow
    def preprocess_data(state: GraphState):
        """
        Preprocesses data required for risk management analysis.

        This step calculates the total portfolio value and stores it in the state
        for use in subsequent steps. It ensures that common calculations are not
        repeated, improving efficiency.

        Parameters
        ----------
        state : GraphState
            The input state containing portfolio and analysis configuration data.

        Returns
        -------
        GraphState
            Updated state with the total portfolio value stored in `state["data"]["total_portfolio_value"]`.
        """
        print("    * PREPROCESSING DATA")
        portfolio = state["data"]["portfolio"]
        total_portfolio_value = portfolio.get("cash", 0) + sum(
            portfolio.get("cost_basis", {}).get(ticker, 0) for ticker in portfolio.get("cost_basis", {})
        )
        state["data"]["total_portfolio_value"] = total_portfolio_value
        print(f"    * TOTAL PORTFOLIO VALUE CALCULATED: {total_portfolio_value}")
        return state

    def analyze_prices(state: GraphState):
        """
        Analyzes price data for risk management.

        This step calculates position limits, remaining limits, and recommended
        maximum position sizes for each ticker in the portfolio. Results are stored
        in the state for further processing.

        Parameters
        ----------
        state : GraphState
            The input state containing ticker data, portfolio details, and calculated
            total portfolio value.

        Returns
        -------
        GraphState
            Updated state with the risk analysis results stored in `state["result"]`.
        """
        print("    * ANALYZING PRICE DATA")
        portfolio = state["data"]["portfolio"]
        tickers = state["data"]["tickers"]
        total_portfolio_value = state["data"]["total_portfolio_value"]
        api_key = state.get("api_key")

        # Initialize risk analysis for each ticker
        risk_analysis = {}
        current_prices = {}  # This is initialized but not currently utilized


        for ticker in tickers:
            try:
                prices = get_prices(
                    api_key=api_key,
                    ticker=ticker,
                    start_date=state["data"]["start_date"],
                    end_date=state["data"]["end_date"],
                )
                prices_df = prices_to_df(prices)

                current_price = prices_df["close"].iloc[-1]
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

                position_limit = total_portfolio_value * 0.20
                remaining_position_limit = position_limit - current_position_value
                max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))
                max_shares_allowed = remaining_position_limit / current_price

                risk_analysis[ticker] = {
                    "remaining_position_limit": float(max_position_size),
                    "current_price": float(current_price),
                    "max_shares_allowed": float(max_shares_allowed),
                    "reasoning": {
                        "portfolio_value": float(total_portfolio_value),
                        "current_position": float(current_position_value),
                        "position_limit": float(position_limit),
                        "remaining_limit": float(remaining_position_limit),
                        "available_cash": float(portfolio.get("cash", 0)),
                    },
                }
            except Exception as e:
                error_message = f"Error processing ticker {ticker}: {str(e)}"
                state.setdefault("errors", []).append(error_message)
                print(f"    * ERROR PROCESSING {ticker}")
        state["result"] = risk_analysis
        return state


    def finalize(state: GraphState):
        """
        Finalizes the risk management workflow.

        This step summarizes the risk analysis results and performs any necessary
        post-processing, such as saving results or displaying key metrics.

        Parameters
        ----------
        state : GraphState
            The state containing the results of the analysis and any errors encountered.

        Returns
        -------
        GraphState
            The final state, potentially updated with post-processing results.
        """
        print("    * FINALIZING WORKFLOW")
        try:
            if "result" in state and state["result"]:
                print("    * SUMMARY OF RISK ANALYSIS:")


                # Calculate next trade date
                end_date = state["data"]["end_date"]
                next_trade_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

                # Generate the formatted risk management report
                report = "## **Risk Management Report**\n\n"
                results = state["result"]
                reasoning_dict = {}  # Initialize reasoning_dict for structured data

                for ticker, analysis in results.items():
                    remaining_limit = analysis["remaining_position_limit"]
                    current_price = analysis["current_price"]
                    reasoning = analysis["reasoning"]
                    max_shares_allowed = analysis["max_shares_allowed"]

                    report += f"### **Ticker: {ticker}**\n"
                    report += f"- **Date**: {next_trade_date}\n"
                    report += f"- **Remaining Position Limit**: {remaining_limit}\n"
                    report += f"- **Current Price**: {current_price}\n"
                    report += f"- **Max Shares Allowed**: {max_shares_allowed:,.2f}\n\n"

                    report += "#### **Reasoning**:\n"
                    report += f"- **Portfolio Value**: {reasoning['portfolio_value']}\n"
                    report += f"- **Current Position Value**: {reasoning['current_position']}\n"
                    report += f"- **Position Limit**: {reasoning['position_limit']}\n"
                    report += f"- **Remaining Limit**: {reasoning['remaining_limit']}\n"
                    report += f"- **Available Cash**: {reasoning['available_cash']}\n\n"

                    # Add structured data for this ticker to reasoning_dict
                    reasoning_dict[ticker] = {
                        "Date": next_trade_date,
                        "Remaining Position Limit": remaining_limit,
                        "Current Price": current_price,
                        "Max Shares Allowed": round(max_shares_allowed, 2),
                        "Portfolio Value": reasoning["portfolio_value"],
                        "Current Position Value": reasoning["current_position"],
                        "Position Limit": reasoning["position_limit"],
                        "Remaining Limit": reasoning["remaining_limit"],
                        "Available Cash": reasoning["available_cash"],
                    }

                # Print the report for debugging
                # print(report)

                # Add the report to the state
                state["report"] = report
                state["reasoning_dict"] = reasoning_dict  # Save the structured reasoning_dict in the state

            else:
                print("    * NO RESULTS TO DISPLAY")

        except Exception as e:
            print(f"Error in finalize: {e}")
            state.setdefault("errors", []).append(f"Error in finalize: {e}")

        return state

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    # Add nodes to the workflow
    workflow.add_node("parse_user_instructions", parse_user_instructions)
    workflow.add_node("preprocess_data", preprocess_data)
    workflow.add_node("analyze_prices", analyze_prices)
    workflow.add_node("finalize", finalize)

    # Set entry point and workflow edges
    workflow.set_entry_point("parse_user_instructions")
    workflow.add_edge("parse_user_instructions", "preprocess_data")    
    workflow.add_edge("preprocess_data", "analyze_prices")
    workflow.add_edge("analyze_prices", "finalize")
    workflow.add_edge("finalize", END)

    # Compile the app
    app = workflow.compile()

    return app
