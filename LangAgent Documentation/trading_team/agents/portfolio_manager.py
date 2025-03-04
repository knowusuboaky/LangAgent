# LangAgent/trading_team/agents/portfolio_manager.py


# * LIBRARIES

import os
import re
import json
import math
import time
import traceback
from typing import Annotated, Any, Dict, List, Sequence, TypedDict, Callable, Dict, Type, Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import requests
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from langagent.trading_team.agents.risk_manager import make_risk_manager_agent
from langagent.trading_team.agents.sentiments_analyst import make_sentiments_agent
from langagent.trading_team.agents.technicals_analyst import make_technicals_agent
from langagent.trading_team.agents.valuations_analyst import make_valuations_agent
from langagent.trading_team.agents.fundamentals_analyst import make_fundamentals_agent
from langgraph.graph import StateGraph, END

# Function for making trading decisions
def comprehensive_ticker_analysis(data, condition):
    """
    Performs a complete analysis of tickers by combining analyst signals, portfolio risk management,
    and final decision-making into a single function.

    Parameters:
        data (dict): Nested data structure containing analyst signals and portfolio risk management data.
        condition (str): Investment condition for determining weighting and risk preferences.

    Returns:
        dict: Final evaluation for each ticker with detailed actions and decisions.
    """
    # Signal scoring and condition weights
    SIGNAL_SCORES = {
        "Very Bearish": 0,
        "Bearish": 25,
        "Neutral": 50,
        "Bullish": 75,
        "Very Bullish": 100,
    }

    CONDITIONS_WEIGHTS = {
        "Long-term Growth": {"Fundamentals": 30, "Sentiments": 10, "Valuations": 35, "Technicals": 5},
        "Short-term Momentum": {"Fundamentals": 10, "Sentiments": 35, "Valuations": 10, "Technicals": 30},
        "Income-Oriented": {"Fundamentals": 35, "Sentiments": 5, "Valuations": 30, "Technicals": 5},
        "Value-Oriented": {"Fundamentals": 20, "Sentiments": 10, "Valuations": 40, "Technicals": 5},
        "Risk-Averse": {"Fundamentals": 25, "Sentiments": 5, "Valuations": 30, "Technicals": 10},
        "Balanced": {"Fundamentals": 25, "Sentiments": 20, "Valuations": 25, "Technicals": 10},
        "Speculative": {"Fundamentals": 10, "Sentiments": 30, "Valuations": 10, "Technicals": 40},
        "Growth-At-Reasonable-Price": {"Fundamentals": 30, "Sentiments": 15, "Valuations": 30, "Technicals": 5},
        "Event-Driven": {"Fundamentals": 10, "Sentiments": 40, "Valuations": 10, "Technicals": 25},
        "Contrarian": {"Fundamentals": 20, "Sentiments": 10, "Valuations": 40, "Technicals": 10},
        "High-Yield Focus": {"Fundamentals": 40, "Sentiments": 10, "Valuations": 20, "Technicals": 5},
        "Momentum with Risk": {"Fundamentals": 10, "Sentiments": 30, "Valuations": 10, "Technicals": 40},
        "Sector Rotation": {"Fundamentals": 25, "Sentiments": 30, "Valuations": 15, "Technicals": 15},
        "Unspecified": {"Fundamentals": 25, "Sentiments": 25, "Valuations": 25, "Technicals": 25},
    }

    DECISION_WEIGHTS = {
        "Long-term Growth": {"Analyst Weight": 0.6, "Portfolio Weight": 0.4},
        "Short-term Momentum": {"Analyst Weight": 0.5, "Portfolio Weight": 0.5},
        "Income-Oriented": {"Analyst Weight": 0.4, "Portfolio Weight": 0.6},
        "Value-Oriented": {"Analyst Weight": 0.5, "Portfolio Weight": 0.5},
        "Risk-Averse": {"Analyst Weight": 0.3, "Portfolio Weight": 0.7},
        "Balanced": {"Analyst Weight": 0.5, "Portfolio Weight": 0.5},
        "Speculative": {"Analyst Weight": 0.7, "Portfolio Weight": 0.3},
        "Growth-At-Reasonable-Price": {"Analyst Weight": 0.6, "Portfolio Weight": 0.4},
        "Event-Driven": {"Analyst Weight": 0.7, "Portfolio Weight": 0.3},
        "Contrarian": {"Analyst Weight": 0.5, "Portfolio Weight": 0.5},
        "High-Yield Focus": {"Analyst Weight": 0.4, "Portfolio Weight": 0.6},
        "Momentum with Risk": {"Analyst Weight": 0.6, "Portfolio Weight": 0.4},
        "Sector Rotation": {"Analyst Weight": 0.5, "Portfolio Weight": 0.5},
        "Unspecified": {"Analyst Weight": 0.5, "Portfolio Weight": 0.5},
    }

    weights = CONDITIONS_WEIGHTS.get(condition, CONDITIONS_WEIGHTS["Unspecified"])
    decision_weights = DECISION_WEIGHTS.get(condition, DECISION_WEIGHTS["Unspecified"])
    total_weight = sum(weights.values())

    results = {}

    for ticker, ticker_data in data.get("data", {}).items():
        # Analyst signals processing
        weighted_score = 0
        bearish_count = 0
        bullish_count = 0
        neutral_count = 0

        for category, category_data in ticker_data.get("analyst", {}).items():
            if isinstance(category_data, dict) and "Overall Signal" in category_data and "Confidence" in category_data:
                signal = category_data["Overall Signal"]
                confidence = category_data["Confidence"]

                signal_score = SIGNAL_SCORES.get(signal, 50)
                if signal in ["Very Bearish", "Bearish"]:
                    bearish_count += 1
                elif signal in ["Bullish", "Very Bullish"]:
                    bullish_count += 1
                elif signal == "Neutral":
                    neutral_count += 1

                category_weight = weights.get(category, 0)
                contribution = (signal_score * confidence * category_weight) / (100 * total_weight)
                weighted_score += contribution

        weighted_score = max(0, min(100, weighted_score))
        analyst_decision = "Buy" if weighted_score > 60 else "Sell" if weighted_score < 40 else "Hold"

        # Calculate portfolio risk weighted score (always 100)
        portfolio_risk_score = 100

        # Calculate overall weighted score using decision weights
        analyst_weight = decision_weights["Analyst Weight"]
        portfolio_weight = decision_weights["Portfolio Weight"]
        overall_weighted_score = (weighted_score * analyst_weight) + (portfolio_risk_score * portfolio_weight)

        # Portfolio risk management processing
        portfolio_data = ticker_data.get("portfolio", {}).get("Risk Management", {})
        date = portfolio_data.get("Date", "N/A")
        remaining_position_limit = portfolio_data.get("Remaining Position Limit", 0)
        current_price = portfolio_data.get("Current Price", 0)
        max_shares_allowed = portfolio_data.get("Max Shares Allowed", 0)
        portfolio_value = portfolio_data.get("Portfolio Value", 0)
        current_position_value = portfolio_data.get("Current Position Value", 0)
        position_limit = portfolio_data.get("Position Limit", 0)
        remaining_limit = portfolio_data.get("Remaining Limit", 0)
        available_cash = portfolio_data.get("Available Cash", 0)

        portfolio_action = "Hold"
        quantity = 0

        if remaining_position_limit > 0 and available_cash > 0:
            max_buyable_shares = min(remaining_position_limit / current_price, available_cash / current_price)
            quantity = int(max_buyable_shares)
            if quantity > 0:
                portfolio_action = "Buy"
        elif current_position_value > position_limit:
            excess_value = current_position_value - position_limit
            quantity = int(excess_value / current_price)
            if quantity > 0:
                portfolio_action = "Sell"

        # Decision resolution with weights
        analyst_weight = decision_weights["Analyst Weight"]
        portfolio_weight = decision_weights["Portfolio Weight"]

        final_score = (weighted_score * analyst_weight) + ((100 if portfolio_action == "Buy" else 0 if portfolio_action == "Sell" else 50) * portfolio_weight)
        final_decision = "Buy" if final_score > 60 else "Sell" if final_score < 40 else "Hold"

        # Adjust final decision if max_shares_allowed is 0
        if final_decision in ["Buy", "Sell"] and int(abs(max_shares_allowed)) == 0:
            final_decision = "Hold"

        # Calculate quantity if the final decision is Buy
        if final_decision == "Buy" and remaining_position_limit > 0 and available_cash > 0:
            max_buyable_shares = min(remaining_position_limit / current_price, available_cash / current_price)
            quantity = int(max_buyable_shares)
        else:
            quantity = 0

        # Combine results with debugging
        results[ticker] = {
            "Date": date,
            "Weighted Score": round(overall_weighted_score, 2),
            "Final Decision": final_decision,
            "Quantity": quantity,
            "Bearish Signals": bearish_count,
            "Bullish Signals": bullish_count,
            "Neutral Signals": neutral_count,
            "Remaining Position Limit": remaining_position_limit,
            "Current Price": current_price,
            "Max Shares Allowed": max_shares_allowed,
            "Portfolio Value": portfolio_value,
            "Current Position Value": current_position_value,
            "Position Limit": position_limit,
            "Remaining Limit": remaining_limit,
            "Available Cash": available_cash,
        }

        print(f"DEBUG: Ticker: {ticker}, Results: {results[ticker]}")

    return results

# Function for creating a dictionary of analysts and portfolio
def generate_analysts_reports(
    fundamentals_response,
    sentiments_response,
    valuations_response,
    technicals_response,
    risk_management_response,
):
    """
    Combines reasoning_dict data from Fundamentals, Sentiments, Valuations, Technicals, 
    and Risk Management into a unified report structure for each ticker.

    Parameters
    ----------
    fundamentals_response : dict
        The fundamentals analysis response.
    sentiments_response : dict
        The sentiment analysis response.
    valuations_response : dict
        The valuation analysis response.
    technicals_response : dict
        The Hold
    risk_management_response : dict
        The risk management analysis response.

    Returns
    -------
    dict
        Combined structured reports for all tickers.
    """
    # Extract reasoning_dict from each response
    fundamentals_reasoning = fundamentals_response.get("reasoning_dict", {})
    sentiments_reasoning = sentiments_response.get("reasoning_dict", {})
    valuations_reasoning = valuations_response.get("reasoning_dict", {})
    technicals_reasoning = technicals_response.get("reasoning_dict", {})
    risk_management_reasoning = risk_management_response.get("reasoning_dict", {})

    # Combine reasoning_dict data for all tickers
    combined_reasoning = {}

    # Get all unique tickers from all responses
    tickers = set(
        fundamentals_reasoning.keys()
        | sentiments_reasoning.keys()
        | valuations_reasoning.keys()
        | technicals_reasoning.keys()
        | risk_management_reasoning.keys()
    )

    # Merge data for each ticker
    for ticker in tickers:
        combined_reasoning[ticker] = {
            "portfolio": {
                "Risk Management": risk_management_reasoning.get(ticker, {})
            },
            "analyst": {
                "Fundamentals": fundamentals_reasoning.get(ticker, {}),
                "Sentiments": sentiments_reasoning.get(ticker, {}),
                "Valuations": valuations_reasoning.get(ticker, {}),
                "Technicals": technicals_reasoning.get(ticker, {})
            }
        }

    # Transform into final output format
    output = {"data": {}}
    for ticker, data in combined_reasoning.items():
        output["data"][ticker] = data

    return output



# Setup
def make_portfolio_manager_agent(
    model,
    industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None, 
    dcf_assumptions: Optional[Dict[str, float]] = None,
    owner_earnings_assumptions: Optional[Dict[str, float]] = None,
    sensitivity_ranges: Optional[Dict[str, List[float]]] = None,
    market_index_df: Optional[pd.DataFrame] = None,
    related_securities_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    strategy_weights: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
):
    """
    Creates a portfolio management agent that makes trading decisions for multiple tickers.

    Returns
    -------
    function
        The portfolio management agent function.
    """

    # Define the GraphState for the workflow
    class GraphState(TypedDict):
        """
        Defines the state structure for the portfolio management workflow.

        Attributes
        ----------
        user_instructions : Optional[str]
            Instructions provided by the user to guide the workflow.
        condition : Optional[str]
            Condition or context under which the analysis is performed.
        combined_report : Optional[Dict[str, Any]]
            A combined report summarizing the overall portfolio management workflow.
        analyst_inputs : Optional[str]
            Raw inputs from analysts containing signals and data for portfolio decisions.
        data : Optional[Dict[str, Any]]
            Contains structured data required for decision-making.
        portfolio : Optional[Dict[str, Any]]
            Detailed portfolio information such as tickers, positions, and cash.
        analysis : Optional[Dict[str, Any]]
            Results of the detailed analysis used for decision-making.
        errors : Optional[List[str]]
            A list of errors encountered during the workflow execution.
        result : Optional[Dict[str, Any]]
            Final trading decisions and their details for all tickers.
        metadata : Optional[Dict[str, Any]]
            Additional metadata about the workflow execution.
        logs : Optional[List[str]]
            Logs capturing the sequence of actions and decisions.
        api_key : Optional[str]
            API key for external services if required.
        trading_table : Optional[pd.DataFrame]
            A summary table of trading decisions and their outcomes.
        trading_analyst_summary : Optional[str]
            A detailed markdown summary of analyst insights and explanations.
        """
        user_instructions: Optional[str]
        condition: Optional[str]
        combined_report: Optional[Dict[str, Any]]
        data: Optional[Dict[str, Any]]
        portfolio: Optional[Dict[str, Any]]
        analysis: Optional[Dict[str, Any]]
        errors: Optional[List[str]]
        result: Optional[Dict[str, Any]]
        metadata: Optional[Dict[str, Any]]
        logs: Optional[List[str]]
        api_key: Optional[str]
        trading_table: Optional[pd.DataFrame]
        trading_analysts_summary: Optional[str]
        fundamentals_report: Optional[Dict[str, str]]
        sentiments_report: Optional[Dict[str, str]]
        valuations_report: Optional[Dict[str, str]]
        technicals_report: Optional[Dict[str, str]]
        risk_management_report: Optional[Dict[str, str]]

    def parse_user_instructions(state: Dict) -> Dict:
        """
        Parses user instructions to extract actionable details.

        Parameters:
        - state (Dict): A dictionary containing the user instructions and additional context.

        Returns:
        - Dict: Updated state with the parsed "condition" value.
        """
        print("    * PARSE USER INSTRUCTIONS")

        try:
            # Extract user instructions from the state
            user_instructions = state.get("user_instructions")
            if not user_instructions:
                raise ValueError("User instructions are missing in the state.")

            # Define the prompt for parsing user instructions
            parse_prompt = PromptTemplate(
                template="""
                You are an expert financial analysis assistant, skilled in interpreting user instructions and identifying actionable conditions.

                Based on the provided user instructions, extract the following detail:
                - **condition**: A term that matches one of the following predefined conditions:
                    1. Long-term Growth  
                    2. Short-term Momentum  
                    3. Income-Oriented  
                    4. Value-Oriented  
                    5. Risk-Averse  
                    6. Balanced  
                    7. Speculative  
                    8. Growth-At-Reasonable-Price  
                    9. Event-Driven  
                    10. Contrarian  
                    11. High-Yield Focus  
                    12. Momentum with Risk  
                    13. Sector Rotation  
                    14. Unspecified (use this if no clear match is found).

                Your output must be a valid JSON object in the following format:
                {{
                    "condition": <string>
                }}

                ### Example 1:
                Input:
                "Analyze my portfolio of $10,000 in cash, with $3,000 in Apple (AAPL) and $4,000 in Microsoft (MSFT), focusing on long-term investment goals. Using the last two statements, evaluate the portfolio's performance and conduct a TTM analysis of AAPL and MSFT from October 1, 2024, to December 31, 2024, to inform long-term strategies."

                Output:
                {{
                    "condition": "Long-term Growth"
                }}

                ### Example 2:
                Input:
                "I'm looking for short-term gains with a focus on momentum. Evaluate recent market trends and provide actionable insights for speculative opportunities."

                Output:
                {{
                    "condition": "Speculative"
                }}

                ### Example 3:
                Input:
                "Diversify my investments for stable returns while minimizing risk."

                Output:
                {{
                    "condition": "Risk-Averse"
                }}

                ### User Instructions:
                {user_instructions}

                Ensure the JSON is valid and includes a condition, defaulting to "Unspecified" if no explicit match is found.
                """,
                input_variables=["user_instructions"]
            )

            # Parse the user instructions
            try:
                parsed_preprocessor = parse_prompt | model | JsonOutputParser()
                details = parsed_preprocessor.invoke({"user_instructions": user_instructions})
            except Exception as e:
                raise ValueError(f"Failed to parse user instructions: {str(e)}")

            # Extract the condition
            condition = details.get("condition", "Unspecified")

            # Log details for debugging
            print(f"Parsed Condition: {condition}")

            # Update state with the parsed condition
            state.update({
                "condition": condition,
            })
            return state

        except Exception as e:
            print(f"Error in parsing user instructions: {e}")
            raise    

    def run_analysts_workflow(state: GraphState, 
        industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None, 
        dcf_assumptions: Optional[Dict[str, float]] = None,
        owner_earnings_assumptions: Optional[Dict[str, float]] = None,
        sensitivity_ranges: Optional[Dict[str, List[float]]] = None,
        market_index_df: Optional[pd.DataFrame] = None,
        related_securities_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        strategy_weights: Optional[Dict[str, float]] = None
        ):
        """
        Runs all analyses (Fundamentals, Sentiments, Valuations, Technicals, Risk Management)
        and generates a combined report.

        Parameters
        ----------
        state : dict
            The state containing user instructions and other necessary parameters.
        industry_benchmarks : Optional[Dict[str, Dict[str, float]]]
            Optional industry benchmark data for the fundamentals analysis.

        Returns
        -------
        dict
            The combined report for all tickers.
        """
        print("    * RUNNING ANALYSTS WORKFLOW")
        try:
            user_instructions = state.get("user_instructions")

            api_key = os.environ.get("api_key")
            if not api_key:
                raise ValueError("API key for the financial data API is missing. Please set 'api_key' in the environment variables.")
            
            # Initialize placeholders for analysis responses
            fundamentals_response = sentiments_response = valuations_response = technicals_response = risk_management_response = {}

            # Fundamentals Analysis
            try:
                print("    * GENERATE FUNDAMENTALS ANALYSIS")
                time.sleep(1)
                fundamentals_agent = make_fundamentals_agent(model=model, industry_benchmarks=industry_benchmarks)
                fundamentals_response = fundamentals_agent.invoke({"user_instructions": user_instructions, "financial_data_api_key": api_key})
                state["fundamentals_report"] = fundamentals_response["report"]             
            except Exception as e:
                print(f"Error in Fundamentals Analysis: {e}")
            time.sleep(1)
            # Sentiments Analysis
            try:
                print("    * GENERATE SENTIMENTS ANALYSIS")
                time.sleep(1)
                sentiments_agent = make_sentiments_agent(model=model)
                sentiments_response = sentiments_agent.invoke({"user_instructions": user_instructions, "financial_data_api_key": api_key})
                state["sentiments_report"] = sentiments_response["report"]            
            except Exception as e:
                print(f"Error in Sentiments Analysis: {e}")

            # Valuations Analysis
            try:
                print("    * GENERATE VALUATIONS ANALYSIS")
                time.sleep(1)
                valuations_agent = make_valuations_agent(
                    model=model, 
                    dcf_assumptions=dcf_assumptions, 
                    owner_earnings_assumptions=owner_earnings_assumptions, 
                    sensitivity_ranges=sensitivity_ranges
                )
                valuations_response = valuations_agent.invoke({"user_instructions": user_instructions, "financial_data_api_key": api_key})
                state["valuations_report"] = valuations_response["report"]            
            except Exception as e:
                print(f"Error in Valuations Analysis: {e}")

            # Technicals Analysis
            try:
                print("    * GENERATE TECHNICALS ANALYSIS")
                time.sleep(1)
                technicals_agent = make_technicals_agent(
                    model=model, 
                    market_index_df=market_index_df, 
                    related_securities_dfs=related_securities_dfs, 
                    strategy_weights=strategy_weights
                )
                technicals_response = technicals_agent.invoke({"user_instructions": user_instructions, "financial_data_api_key": api_key})
                state["technicals_report"] = technicals_response["report"]
            except Exception as e:
                print(f"Error in Technicals Analysis: {e}")

            # Risk Management Analysis
            try:
                print("    * GENERATE RISK MANAGEMENT ANALYSIS")
                time.sleep(1)
                risk_manager_agent = make_risk_manager_agent(model=model)
                risk_management_response = risk_manager_agent.invoke({"user_instructions": user_instructions, "financial_data_api_key": api_key})
                state["risk_management_report"] = risk_management_response["report"]             
            except Exception as e:
                print(f"Error in Risk Management Analysis: {e}")

            # Combine the reports into a single unified report
            combined_report = generate_analysts_reports(
                fundamentals_response=fundamentals_response,
                sentiments_response=sentiments_response,
                valuations_response=valuations_response,
                technicals_response=technicals_response,
                risk_management_response=risk_management_response,
            )

            # Optionally, attach the combined report to the state
            state["data"] = combined_report
            print(f"Data: {combined_report}")

        except Exception as e:
            print(f"Error during finalization: {e}")
            raise  # Re-raise the exception after logging or handling as needed

        return state

    def finalize(state: GraphState):
        """
        Finalizes the portfolio management workflow.

        This step summarizes the portfolio management decisions, generates a detailed
        report, and performs any necessary post-processing, such as saving results or
        displaying key metrics.

        Parameters
        ----------
        state : GraphState
            The state containing the results of the analysis and any errors encountered.

        Returns
        -------
        GraphState
            The final state, potentially updated with a portfolio management report.
        """
        print("    * FINALIZING PORTFOLIO MANAGEMENT WORKFLOW")
        try:
            # Extract data and condition
            data = state.get("data")
            print(data)

            condition = state.get("condition", "Unspecified")  # Default to "Unspecified"

            print("    * SUMMARY OF PORTFOLIO MANAGEMENT DECISIONS:")


            # Perform the comprehensive analysis
            analysis_results = comprehensive_ticker_analysis(data, condition)

            # Prepare summaries
            result_summary = []

            for ticker, result in analysis_results.items():
                # Extract results
                date = result["Date"]
                weighted_score = result["Weighted Score"]
                final_decision = result["Final Decision"]
                bearish_count = result["Bearish Signals"]
                bullish_count = result["Bullish Signals"]
                neutral_count = result["Neutral Signals"]
                remaining_position_limit = result["Remaining Position Limit"]
                current_price = result["Current Price"]
                max_shares_allowed = result["Max Shares Allowed"]
                portfolio_value = result["Portfolio Value"]
                current_position_value = result["Current Position Value"]
                position_limit = result["Position Limit"]
                remaining_limit = result["Remaining Limit"]
                available_cash = result["Available Cash"]

                # Shares bought/sold and arrows
                shares_bought = 0
                shares_sold = 0
                cash_arrow = position_arrow = quantity_arrow = shares_bought_arrow = shares_sold_arrow = ""

                if final_decision == "Buy":
                    max_buyable_shares = int(min(remaining_position_limit / current_price, available_cash / current_price))
                    shares_bought = max(0, max_buyable_shares)
                    new_quantity = shares_bought
                    new_cash = available_cash - (shares_bought * current_price)
                    new_position_value = current_position_value + (shares_bought * current_price)

                    cash_arrow = "↓" if new_cash < available_cash else ""
                    position_arrow = "↑" if new_position_value > current_position_value else ""
                    shares_bought_arrow = "↑" if shares_bought > 0 else ""

                elif final_decision == "Sell":
                    max_shares_to_sell = int((current_position_value - position_limit) / current_price)
                    shares_sold = max(0, max_shares_to_sell)
                    new_quantity = -shares_sold
                    new_cash = available_cash + (shares_sold * current_price)
                    new_position_value = current_position_value - (shares_sold * current_price)

                    cash_arrow = "↑" if new_cash > available_cash else ""
                    position_arrow = "↓" if new_position_value < current_position_value else ""
                    shares_sold_arrow = "↓" if shares_sold > 0 else ""

                else:  # Hold or no action
                    new_quantity = 0
                    new_cash = available_cash
                    new_position_value = current_position_value

                # Append results
                result_summary.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Action": final_decision,
                    "Quantity": f"{new_quantity} {quantity_arrow}",
                    "Shares Bought": f"{shares_bought} {shares_bought_arrow}" if shares_bought > 0 else "0",
                    "Shares Sold": f"{shares_sold} {shares_sold_arrow}" if shares_sold > 0 else "0",
                    "Price": current_price,
                    "Available Cash": f"{round(new_cash, 2)} {cash_arrow}",
                    "Position Value": f"{round(new_position_value, 2)} {position_arrow}",
                    "Portfolio Value": round(portfolio_value, 2),
                    "Bullish": bullish_count,
                    "Bearish": bearish_count,
                    "Neutral": neutral_count,
                })

            # Create DataFrame for summaries
            df_result_summary = pd.DataFrame(result_summary)

            # Update state with the summary table
            state["trading_table"] = df_result_summary

        except Exception as e:
            error_message = f"Error in finalize: {str(e)}"
            state.setdefault("errors", []).append(error_message)
            print(error_message)

        return state

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("parse_user_instructions", parse_user_instructions)
    workflow.add_node("run_analysts_workflow", run_analysts_workflow)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("parse_user_instructions")
    workflow.add_edge("parse_user_instructions", "run_analysts_workflow")
    workflow.add_edge("run_analysts_workflow", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile()
    return app
