# LangAgent/trading_team/agents/technicals_analyst.py


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


#### Helper functions
def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "Very Bullish" if trend_strength > 0.75 else "Bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "Very Bearish" if trend_strength > 0.75 else "Bearish"
        confidence = trend_strength
    else:
        signal = "Neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "average_directional_index": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }

def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (
        bb_upper.iloc[-1] - bb_lower.iloc[-1]
    )

    # Combine 
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "Very Bullish" if abs(z_score.iloc[-1]) > 3 else "Bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "Very Bearish" if abs(z_score.iloc[-1]) > 3 else "Bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "Neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bollinger_bands": float(price_vs_bb),
            "relative_strength_index_14": float(rsi_14.iloc[-1]),
            "relative_strength_index_28": float(rsi_28.iloc[-1]),
        },
    }

def calculate_momentum_signals(prices_df, market_index_df = None):
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)
        # Relative strength
    if market_index_df is not None:
        # Calculate market returns
        market_returns = market_index_df["close"].pct_change()

        # Calculate relative strength
        relative_strength = (
            returns.rolling(21).mean() / market_returns.rolling(21).mean()
        ).iloc[-1]
    else:
        relative_strength = None

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "Very Bullish" if momentum_score > 0.1 else "Bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "Very Bearish" if momentum_score < -0.1 else "Bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "Neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1_month": float(mom_1m.iloc[-1]),
            "momentum_3_months": float(mom_3m.iloc[-1]),
            "momentum_6_months": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
            "relative_strength": float(relative_strength) if relative_strength else None,
        },
    }

def calculate_volatility_signals(prices_df):
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "Very Bullish" if vol_z < -2 else "Bullish"
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "Very Bearish" if vol_z > 2 else "Bearish"
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "Neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "average_true_range_ratio": float(atr_ratio.iloc[-1]),
        },
    }
    
def calculate_stat_arb_signals(prices_df, related_securities_dfs=None):
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent(prices_df["close"])

    # Correlation analysis
    # (would include correlation with related securities in real implementation)
    correlations = None
    if related_securities_dfs:
        correlations = {}
        for name, related_df in related_securities_dfs.items():
            related_returns = related_df["close"].pct_change()
            corr = returns.rolling(63).corr(related_returns).iloc[-1]
            correlations[name] = float(corr)

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        if skew.iloc[-1] > 2:  # Strong bullish skew
            signal = "Very Bullish"
        else:
            signal = "Bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        if skew.iloc[-1] < -2:  # Strong bearish skew
            signal = "Very Bearish"
        else:
            signal = "Bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "Neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
            "correlations": correlations,
        },
    }

def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach.
    """
    # Convert signals to numeric values
    signal_values = {
        #"Very Bullish": 2,
        "Bullish": 1,
        "Neutral": 0,
        "Bearish": -1,
        #"Very Bearish": -2,
    }


    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        #print(numeric_signal)
        weight = weights[strategy]
        #print(weight)
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence
        #print(total_confidence)

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal with expanded categories
    if final_score > 0.6:
        signal = "Very Bullish"
    elif final_score > 0.2:
        signal = "Bullish"
    elif final_score < -0.6:
        signal = "Very Bearish"
    elif final_score < -0.2:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return {"signal": signal, "confidence": abs(final_score)}

def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(
    prices_df: pd.DataFrame, window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0
    )
    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0
    )

    # Calculate ADX
    df["+di"] = 100 * (
        df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["-di"] = 100 * (
        df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()

def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [
        max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))))
        for lag in lags
    ]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5


# Setup
def make_technicals_agent(
    model,
    api_key: Optional[str] = None,
    market_index_df: Optional[pd.DataFrame] = None,
    related_securities_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    strategy_weights: Optional[Dict[str, float]] = None,
):
    """
    Creates a technical agent that combines multiple trading strategies:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use for reasoning and explanations.
    api_key : str, optional
        API key for accessing financial datasets. Defaults to None.
    market_index_df : pd.DataFrame, optional
        DataFrame containing market index data for relative strength calculations. Defaults to None.
    related_securities_dfs : dict of str -> pd.DataFrame, optional
        Dictionary of related securities with their respective DataFrames for correlation analysis. Defaults to None.
        strategy_weights (dict of str -> float, optional): Weights for each strategy in the combination.

    Returns
    -------
    function
        The technical agent function.
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
        reasoning: Optional[str]  # Reasoning or context built by the agent (for all tickers if applicable)
        signal: Optional[str]  # Overall signal for single ticker or aggregate
        confidence: Optional[float]  # Overall confidence score
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
        prices: Optional[List[Dict[str, any]]]  # Legacy field for prices (if single ticker)
        prices_data: Optional[Dict[str, List[Dict[str, any]]]]  # Dictionary of prices for each ticker

        # Analysis Results
        analysis_results: Optional[Dict[str, Dict[str, Any]]]  # Detailed analysis for each ticker
        overall_results: Optional[Dict[str, Dict[str, Any]]]  # Signal and confidence for each ticker
        reasoning_reports: Optional[Dict[str, str]]  # Detailed reasoning for each ticker
        overall_summary: Optional[str]  # Overall textual summary of the analysis
        report: Optional[Dict[str, str]]  # Complete combined report for all tickers
        reasoning_dict: Optional[Dict[str, Dict[str, Dict[str, Any]]]]  # Structured reasoning for each ticker

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
        market_index: Optional[pd.DataFrame]  # Market index data for relative strength calculations
        related_securities: Optional[Dict[str, pd.DataFrame]]  # Related securities data for correlation analysis
        strategy_weights: Optional[Dict[str, float]]  # Strategy weights for each strategy



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
            limit = details.get("limit", 1)  # Default limit # Not needed though

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
    
             
    def fetch_historical_prices_data(state: GraphState):
        """
        Fetch prices data based on the user's instructions.
        """
        print("    * FETCH HISTORICAL PRICES DATA")

        tickers = state.get("tickers")
        start_date = state.get("start_date")
        end_date = state.get("end_date")
        api_key = state.get("financial_data_api_key")

        prices_data = {}
        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")
                prices = get_prices(
                    api_key=api_key,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
                prices_data[ticker] = prices
                print(f"Data for {ticker}: {prices}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                prices_data[ticker] = None
                
        state["prices_data"] = prices_data
        
        print("\nAll fetched data:")
        for ticker, data in prices_data.items():
            print(f"{ticker}: {data}")
        
        return state



    def generate_analysis(
        state: GraphState,
        market_index_df: Optional[pd.DataFrame] = None,
        related_securities_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Generate a technical analysis for each ticker based on historical prices.

        Returns:
            dict: Updated state with overall signals, confidence, and reasoning reports for all tickers.
        """

        print("    * GENERATE ANALYSIS")

        prices_data = state.get("prices_data")

        if not prices_data:
            raise ValueError("No price data available for analysis.")

        overall_results = {}
        reasoning_reports = {}
        reasoning_dict = {}  # Initialize reasoning_dict for structured analysis

        for ticker, ticker_prices in prices_data.items():
            try:
                print(f"Analyzing data for {ticker}...")

                # Convert ticker prices to a DataFrame
                prices_df = prices_to_df(ticker_prices)

                # 1. Trend Following Strategy
                trend_signals = calculate_trend_signals(prices_df)

                # 2. Mean Reversion Strategy
                mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

                # 3. Momentum Strategy
                momentum_signals = calculate_momentum_signals(prices_df, market_index_df)

                # 4. Volatility Strategy
                volatility_signals = calculate_volatility_signals(prices_df)

                # 5. Statistical Arbitrage Signals
                stat_arb_signals = calculate_stat_arb_signals(prices_df, related_securities_dfs)

                # Combine all signals using a weighted ensemble approach
                strategy_weights = strategy_weights or {
                    "trend": 0.25,
                    "mean_reversion": 0.20,
                    "momentum": 0.25,
                    "volatility": 0.15,
                    "stat_arb": 0.15,
                }

                combined_signal = weighted_signal_combination(
                    {
                        "trend": trend_signals,
                        "mean_reversion": mean_reversion_signals,
                        "momentum": momentum_signals,
                        "volatility": volatility_signals,
                        "stat_arb": stat_arb_signals,
                    },
                    strategy_weights,
                )

                # Extract overall signal and confidence
                overall_signal = combined_signal['signal']
                confidence = combined_signal['confidence']

                # Generate Technical Analysis Report for this ticker
                analysis_report = f"""## **Technical Analysis Report for {ticker}**\n\n"""
                analysis_report += f"### **Overall Signal**: {overall_signal}\n"
                analysis_report += f"### **Confidence**: {round(confidence * 100)}%\n\n"
                analysis_report += "### **Strategy Signals Analysis**\n"

                # Initialize strategy_signals for reasoning_dict
                strategy_signals = {}

                # Add strategy-specific signals and metrics
                strategies = {
                    "Trend Following": trend_signals,
                    "Mean Reversion": mean_reversion_signals,
                    "Momentum": momentum_signals,
                    "Volatility": volatility_signals,
                    "Statistical Arbitrage": stat_arb_signals,
                }

                for strategy_name, signals in strategies.items():
                    analysis_report += f"- **{strategy_name} Strategy**:\n"
                    analysis_report += f"  - Signal: {signals['signal']}\n"
                    analysis_report += f"  - Confidence: {round(signals['confidence'] * 100)}%\n"
                    analysis_report += f"  - Metrics: {normalize_pandas(signals['metrics'])}\n\n"

                    # Add to strategy_signals for reasoning_dict
                    strategy_signals[strategy_name] = {
                        "Signal": signals["signal"],
                        "Confidence": round(signals["confidence"] * 100),
                        "Metrics": signals["metrics"],
                    }

                # Store the analysis for the ticker
                overall_results[ticker] = {"signal": overall_signal, "confidence": confidence}
                reasoning_reports[ticker] = analysis_report

                # Add structured data to reasoning_dict
                reasoning_dict[ticker] = {
                        "Overall Signal": overall_signal,
                        "Confidence": round(confidence * 100),
                        "Strategy Signals": strategy_signals,
                }

            except Exception as e:
                print(f"Error analyzing data for {ticker}: {e}")
                continue

        # Update state with results for all tickers
        full_report = ""
        for ticker, analysis_report in reasoning_reports.items():
            full_report += analysis_report + "\n" + "-" * 80 + "\n"

        state["report"] = full_report  # Save the consolidated report in the state
        state["overall_results"] = overall_results
        state["reasoning_reports"] = reasoning_reports
        state["reasoning_dict"] = reasoning_dict  # Save the structured reasoning dict in the state

        return state

    # Define the graph
    node_functions = {
        "parse_user_instructions": parse_user_instructions,
        "fix_parse_instructions_code": fix_parse_instructions_code,  # Ensure this node comes after parsing
        "fetch_historical_prices_data": fetch_historical_prices_data,  # Use the correct key
        "generate_analysis": generate_analysis,
    }

    # Corrected call to create_coding_agent_graph
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="parse_user_instructions",
        create_code_node_name="fetch_historical_prices_data",  # This now matches the corrected key
        execute_code_node_name="generate_analysis",
        fix_code_node_name="fix_parse_instructions_code",  # Comes after parsing
    )

    return app