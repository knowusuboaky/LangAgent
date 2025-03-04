import io
import pandas as pd
import sqlalchemy as sql
from typing import Union, List, Dict

def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 30,
    skip_stats: bool = False,
) -> List[str]:
    """
    Generate a summary for one or more DataFrames. Accepts a single DataFrame, a list of DataFrames,
    or a dictionary mapping names to DataFrames.

    Parameters
    ----------
    dataframes : pandas.DataFrame or list of pandas.DataFrame or dict of (str -> pandas.DataFrame)
        - Single DataFrame: produce a single summary (returned within a one-element list).
        - List of DataFrames: produce a summary for each DataFrame, using index-based names.
        - Dictionary of DataFrames: produce a summary for each DataFrame, using dictionary keys as names.
    n_sample : int, default 30
        Number of rows to display in the "Data (first 30 rows)" section.
    skip_stats : bool, default False
        If True, skip the descriptive statistics and DataFrame info sections.
        
    Example:
    --------
    ``` python
    import pandas as pd
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    dataframes = {
        "iris": data.frame,
        "iris_target": data.target,
    }
    summaries = get_dataframe_summary(dataframes)
    print(summaries[0])
    ```

    Returns
    -------
    list of str
        A list of summaries, one for each provided DataFrame. Each summary includes:
        - Shape of the DataFrame (rows, columns)
        - Column data types
        - Missing value percentage
        - Unique value counts
        - First 30 rows
        - Descriptive statistics
        - DataFrame info output
    """

    summaries = []

    # --- Dictionary Case ---
    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(_summarize_dataframe(df, dataset_name, n_sample, skip_stats))

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(_summarize_dataframe(dataframes, "Single_Dataset", n_sample, skip_stats))

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(_summarize_dataframe(df, dataset_name, n_sample, skip_stats))

    else:
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries


def _summarize_dataframe(df: pd.DataFrame, dataset_name: str, n_sample=30, skip_stats=False) -> str:
    """Generate a summary string for a single DataFrame."""
    # 1. Convert dictionary-type cells to strings
    #    This prevents unhashable dict errors during df.nunique().
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))
    
    # 2. Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    # 3. Calculate missing value stats
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])

    # 4. Get column data types
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    # 5. Get unique value counts
    unique_counts = df.nunique()  # Will no longer fail on unhashable dict
    unique_counts_summary = "\n".join([f"{col}: {count}" for col, count in unique_counts.items()])

    # 6. Generate the summary text
    if not skip_stats:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}

        Unique Value Counts:
        {unique_counts_summary}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}

        Data Description:
        {df.describe().to_string()}

        Data Info:
        {info_text}
        """
    else:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}
        """
        
    return summary_text.strip()



def get_database_metadata(connection: Union[sql.engine.base.Connection, sql.engine.base.Engine],
                          n_samples: int = 10) -> str:
    """
    Collects metadata and sample data from a database, with safe identifier quoting and
    basic dialect-aware row limiting. Prevents issues with spaces/reserved words in identifiers.
    
    Parameters
    ----------
    connection : Union[sql.engine.base.Connection, sql.engine.base.Engine]
        An active SQLAlchemy connection or engine.
    n_samples : int
        Number of sample values to retrieve for each column.

    Returns
    -------
    str
        A formatted string with database metadata, including some sample data from each column.
    """

    # If a connection is passed, use it; if an engine is passed, connect to it
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    output = []
    try:
        # Grab the engine off the connection
        sql_engine = conn.engine
        dialect_name = sql_engine.dialect.name.lower()

        output.append(f"Database Dialect: {sql_engine.dialect.name}")
        output.append(f"Driver: {sql_engine.driver}")
        output.append(f"Connection URL: {sql_engine.url}")

        # Inspect the database
        inspector = sql.inspect(sql_engine)
        tables = inspector.get_table_names()
        output.append(f"Tables: {tables}")
        output.append(f"Schemas: {inspector.get_schema_names()}")

        # Helper to build a dialect-specific limit clause
        def build_query(col_name_quoted: str, table_name_quoted: str, n: int) -> str:
            """
            Returns a SQL query string to select N rows from the given column/table
            across different dialects (SQLite, MySQL, Postgres, MSSQL, Oracle, etc.)
            """
            if "sqlite" in dialect_name or "mysql" in dialect_name or "postgres" in dialect_name:
                # Common dialects supporting LIMIT
                return f"SELECT {col_name_quoted} FROM {table_name_quoted} LIMIT {n}"
            elif "mssql" in dialect_name:
                # Microsoft SQL Server syntax
                return f"SELECT TOP {n} {col_name_quoted} FROM {table_name_quoted}"
            elif "oracle" in dialect_name:
                # Oracle syntax
                return f"SELECT {col_name_quoted} FROM {table_name_quoted} WHERE ROWNUM <= {n}"
            else:
                # Fallback
                return f"SELECT {col_name_quoted} FROM {table_name_quoted} LIMIT {n}"

        # Prepare for quoting
        preparer = inspector.bind.dialect.identifier_preparer

        # For each table, get columns and sample data
        for table_name in tables:
            output.append(f"\nTable: {table_name}")
            # Properly quote the table name
            table_name_quoted = preparer.quote_identifier(table_name)

            for column in inspector.get_columns(table_name):
                col_name = column["name"]
                col_type = column["type"]
                output.append(f"  Column: {col_name} Type: {col_type}")

                # Properly quote the column name
                col_name_quoted = preparer.quote_identifier(col_name)

                # Build a dialect-aware query with safe quoting
                query = build_query(col_name_quoted, table_name_quoted, n_samples)

                # Read a few sample values
                df = pd.read_sql(sql.text(query), conn)
                first_values = df[col_name].tolist()
                output.append(f"    First {n_samples} Values: {first_values}")

    finally:
        # Close connection if created inside the function
        if is_engine:
            conn.close()

    return "\n".join(output)