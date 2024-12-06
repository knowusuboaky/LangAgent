# langagent 2.1.5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/langagent.svg)](https://pypi.org/project/langagent/)
[![langagent Version](https://img.shields.io/pypi/v/langagent.svg)](https://pypi.org/project/langagent/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/knowusuboaky/langagent/main.yml)](https://github.com/knowusuboaky/langagent/actions)
[![Issues](https://img.shields.io/github/issues/knowusuboaky/langagent)](https://github.com/knowusuboaky/langagent/issues)
[![Contact](https://img.shields.io/badge/Email-Contact-blue.svg)](mailto:kwadwo.owusuboakye@outlook.com)

**langagent** is a versatile multi-agent system designed to automate and streamline a wide range of complex tasks, including research, code generation, logical reasoning, data analysis, and dynamic reporting. Powered by advanced language models, **langagent** integrates seamlessly with APIs, databases, and various data formats, making it an essential tool for developers, data analysts, and business professionals to optimize workflows and extract valuable insights.

## Key Features

- **Research Automation**: Automatically gather, analyze, and summarize information from multiple sources such as web content, databases, and documents.
- **Code Generation & Debugging**: Generate, debug, and optimize code for various programming languages, supporting complex multi-step tasks.
- **Mathematical Reasoning**: Solve logical and mathematical queries with clear step-by-step breakdowns.
- **Data Analysis & Topic Generation**: Perform data clustering, trend analysis, and generate actionable insights from structured and unstructured data.
- **Professional Reporting**: Create detailed, high-quality reports with dynamic data visualizations and clear summaries.
- **Supervisor Chain**: Manage and coordinate workflows between multiple agents for complex tasks.

## Installation

You can install the `langagent` package via `pip`:

```bash
pip install langagent==2.1.1
```

## Getting Started

To start using the **langagent** library, you need to import the agents from their respective teams. Below is a detailed guide explaining each agent, its arguments, and how to effectively use them in your projects.

### Research Team Agents

The **Research Team** consists of three agents: **Researcher**, **Coder**, and **Weather**. Each agent is designed to assist with gathering research, generating code, or fetching weather data.

1. **Researcher Agent**
    - The researcher agent gathers and synthesizes information based on user queries. It automates the research process by providing structured and concise insights.
    
    **Arguments**:
    - `llm`: The language model to be used (e.g., `ChatOpenAI`).
    - `messages`: A list of `HumanMessage` objects representing the research query.

    **Example**:
    ```python
    from langagent.research_team.agents import researcher
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI()
    researcher_agent = researcher(llm)

    # Invoke the agent with a query
    result = researcher_agent.invoke({"messages": [HumanMessage(content="Who is Messi?")]})
    print(result['output'])
    ```

#### `Example of Resarcher Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-earch.png?raw=true" width="778" height="789" alt="Optional Alt Text"> 

2. **Coder Agent**
    - The coder agent generates, debugs, and optimizes code based on user input. It handles various programming challenges, from basic functions to advanced tasks.
    
    **Arguments**:
    - `llm`: The language model you want to use.
    - `messages`: A list of `HumanMessage` objects representing the code generation task.

    **Example**:
    ```python
    from langagent.research_team.agents import coder
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI()
    coder_agent = coder(llm)

    # Ask the agent to generate code
    result = coder_agent.invoke({"messages": [HumanMessage(content="Write a Python function to calculate the factorial of a number.")]} )
    print(result['output'])
    ```

#### `Example of Coder Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-cod.png?raw=true" width="447" height="544" alt="Optional Alt Text"> 

3. **Weather Agent**
    - The weather agent fetches and interprets weather data for a specified location using API integrations.
    
    **Arguments**:
    - `llm`: The language model to be used.
    - `OPENWEATHERMAP_API_KEY`: Your OpenWeatherMap API key.
    - `messages`: A list of `HumanMessage` objects representing the weather query.

    **Example**:
    ```python
    from langagent.research_team.agents import weather
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI()
    weather_agent = weather(llm, OPENWEATHERMAP_API_KEY="your-api-key-here")

    # Ask for weather details
    result = weather_agent.invoke({"messages": [HumanMessage(content="What is the weather today in New York?")]})
    print(result['output'])
    ```

#### `Example of Weather Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-openwea.png?raw=true" width="757" height="792" alt="Optional Alt Text"> 

### Logic Team Agents

The **Logic Team** consists of two agents: **Reasoner** and **Calculator**, both designed to solve logical and mathematical problems efficiently.

1. **Reasoner Agent**
    - The reasoner agent handles logical reasoning tasks and provides clear, step-by-step explanations for complex problems.
    
    **Arguments**:
    - `llm`: The language model to be used.
    - `messages`: A list of `HumanMessage` objects describing the logic problem.

    **Example**:
    ```python
    from langagent.logic_team.agents import reasoner
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI()
    reasoner_agent = reasoner(llm)

    # Solve a logic problem
    result = reasoner_agent.invoke({"messages": [HumanMessage(content="I have a 7 in the tens place. I have an even number in the ones place. I am lower than 74. What number am I?")]})
    print(result['output'])
    ```

#### `Example of Reasoner Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-rea.png?raw=true" width="408" height="548" alt="Optional Alt Text"> 

2. **Calculator Agent**
    - The calculator agent solves mathematical queries and calculations based on natural language inputs.
    
    **Arguments**:
    - `llm`: The language model to be used.
    - `messages`: A list of `HumanMessage` objects containing the mathematical query.

    **Example**:
    ```python
    from langagent.logic_team.agents import calculator
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI()
    calculator_agent = calculator(llm)

    # Solve a mathematical problem
    result = calculator_agent.invoke({"messages": [HumanMessage(content="Calculate the square root of 16")]})
    print(result['output'])
    ```

#### `Example of Calculator Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-cal.png?raw=true" width="410" height="549" alt="Optional Alt Text"> 

### Analysis Team Agents

The **Analysis Team** consists of two agents: **Topic Generator** and **SQL Databaser**, both designed to provide insights from data, either through text clustering or SQL-based analysis.


1. **Topic Generator Agent**
    - The topic generator agent clusters and analyzes text data, producing topics and visualizations based on user data.
    
    **Arguments**:
    - `llm`: The language model to be used.
    - `path`: The path to the CSV file containing text data.
    - `text_column`: The column in the CSV that contains the text data.
    - `user_topics`: Optional, user-provided topics for clustering.

    **Example**:
    ```python
    from langagent.analysis_team.agents import topic_generator
    from sentence_transformers import SentenceTransformer
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI()
    topic_generator_agent = topic_generator(llm)

    inputs = {
        'path': '../Comments.csv',
        'text_column': 'Comments',
        'user_topics': None
    }

    result = topic_generator_agent.invoke(inputs)
    result['summary_df']  # View the summarized data
    result['fig'].show()  # Show the visualization
    ```

#### `Example of Topic Generator Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-topic.png?raw=true" width="335" height="730" alt="Optional Alt Text"> 

2. **SQL Databaser Agent**
    - The SQL databaser agent executes SQL queries and provides insights from the results, including generating charts.
    
    **Arguments**:
    - `db_path`: The path to your SQLite or SQL database.
    - `llm`: The language model.
    - `user_question`: The question that drives the SQL query and chart generation.

    **Example**:
    ```python
    from langagent.analysis_team.agents import sql_databaser
    from langchain_openai import ChatOpenAI
    from langchain_experimental.utilities import PythonREPL

    llm = ChatOpenAI()
    PATH_DB = "sqlite:///database/leads_scored.db"
    sql_databaser_agent = sql_databaser(db_path=PATH_DB, llm=llm)

    question = """
    What are the total sales by month-year? 
    Use suggested price as a proxy for revenue for each transaction and a quantity of 1. 
    Make a line chart of sales over time.
    """

    initial_input = {
        "user_question": question
    }

    # Invoke the agent with the input
    result = sql_databaser_agent.invoke(initial_input)
    print(result['sql_query'])  # SQL Query
    print(result['summary'])  # Summary of results

    # Execute the Python code for chart generation
    repl = PythonREPL()
    repl.run(result['chart_plotly_code'])
    ```

#### `Example of SQL Databaser Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-sql.png?raw=true" width="502" height="870" alt="Optional Alt Text"> 

### Reporting Team Agents

The **Reporting Team** consists of two agents: **Interpreter** and **Summarizer**, both designed to provide insights and summaries of outputs from various sources such as data, charts, or documents.

1. **Interpreter Agent**
    - The interpreter agent provides clear, insightful interpretations of various types of outputs, including visual plots, tables, and data.
    
    **Arguments**:
    - `llm`: The language model.
    - `code_output`: The result or output that needs to be interpreted.

    **Example**:
    ```python
    from langagent.reporting_team.agents import interpreter
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI()
    interpreter_agent = interpreter(llm)

    result = interpreter_agent.invoke({"code_output": "Bar chart showing sales data."})
    print(result['output'])
    ```

#### `Example of Interpreter Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-inte.png?raw=true" width="403" height="546" alt="Optional Alt Text"> 

2. **Summarizer Agent**
    - The summarizer agent summarizes documents like PDFs, PPT, DOCX, TXT files or Plain Text into concise, actionable insights.
    
    **Arguments**:
    - `llm`: The language model.
    - `input_data`: The path to the document or text file that needs to be summarized.

    **Example**:
    ```python
    from langagent.reporting_team.agents import summarizer
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI()
    summarizer_agent = summarizer(llm)

    inputs = {
        'input_data': '../Comments.csv'
    }

    result = summarizer_agent.invoke(inputs)
    print(result['summary'])
    ```

#### `Example of Summarizer Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-sum.png?raw=true" width="358" height="535" alt="Optional Alt Text"> 

### Supervisor Chain

The **Supervisor Chain** manages the workflow between multiple agents. It selects the next agent based on predefined criteria or completion conditions. This is useful when you want to automate multi-step tasks and route tasks to different agents based on the flow of the conversation or task requirements.

- The **Supervisor Chain** allows you to manage complex workflows by selecting different agents based on their capabilities or status.

**Arguments**:
- `subagent_names`: A list of subagents (workers) that will perform tasks.
- `llm`: The language model that powers decision-making.
- `subagent_descriptions`: A dictionary of subagent names with descriptions of their roles (optional).
- `completion_criteria`: A string representing the criteria that signals completion of the workflow.
- `history_depth`: The number of previous messages to consider when making routing decisions (optional).

**Example**:
```python
from langagent.supervisor import supervisor_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

# Define the subagents and their descriptions
subagent_names = ["researcher", "coder", "calculator"]
subagent_descriptions = {
    "researcher": "Finds relevant research and gathers data.",
    "coder": "Generates, debugs, and optimizes code.",
    "calculator": "Solves mathematical problems and queries."
}

# Create a supervisor chain to manage the workflow
supervisor = supervisor_chain(
    subagent_names=subagent_names, 
    llm=llm, 
    subagent_descriptions=subagent_descriptions, 
    completion_criteria="FINISH"
)
```

#### `Example of Supervision Workflow`

<img src="https://github.com/knowusuboaky/langagent/blob/main/README_files/figure-markdown/mermaid-figure-4.png?raw=true" width="802" height="481" alt="Optional Alt Text">

## Dependencies

langagent relies on several core libraries, including:
- **LangChain** for agent creation and task management.
- **Sentence Transformers** for text embedding.
- **SQLAlchemy** for database interactions.
- **Pandas** for data manipulation.
- **Plotly** and **Matplotlib** for data visualization.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions to langagent! If you'd like to contribute:
1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push your branch (`git push origin feature/new-feature`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please contact:

- **Author**: Kwadwo Daddy Nyame Owusu - Boakye
- **Email**: [kwadwo.owusuboakye@outlook.com](mailto:kwadwo.owusuboakye@outlook.com)
- **GitHub**: [https://github.com/knowusuboaky/langagent](https://github.com/knowusuboaky/langagent)
