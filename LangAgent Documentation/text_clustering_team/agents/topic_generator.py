# LangAgent/analysis_team/agents/topic_generator.py


# * LIBRARIES

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from tqdm import tqdm
import time
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
import warnings
import yaml

warnings.filterwarnings("ignore")


def create_topic_generator(
    llm, 
    embedding_model=None,
    # Summary prompts
    summary_system_prompt=None, 
    summary_user_prompt=None,
    # Topic generator prompts: with user_topics
    topic_system_prompt_with_user_topics=None,
    topic_user_prompt_with_user_topics=None,
    # Topic generator prompts: without user_topics
    topic_system_prompt_no_user_topics=None,
    topic_user_prompt_no_user_topics=None
):
    """
    Factory function that returns a compiled StateGraph `app`.

    You can customize:
      - summary_system_prompt
      - summary_user_prompt
      - topic_system_prompt_with_user_topics
      - topic_user_prompt_with_user_topics
      - topic_system_prompt_no_user_topics
      - topic_user_prompt_no_user_topics

    If any of these prompts are None, the function uses default prompts.

    Usage:
        from your_module import create_topic_generator

        app = create_topic_generator(
            llm=my_llm_model,
            summary_system_prompt="Your custom system prompt for summary",
            summary_user_prompt="Your custom user prompt for summary",
            topic_system_prompt_with_user_topics="...",
            topic_user_prompt_with_user_topics="...",
            topic_system_prompt_no_user_topics="...",
            topic_user_prompt_no_user_topics="..."
        )

        # Then create an initial state and run the app:
        initial_state = {
            "path": "path_to_file.csv",
            "text_column": "review_text",
            "df": None,
            "embedding_df": None,
            "cluster_results": None,
            "optimal_clusters": 0,
            "cluster_labels": [],
            "summary_df": None,
            "user_topics": ["Topic1", "Topic2"],  # or None/empty if you want to auto-generate topics
            "fig": None
        }
        output_state = app.run(initial_state)
    """

    # ------------------- DEFAULT PROMPTS ------------------- #
    DEFAULT_SUMMARY_SYSTEM_PROMPT = (
        "You are an expert summarizer. Your task is to generate precise and concise summaries."
    )
    DEFAULT_SUMMARY_USER_PROMPT = (
        "Generate a ten-word summary of the text below:\nText: {transcript}"
    )

    DEFAULT_TOPIC_SYSTEM_PROMPT_WITH_USER_TOPICS = (
        "You are an expert in {user_request}, tasked with analyzing and categorizing diverse "
        "user-generated content. Your goal is to accurately assign each summary to one of "
        "the following topics: {topics_str}. Ensure that your categorizations are precise "
        "and based on a deep understanding of the context."
    )
    DEFAULT_TOPIC_USER_PROMPT_WITH_USER_TOPICS = (
        "Based on the summaries provided, match each summary to the most relevant topic from the "
        "list below. Summaries: {{{column_name}}}.\n\nPlease return your response in the format "
        "'Summary: Topic'."
    )

    DEFAULT_TOPIC_SYSTEM_PROMPT_NO_USER_TOPICS = (
        "You are an expert in {user_request}, specialized in analyzing user-generated content such "
        "as reviews, comments, feedback, and discussions. Your task is to generate a concise and "
        "informative topic title that accurately summarizes a set of summaries in the given context."
    )
    DEFAULT_TOPIC_USER_PROMPT_NO_USER_TOPICS = (
        "Review the following summaries and generate a clear, concise topic title that encapsulates "
        "the main themes:\n\nSummaries: {{{column_name}}}.\n\nTOPIC TITLE:"
    )

    # ------------------- USE PROVIDED PROMPTS OR FALLBACKS ------------------- #
    summary_system_prompt = summary_system_prompt or DEFAULT_SUMMARY_SYSTEM_PROMPT
    summary_user_prompt = summary_user_prompt or DEFAULT_SUMMARY_USER_PROMPT

    topic_system_prompt_with_user_topics = (
        topic_system_prompt_with_user_topics or DEFAULT_TOPIC_SYSTEM_PROMPT_WITH_USER_TOPICS
    )
    topic_user_prompt_with_user_topics = (
        topic_user_prompt_with_user_topics or DEFAULT_TOPIC_USER_PROMPT_WITH_USER_TOPICS
    )

    topic_system_prompt_no_user_topics = (
        topic_system_prompt_no_user_topics or DEFAULT_TOPIC_SYSTEM_PROMPT_NO_USER_TOPICS
    )
    topic_user_prompt_no_user_topics = (
        topic_user_prompt_no_user_topics or DEFAULT_TOPIC_USER_PROMPT_NO_USER_TOPICS
    )

    # ------------------- CORE FUNCTIONS ------------------- #

    def load_document(path):
        file_type = path.split('.')[-1]
        if file_type == "csv":
            df = pd.read_csv(path)
        elif file_type == "xlsx":
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type.")
        return df

    # Specify the Column of Data for Text Embedding
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def apply_embeddings(df, text_column):
        df[text_column] = df[text_column].fillna("None")
        df['embedding'] = df[text_column].apply(lambda x: embedding_model.encode(str(x)))
        df['embedding'] = df['embedding'].apply(np.array)
        return df

    # Cluster Optimization
    def calculate_silhouette_scores(data_matrix, min_clusters=3, max_clusters=25):
        cluster_results_km = pd.DataFrame(columns=['k', 'score'])
        for k in tqdm(range(min_clusters, max_clusters + 1)):
            km_model = KMeans(n_clusters=k, init='k-means++', random_state=42)
            y = km_model.fit_predict(data_matrix)
            silhouette = silhouette_score(data_matrix, y)
            dic = {'k': [k], 'score': [silhouette]}
            cluster_results_km = pd.concat([cluster_results_km, pd.DataFrame(dic)])    
        return cluster_results_km

    def find_optimal_cluster(cluster_results):
        cluster_results = cluster_results.reset_index(drop=True)
        optimal_cluster = cluster_results['score'].idxmax()
        optimal_cluster = cluster_results['k'].iloc[optimal_cluster]
        return optimal_cluster

    # K-Means Clustering
    def perform_kmeans_clustering(data_matrix, num_clusters):
        km_model = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
        cluster_labels = km_model.fit_predict(data_matrix)
        return cluster_labels

    # Visualizing Embedded Vectors in 2-D Space Using t-SNE with Plotly
    def visualize_embeddings_tsne_plotly(data_matrix, cluster_labels, num_clusters):
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
        vis_dims2 = tsne.fit_transform(data_matrix)
        
        df_vis = pd.DataFrame({
            'x': vis_dims2[:, 0],
            'y': vis_dims2[:, 1],
            'Cluster': cluster_labels
        })
        
        fig = px.scatter(
            df_vis, x='x', y='y', color='Cluster', 
            title="Reduced Dimension: Embeddings visualized using t-SNE",
            labels={'x': 'Axis 1', 'y': 'Axis 2'},
            color_continuous_scale=px.colors.sequential.Viridis,
            symbol_sequence=['circle', 'diamond', 'square', 'triangle-up', 'triangle-down', 'cross', 'x']
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7),
                        selector=dict(mode='markers'))
        
        return fig  # Return the figure object


    # Summarize Text
    def summarize_text(transcript):
        # Use the prompts passed to create_topic_generator
        system_prompt = summary_system_prompt
        # Format the user prompt with the text to be summarized
        user_prompt = summary_user_prompt.format(transcript=transcript)
        
        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(user_prompt)
            ]
        )
        
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        response = llm_chain.run({})
        
        return response.strip()

    # Generate Topic Title Prompt
    def get_prompt(user_request, column_name, user_topics=None):
        """
        Dynamically build the system and user prompts for topic generation
        depending on whether user_topics is provided or not.
        """
        if user_topics:
            # We have a list of user topics
            topics_str = ', '.join(user_topics)

            # Format the system prompt (with user_topics)
            system_prompt = topic_system_prompt_with_user_topics.format(
                user_request=user_request,
                topics_str=topics_str
            )
            # Format the user prompt (with user_topics)
            user_prompt = topic_user_prompt_with_user_topics.format(column_name=column_name)

        else:
            # user_topics is None; generate a topic title
            system_prompt = topic_system_prompt_no_user_topics.format(
                user_request=user_request
            )
            user_prompt = topic_user_prompt_no_user_topics.format(column_name=column_name)
        
        return ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(user_prompt)
            ],
            input_variables=[column_name],
        )

    def generate_topic_title(df, user_request, column_name, user_topics=None):
        output_column = f"{column_name}_Topic"
        for c in df["Cluster"].unique():
            prompt_template = get_prompt(user_request, column_name, user_topics)
            chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
            combined_text = "\n".join(
                [
                    row['Summary_Model']
                    for row in df.query(f"Cluster == {c}").to_dict(orient="records")
                ]
            )
            result = chain.run({column_name: combined_text})
            
            if user_topics:
                categorized_topics = []
                for line in result.strip().split('\n'):
                    if ':' in line:
                        summary, topic = line.split(':', 1)
                        categorized_topics.append(topic.strip())
                if categorized_topics:
                    df.loc[df["Cluster"] == c, output_column] = ', '.join(categorized_topics)
                else:
                    df.loc[df["Cluster"] == c, output_column] = result.strip()
            else:
                df.loc[df["Cluster"] == c, output_column] = result.strip()

    def clustering_topics(df, user_request, column_name, chunk_size=100, sleep_time=60, user_topics=None):
        df1 = pd.DataFrame()
        num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
        
        for i in range(num_chunks):
            init = i * chunk_size
            final = (i + 1) * chunk_size
            dk = df.iloc[init:final]
            print(f"Processing chunk {i + 1}/{num_chunks}: rows {init} to {final}")
            dk['Summary_Model'] = dk[column_name].map(lambda x: summarize_text(x) if pd.notnull(x) else "")
            df1 = pd.concat([df1, dk])
            time.sleep(sleep_time)
            del dk

        generate_topic_title(df1, user_request, column_name, user_topics)
        return df1

    # Define the workflow using Langraph

    class GraphState(TypedDict):
        path: str
        text_column: str
        df: pd.DataFrame
        embedding_df: pd.DataFrame
        cluster_results: pd.DataFrame
        optimal_clusters: int
        cluster_labels: List[int]
        summary_df: pd.DataFrame
        user_topics: List[str]  # Add user_topics to the state
        fig: Figure  # Add a field for the figure object


    def preprocess_document(state: GraphState) -> GraphState:
        print("---PREPROCESS DOCUMENT---")
        time.sleep(1)
        return state

    def load_document_node(state: GraphState) -> GraphState:
        print("---LOAD DOCUMENT---")
        state["df"] = load_document(state["path"])
        time.sleep(1)
        return state

    def apply_embeddings_node(state: GraphState) -> GraphState:
        print("---APPLY EMBEDDINGS---")
        state["embedding_df"] = apply_embeddings(state["df"], state["text_column"])
        time.sleep(1)
        return state

    def calculate_silhouette_scores_node(state: GraphState) -> GraphState:
        print("---CALCULATE SILHOUETTE SCORES---")
        data_matrix = np.vstack(state["embedding_df"]['embedding'].values)
        state["cluster_results"] = calculate_silhouette_scores(data_matrix)
        time.sleep(1)
        return state

    def find_optimal_cluster_node(state: GraphState) -> GraphState:
        print("---FIND OPTIMAL CLUSTER---")
        state["optimal_clusters"] = find_optimal_cluster(state["cluster_results"])
        time.sleep(1)
        return state

    def perform_kmeans_clustering_node(state: GraphState) -> GraphState:
        print("---PERFORM KMEANS CLUSTERING---")
        data_matrix = np.vstack(state["embedding_df"]['embedding'].values)
        state["cluster_labels"] = perform_kmeans_clustering(data_matrix, state["optimal_clusters"])
        state["df"]["Cluster"] = state["cluster_labels"]
        time.sleep(1)
        return state

    def visualize_embeddings_tsne_plotly_node(state: GraphState) -> GraphState:
        print("---VISUALIZE EMBEDDINGS---")
        data_matrix = np.vstack(state["embedding_df"]['embedding'].values)
        fig = visualize_embeddings_tsne_plotly(data_matrix, state["cluster_labels"], state["optimal_clusters"])
        state["fig"] = fig  # Save the figure object to the state
        time.sleep(1)
        return state

    def summarize_text_node(state: GraphState) -> GraphState:
        print("---SUMMARIZE TEXT---")
        state["summary_df"] = state["df"].copy()
        state["summary_df"]['Summary_Model'] = state["summary_df"][state["text_column"]].map(lambda x: summarize_text(x) if pd.notnull(x) else "")
        time.sleep(1)
        return state

    def clustering_topics_node(state: GraphState) -> GraphState:
        print("---CLUSTERING TOPICS---")
        # Pass user_topics from the state into the clustering_topics function
        state["summary_df"] = clustering_topics(
            state["df"], 
            "Text Categorization and Insights Generation", 
            state["text_column"], 
            user_topics=state["user_topics"]  # Ensure user_topics are passed
        )
        time.sleep(1)
        return state

    def generate_topic_title_node(state: GraphState) -> GraphState:
        print("---GENERATE TOPIC TITLE---")
        # Pass user_topics from the state into the generate_topic_title function
        generate_topic_title(
            df=state["summary_df"], 
            user_request="Text Categorization and Insights Generation", 
            column_name=state["text_column"],
            user_topics=state["user_topics"]  # Ensure user_topics are passed
        )
        time.sleep(1)
        return state


    def state_printer(state: GraphState) -> GraphState:
        """Print the state"""
        print("---STATE PRINTER---")
        #print(f"Path: {state.get('path')}")
        #print(f"Text Column: {state.get('text_column')}")
        #print(f"DataFrame: {state.get('df').head()}")
        #print(f"Embedding DataFrame: {state.get('embedding_df').head()}")
        #print(f"Cluster Results: {state.get('cluster_results').head()}")
        #print(f"Optimal Clusters: {state.get('optimal_clusters')}")
        #print(f"Cluster Labels: {state.get('cluster_labels')[:10]}")
        #print(f"Summary DataFrame: {state.get('summary_df').head()}")
        #print(f"Num Steps: {state['num_steps']}")
        return state

    # Workflow DAG
    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_document", preprocess_document)
    workflow.add_node("load_document_node", load_document_node)
    workflow.add_node("apply_embeddings_node", apply_embeddings_node)
    workflow.add_node("calculate_silhouette_scores_node", calculate_silhouette_scores_node)
    workflow.add_node("find_optimal_cluster_node", find_optimal_cluster_node)
    workflow.add_node("perform_kmeans_clustering_node", perform_kmeans_clustering_node)
    workflow.add_node("visualize_embeddings_tsne_plotly_node", visualize_embeddings_tsne_plotly_node)
    workflow.add_node("summarize_text_node", summarize_text_node)
    workflow.add_node("generate_topic_title_node", generate_topic_title_node)
    workflow.add_node("clustering_topics_node", clustering_topics_node)
    workflow.add_node("state_printer", state_printer)

    workflow.set_entry_point("preprocess_document")

    workflow.add_edge("preprocess_document", "load_document_node")
    workflow.add_edge("load_document_node", "apply_embeddings_node")
    workflow.add_edge("apply_embeddings_node", "calculate_silhouette_scores_node")
    workflow.add_edge("calculate_silhouette_scores_node", "find_optimal_cluster_node")
    workflow.add_edge("find_optimal_cluster_node", "perform_kmeans_clustering_node")
    workflow.add_edge("perform_kmeans_clustering_node", "visualize_embeddings_tsne_plotly_node")
    workflow.add_edge("visualize_embeddings_tsne_plotly_node", "summarize_text_node")
    workflow.add_edge("summarize_text_node", "generate_topic_title_node")
    workflow.add_edge("generate_topic_title_node", "clustering_topics_node")
    workflow.add_edge("clustering_topics_node", "state_printer")
    workflow.add_edge("state_printer", END)

    app = workflow.compile()

    return app


