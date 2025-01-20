import pandas as pd
import plotly.express as px

import streamlit as st

from database.db_management import db
from rag.embeddings import get_embedding
from plots.clusters import cluster_embeddings_hdbscan, plot_clusters_with_tsne


def get_line_plot() -> None:
    """
    Generates and displays a line plot of average latency per hour.

    Retrieves data from the `chatbot_history` table, computes hourly averages,
    and visualizes it using a Streamlit line chart.

    Returns:
        None
    """
    df_line_plot = db.ask_db(
        sql_query="""
SELECT 
    year,
    month,
    day,
    hour,
    AVG(latency) AS avg_latency
FROM 
    chatbot_history
GROUP BY 
    year, month, day, hour
ORDER BY 
    year, month, day, hour;
"""
    )
    df_line_plot["datetime"] = pd.to_datetime(
        df_line_plot[["year", "month", "day", "hour"]]
    )
    df_line_plot.set_index("datetime", inplace=True)
    st.title("Hourly Latency Plot Over All Days")
    if not df_line_plot.empty:
        st.line_chart(df_line_plot["avg_latency"])
    else:
        st.warning("No data available to plot. Please check your database.")


def get_gr_pie_plot() -> None:
    """
    Generates and displays a pie chart showing the distribution of safe vs unsafe queries.

    Retrieves data from the `chatbot_history` table, groups it by the `safe` column,
    and visualizes the count of safe and unsafe queries as a pie chart.

    Returns:
        None
    """
    df_plot = db.ask_db(
        sql_query="""
SELECT 
    safe, 
    count(*)
FROM 
    chatbot_history
GROUP BY safe
"""
    )
    if not df_plot.empty:
        df_plot["safe"] = df_plot["safe"].map({0: "Unsafe Queries", 1: "Safe Queries"})
        color_map = {"Safe Queries": "green", "Unsafe Queries": "#d6423a"}

        fig = px.pie(
            df_plot,
            names="safe",
            values="count(*)",
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        fig.update_traces(
            marker=dict(colors=[color_map[label] for label in df_plot["safe"]])
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.7,
            )
        )
    else:
        st.warning("No data available to plot. Please check your database.")
    st.plotly_chart(fig)


def get_count_models() -> None:
    """
    Retrieves and displays counts of generative and embedding models from the `chatbot_history` table.

    Queries the database to get the counts of different generative models and embedding models,
    renames columns for clarity, and displays them as tables in Streamlit.

    Returns:
        None
    """
    df_llm = db.ask_db(
        sql_query="""
    SELECT 
        generative_model, 
        count(*)
    FROM 
        chatbot_history
    GROUP BY generative_model
    """
    )
    df_llm = df_llm.rename(columns={"generative_model": "LLM", "count(*)": "Count"})
    df_llm.set_index("LLM", inplace=True)
    st.table(df_llm)
    df_embedding = db.ask_db(
        sql_query="""
    SELECT 
        embedding_model, 
        count(*)
    FROM 
        chatbot_history
    GROUP BY embedding_model
    """
    )
    df_embedding = df_embedding.rename(
        columns={"embedding_model": "Embeddings", "count(*)": "Count"}
    )
    df_embedding.set_index("Embeddings", inplace=True)
    st.table(df_embedding)


def get_kpi() -> tuple[int, float, float, float]:
    """
    Retrieves and calculates key performance indicators (KPIs) from the `chatbot_history` table.

    Queries the database to get data related to query price, energy usage, and GWP,
    then computes the total number of queries, total price, total energy usage, and total GWP.

    Returns:
        tuple[int, float, float, float]: A tuple containing the following KPIs:
            - Total number of queries (int)
            - Total price (float)
            - Total energy usage (float)
            - Total GWP (float)
    """
    df_kpi = db.ask_db(
        sql_query="""
SELECT 
    query_price,
    energy_usage,
    gwp
FROM 
    chatbot_history
"""
    )
    n_queries = df_kpi.shape[0]  # Total number of queries
    sum_price = sum(df_kpi["query_price"])  # Total price
    sum_energy_usage = sum(df_kpi["energy_usage"])  # Total energy usage
    sum_gwp = sum(df_kpi["gwp"])  # Total GWP

    return n_queries, sum_price, sum_energy_usage, sum_gwp


def get_scatter_clusters():
    """
    Retrieves queries from the `chatbot_history` table, generates embeddings,
    performs clustering, and displays a scatter plot of the clusters.

    Queries the database to retrieve the queries, generates their embeddings,
    performs clustering using HDBSCAN, and visualizes the results using t-SNE
    in a scatter plot.

    Returns:
        None
    """
    df_queries = db.ask_db(
        sql_query="""
SELECT 
    query
FROM 
    chatbot_history
"""
    )
    embeddings = get_embedding(df_queries["query"].to_list())
    clusters = cluster_embeddings_hdbscan(embeddings=embeddings)
    plot = plot_clusters_with_tsne(
        embeddings=embeddings,
        clusters=clusters,
        text_data=df_queries["query"].to_list(),
    )
    st.plotly_chart(plot)
