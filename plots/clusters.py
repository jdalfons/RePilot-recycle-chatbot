from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE


def cluster_embeddings_hdbscan(
    embeddings: NDArray[np.float32], min_cluster_size: int = 2, min_samples: int = 1
) -> NDArray[np.float32]:
    """
    Applies HDBSCAN clustering on the given embeddings.

    Args:
        embeddings (NDArray): A 2D numpy array of embeddings to be clustered.
        min_cluster_size (int, optional): Minimum size of clusters. Defaults to 2.
        min_samples (int, optional): Minimum samples in a cluster. Defaults to 1.

    Returns:
        NDArray: An array of cluster labels for the embeddings.
    """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = clusterer.fit_predict(embeddings)

    return clusters


def plot_clusters_with_tsne(
    embeddings: NDArray[np.float32],
    clusters: list[int],
    text_data: Optional[list[str]] = None,
) -> go.Figure():
    """
    Reduces the dimensions of embeddings using t-SNE and plots the resulting clusters.

    Args:
        embeddings (NDArray): A 2D numpy array of embeddings to be visualized.
        clusters (list[int]): Cluster labels corresponding to the embeddings.
        text_data (Optional[list[str]], optional): Optional text data for hover tooltips. Defaults to None.

    Returns:
        go.Figure: A Plotly scatter plot of the clusters.
    """
    if len(embeddings):
        if len(embeddings) < 5:
            perplexity = len(embeddings) - 1
        else:
            perplexity = 5
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        df = pd.DataFrame(tsne_results, columns=["Axis 1", "Axis 2"])
        df["Clusters"] = [str(x) for x in clusters]

        if text_data is not None:
            df["Text"] = text_data

        fig = px.scatter(
            df,
            x="Axis 1",
            y="Axis 2",
            color="Clusters",
            hover_data={"Text": True},
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="User Query Mapping",
            labels={"Label": "Clusters"},
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
