import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def plot_elbow(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["inertia"], marker="o")
    ax.set_title("Elbow Method - K-Means")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "elbow_method.png"),
        dpi=150
    )
    plt.close(fig)


def plot_silhouette(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["silhouette"], marker="o")
    ax.set_title("Silhouette Score - K-Means")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "silhouette_score.png"),
        dpi=150
    )
    plt.close(fig)


def plot_gmm_information_criteria(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results["n_components"],
        results["bic"],
        marker="o",
        label="BIC"
    )
    ax.plot(
        results["n_components"],
        results["aic"],
        marker="s",
        label="AIC"
    )

    ax.set_title("GMM Model Selection")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Information criterion (lower is better)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "gmm_aic_bic.png"),
        dpi=150
    )
    plt.close(fig)


def plot_clusters(X_scaled, labels, output_dir, title):
    os.makedirs(output_dir, exist_ok=True)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="viridis",
        s=12,
        alpha=0.7
    )

    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "customer_clusters.png"),
        dpi=150
    )
    plt.close(fig)


def plot_cluster_sizes(labels, output_dir, title):
    os.makedirs(output_dir, exist_ok=True)

    sizes = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of customers")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "cluster_sizes.png"),
        dpi=150
    )
    plt.close(fig)
