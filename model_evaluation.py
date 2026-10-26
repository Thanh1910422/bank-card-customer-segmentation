import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def evaluate_model(X_scaled, labels, model_name):
    """Tính các chỉ số đánh giá cho một mô hình phân cụm."""

    n_clusters = len(set(labels))

    if n_clusters < 2 or n_clusters >= len(X_scaled):
        raise ValueError(
            f"{model_name} cần ít nhất 2 cụm và ít nhất "
            "một mẫu nhiều hơn số cụm."
        )

    return {
        "Model": model_name,
        "Number_of_Clusters": n_clusters,
        "Silhouette": silhouette_score(X_scaled, labels),
        "Davies_Bouldin": davies_bouldin_score(
            X_scaled, labels
        ),
        "Calinski_Harabasz": calinski_harabasz_score(
            X_scaled, labels
        )
    }


def compare_models(X_scaled, kmeans_labels, gmm_labels):
    """So sánh hai mô hình và trả về bảng kết quả."""

    results = [
        evaluate_model(X_scaled, kmeans_labels, "K-Means"),
        evaluate_model(X_scaled, gmm_labels, "GMM")
    ]

    comparison = pd.DataFrame(results)

    print("\nBẢNG SO SÁNH K-MEANS VÀ GMM")
    print(comparison.to_string(index=False))

    return comparison
