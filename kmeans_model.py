
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def evaluate_clusters(X_scaled, max_k=10):
    """Đánh giá K-Means với nhiều số cụm khác nhau."""

    n_samples = len(X_scaled)
    if n_samples < 3:
        raise ValueError("Cần ít nhất 3 khách hàng để đánh giá.")

    max_k = min(max_k, n_samples - 1)

    results = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": []
    }

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        # Bỏ qua trường hợp mô hình chỉ tạo ra một cụm
        if len(set(labels)) < 2:
            continue

        results["k"].append(k)
        results["inertia"].append(model.inertia_)
        results["silhouette"].append(
            silhouette_score(X_scaled, labels)
        )
        results["davies_bouldin"].append(
            davies_bouldin_score(X_scaled, labels)
        )
        results["calinski_harabasz"].append(
            calinski_harabasz_score(X_scaled, labels)
        )

        print(
            f"K={k}: "
            f"Inertia={model.inertia_:.2f}, "
            f"Silhouette={results['silhouette'][-1]:.4f}, "
            f"Davies-Bouldin={results['davies_bouldin'][-1]:.4f}, "
            f"Calinski-Harabasz="
            f"{results['calinski_harabasz'][-1]:.2f}"
        )

    if not results["k"]:
        raise ValueError("Không tìm được kết quả phân cụm hợp lệ.")

    return results


def train_kmeans(X_scaled, n_clusters=4):
    """Huấn luyện K-Means và trả về mô hình cùng nhãn cụm."""

    if len(X_scaled) < n_clusters:
        raise ValueError(
            "Số khách hàng phải lớn hơn hoặc bằng số cụm."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    return model, labels
