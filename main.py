import os

from data_preprocessing import load_and_preprocess
from kmeans_model import evaluate_clusters, train_kmeans
from gmm_model import evaluate_gmm_components, train_gmm
from visualization import (
    plot_elbow,
    plot_silhouette,
    plot_gmm_information_criteria,
    plot_clusters,
    plot_cluster_sizes
)
from model_evaluation import compare_models


def main():
    print("=" * 65)
    print("DỰ ÁN PHÂN KHÚC KHÁCH HÀNG SỬ DỤNG THẺ TÍN DỤNG")
    print("=" * 65)

    data_path = "CC GENERAL.csv"
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(results_dir, exist_ok=True)

    # 1. Đọc và tiền xử lý dữ liệu
    print("\n[1] ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Không tìm thấy {data_path}. "
            "Hãy đặt file CSV cùng thư mục với main.py."
        )

    df_clean, X_scaled = load_and_preprocess(data_path)

    print("Số khách hàng:", X_scaled.shape[0])
    print("Số biến đầu vào:", X_scaled.shape[1])

    # 2. Đánh giá số cụm K-Means
    print("\n[2] ĐÁNH GIÁ SỐ CỤM K-MEANS")

    kmeans_results = evaluate_clusters(X_scaled, max_k=10)

    plot_elbow(kmeans_results, plots_dir)
    plot_silhouette(kmeans_results, plots_dir)

    import pandas as pd

    pd.DataFrame(kmeans_results).to_csv(
        os.path.join(results_dir, "kmeans_evaluation.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 3. Đánh giá số thành phần GMM
    print("\n[3] ĐÁNH GIÁ SỐ THÀNH PHẦN GMM")

    gmm_results = evaluate_gmm_components(
        X_scaled,
        max_components=10
    )

    plot_gmm_information_criteria(gmm_results, plots_dir)

    pd.DataFrame(gmm_results).to_csv(
        os.path.join(results_dir, "gmm_evaluation.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 4. Huấn luyện mô hình
    # Dùng 4 cụm làm cấu hình ban đầu để so sánh hai mô hình.
    # Có thể điều chỉnh sau khi xem kết quả đánh giá.
    n_clusters = 4

    print(f"\n[4] HUẤN LUYỆN K-MEANS VỚI K={n_clusters}")
    kmeans_model, kmeans_labels = train_kmeans(
        X_scaled,
        n_clusters=n_clusters
    )

    print(f"\n[5] HUẤN LUYỆN GMM VỚI {n_clusters} THÀNH PHẦN")
    gmm_model, gmm_labels = train_gmm(
        X_scaled,
        n_components=n_clusters
    )

    # 6. So sánh mô hình
    print("\n[6] SO SÁNH CHẤT LƯỢNG PHÂN CỤM")

    comparison = compare_models(
        X_scaled,
        kmeans_labels,
        gmm_labels
    )

    comparison.to_csv(
        os.path.join(results_dir, "model_comparison.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 7. Lưu nhãn cụm cùng dữ liệu để phân tích đặc điểm khách hàng
    print("\n[7] LƯU KẾT QUẢ PHÂN CỤM")

    df_result = df_clean.copy()
    df_result["KMeans_Cluster"] = kmeans_labels
    df_result["GMM_Cluster"] = gmm_labels

    output_path = os.path.join(
        results_dir,
        "customer_segments.csv"
    )

    df_result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Số khách hàng theo K-Means:")
    print(df_result["KMeans_Cluster"].value_counts().sort_index())

    print("\nSố khách hàng theo GMM:")
    print(df_result["GMM_Cluster"].value_counts().sort_index())

    # 8. Trực quan hóa kết quả
    print("\n[8] TẠO BIỂU ĐỒ")

    kmeans_plot_dir = os.path.join(plots_dir, "kmeans")
    gmm_plot_dir = os.path.join(plots_dir, "gmm")

    plot_clusters(
        X_scaled,
        kmeans_labels,
        kmeans_plot_dir,
        "Customer Segmentation using K-Means (PCA)"
    )

    plot_cluster_sizes(
        kmeans_labels,
        kmeans_plot_dir,
        "K-Means: Number of Customers in Each Cluster"
    )

    plot_clusters(
        X_scaled,
        gmm_labels,
        gmm_plot_dir,
        "Customer Segmentation using GMM (PCA)"
    )

    plot_cluster_sizes(
        gmm_labels,
        gmm_plot_dir,
        "GMM: Number of Customers in Each Cluster"
    )

    print("\n" + "=" * 65)
    print("HOÀN THÀNH DỰ ÁN PHÂN KHÚC KHÁCH HÀNG")
    print(f"Kết quả CSV: {output_path}")
    print(f"Biểu đồ: {plots_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
