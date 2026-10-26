import os
import pandas as pd

from data_preprocessing import load_and_preprocess
from clustering import evaluate_clusters, train_kmeans
from visualization import (
    plot_elbow,
    plot_silhouette,
    plot_clusters,
    plot_cluster_sizes
)


def main():
    # Xác định đường dẫn dự án
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "CC GENERAL.csv")
    output_dir = os.path.join(base_dir, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Đọc và tiền xử lý dữ liệu
    print("\n=== 1. DATA PREPROCESSING ===")
    df_clean, X_scaled = load_and_preprocess(data_path)

    # 2. Đánh giá số cụm
    print("\n=== 2. EVALUATING CLUSTERS ===")
    results = evaluate_clusters(X_scaled, max_k=10)

    # 3. Lưu chỉ số đánh giá
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(
        os.path.join(output_dir, "cluster_metrics.csv"),
        index=False
    )

    # 4. Vẽ biểu đồ Elbow và Silhouette
    plot_elbow(results, output_dir)
    plot_silhouette(results, output_dir)

    # 5. Huấn luyện mô hình K-Means
    print("\n=== 3. TRAINING K-MEANS ===")
    model, labels = train_kmeans(X_scaled, n_clusters=6)

    # 6. Gắn nhãn cụm cho khách hàng
    df_result = df_clean.copy()
    df_result["Cluster"] = labels

    # 7. Lưu kết quả phân cụm
    df_result.to_csv(
        os.path.join(output_dir, "clustered_customers.csv"),
        index=False
    )

    # 8. Thống kê đặc điểm trung bình mỗi cụm
    summary = df_result.groupby("Cluster").mean(numeric_only=True)
    summary.to_csv(
        os.path.join(output_dir, "cluster_summary.csv")
    )

    # 9. Vẽ biểu đồ kết quả
    plot_clusters(X_scaled, labels, output_dir)
    plot_cluster_sizes(labels, output_dir)

    # 10. In kết quả ra Terminal
    print("\n=== 4. RESULTS ===")
    print("Số khách hàng:", len(df_result))
    print("Số cụm:", df_result["Cluster"].nunique())
    print("\nSố khách hàng trong mỗi cụm:")
    print(df_result["Cluster"].value_counts().sort_index())
    print("\nĐặc điểm trung bình của từng cụm:")
    print(summary)

    print("\nĐã hoàn thành!")
    print("Thư mục kết quả:", output_dir)


if __name__ == "__main__":
    main()
