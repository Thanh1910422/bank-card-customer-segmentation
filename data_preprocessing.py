
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def load_and_preprocess(file_path):
    # 1. Đọc dữ liệu
    df = pd.read_csv(file_path)

    print("Kích thước dữ liệu ban đầu:", df.shape)

    # 2. Loại bỏ dòng trùng lặp
    df = df.drop_duplicates().copy()

    # 3. Loại bỏ mã khách hàng
    if "CUST_ID" in df.columns:
        df = df.drop(columns=["CUST_ID"])

    # 4. Chỉ giữ các biến số
    df = df.select_dtypes(include=[np.number]).copy()

    # 5. Xử lý giá trị vô hạn và cột thiếu hoàn toàn
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")

    if df.empty or df.shape[1] == 0:
        raise ValueError("Không còn dữ liệu hợp lệ sau xử lý.")

    # 6. Điền giá trị thiếu bằng trung vị
    imputer = SimpleImputer(strategy="median")
    data_imputed = imputer.fit_transform(df)

    df_clean = pd.DataFrame(
        data_imputed,
        columns=df.columns,
        index=df.index
    )

    # 7. Biến đổi log các biến tiền và số giao dịch
    log_columns = [
        "BALANCE",
        "PURCHASES",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE",
        "PAYMENTS",
        "MINIMUM_PAYMENTS",
        "CREDIT_LIMIT",
        "CASH_ADVANCE_TRX",
        "PURCHASES_TRX"
    ]

    df_model = df_clean.copy()

    for col in log_columns:
        if col in df_model.columns:
            # Chỉ áp dụng log1p nếu toàn bộ giá trị không âm
            if (df_model[col] >= 0).all():
                df_model[col] = np.log1p(df_model[col])

    # 8. Chuẩn hóa bằng RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_model)

    print("Kích thước dữ liệu sau xử lý:", df_clean.shape)
    print("Số biến được biến đổi log:",
          sum(col in df_model.columns for col in log_columns))
    print("Đã tiền xử lý và chuẩn hóa dữ liệu.")

    # df_clean giữ giá trị gốc để diễn giải đặc điểm khách hàng
    # X_scaled là dữ liệu dùng để huấn luyện mô hình
    return df_clean, X_scaled
