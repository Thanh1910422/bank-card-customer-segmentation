import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def load_and_preprocess(file_path):
    """Đọc, làm sạch và chuẩn hóa dữ liệu khách hàng."""

    # 1. Đọc dữ liệu
    df = pd.read_csv(file_path)
    print("Kích thước dữ liệu ban đầu:", df.shape)

    # 2. Loại bỏ các dòng trùng lặp
    df = df.drop_duplicates().copy()

    # 3. Loại bỏ mã khách hàng vì không dùng để phân cụm
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

    # 7. Biến đổi log1p cho các biến có giá trị không âm
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
    transformed_columns = []

    for col in log_columns:
        if col in df_model.columns and (df_model[col] >= 0).all():
            df_model[col] = np.log1p(df_model[col])
            transformed_columns.append(col)

    # 8. Chuẩn hóa dữ liệu
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_model)

    print("Kích thước dữ liệu sau xử lý:", df_clean.shape)
    print("Số biến được biến đổi log:", len(transformed_columns))
    print("Đã tiền xử lý và chuẩn hóa dữ liệu.")

    return df_clean, X_scaled
