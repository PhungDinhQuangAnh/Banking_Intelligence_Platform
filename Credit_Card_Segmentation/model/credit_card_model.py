import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import math

def check_data(df):
    np.set_printoptions(suppress=True)
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    print("")
    print("THÔNG TIN CƠ BẢN:")
    print(df.info())
    print("-------------------------------\n")

    print("GIÁ TRỊ THIẾU:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("-------------------------------\n")

    print("TRÙNG LẶP:", df.duplicated().sum(), "dòng")
    print("-------------------------------\n")

    print("SỐ GIÁ TRỊ KHÔNG TRÙNG LẶP:")
    for col in df.columns:
        print('{:33} : {:6} : {:}'.format(col, df[col].nunique(), df[col].dtype))
    print("-------------------------------\n")

    print("KIỂM TRA DỮ LIỆU:")
    for col in df.columns:
        print(f"- {col}:", df[col].unique())
    print("-------------------------------\n")

    print("KIỂM TRA PHÂN PHỐI GIÁ TRỊ CÁC CỘT:")
    for col in df.columns:
        print(f"{col} value_counts():")
        print(df[col].value_counts(dropna=False))
        print("")
    print("-------------------------------\n")

    print("KIỂM TRA KIỂU DỮ LIỆU SAI:")
    for col in df.select_dtypes(include='object').columns:
        try:
            pd.to_numeric(df[col])
            print(f"- {col}: chứa số nhưng lưu dạng object")
        except:
            pass
    print("-------------------------------\n")

    print("THỐNG KÊ MÔ TẢ CÁC CỘT DỮ LIỆU SỐ:")
    pd.set_option('display.max_columns', None)
    print(df[num_cols].describe())
    print("-------------------------------\n")
def visualize_data(df):
    num_cols = [col for col in df.select_dtypes(include="number").columns]

    print("CORRELATION MATRIX PLOT:")
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True)
    plt.show()
    print("-------------------------------\n")

    print("HISTOGRAM:")
    n_cols = 5
    n_rows = math.ceil(len(num_cols) / n_cols)
    df.select_dtypes(include="number").hist(layout=(n_rows, n_cols), figsize=(16, 4 * n_rows), sharex=False)
    plt.tight_layout()
    plt.show()
    print("-------------------------------\n")
def check_outliers(df):
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    print("")
    print("OUTLIERS (IQR method):")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            print(f"- {col}: {len(outliers)} outliers")
    print("-------------------------------\n")

    print("BIỂU ĐỒ BOXPLOT (Phát Hiện Outliers Và Độ Lệch Dữ Liệu):")
    n_cols = 5
    n_rows = math.ceil(len(num_cols) / n_cols)
    df[num_cols].plot(kind='box', subplots=True, layout=(n_rows, n_cols), figsize=(16, 4 * n_rows), sharex=False)
    plt.tight_layout(pad=2.5)
    plt.show()
    print("-------------------------------\n")

# Thiết lập đường dẫn thư mục hiện hành
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Tải dữ liệu & Khám phá dữ liệu (EDA) ---
df = pd.read_csv(os.path.join(BASE_DIR, "..", "dataset", "credit_card_dataset.csv"))
# check_data(df)
# visualize_data(df)
# check_outliers(df)

# --- 2. Tiền xử lý dữ liệu ---
# 2_1. Xử lý giá trị thiếu
low_balance_median = df[df["BALANCE"] < 100]["CREDIT_LIMIT"].median()
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(low_balance_median)

# 2_2. Tách riêng những dữ liệu outliers và dữ liệu số đông (Dựa trên 5 cột cốt lõi)
core_financial_cols = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
]
is_outlier = pd.Series(False, index=df.index)
for col in core_financial_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q3 - 1.5 * IQR
    is_outlier = is_outlier | (df[col] > upper_bound) | (df[col] < lower_bound)

df_normal = df[~is_outlier].copy()
df_outlier = df[is_outlier].copy()

print(f"Số lượng dòng tập Số Đông (Normal): {df_normal.shape[0]}")
print(f"Số lượng dòng tập Ngoại Lai (Outliers): {df_outlier.shape[0]}")

# 2_3. Chỉ lấy đúng 5 cột để chuẩn hóa độc lập nhằm Train mô hình
scaler_n = StandardScaler()
X_normal_for_train = df_normal[core_financial_cols]  
X_normal_scaled = scaler_n.fit_transform(X_normal_for_train)

scaler_o = StandardScaler()
X_outlier_for_train = df_outlier[core_financial_cols] 
X_outlier_scaled = scaler_o.fit_transform(X_outlier_for_train)

# --- 3. Hàm vẽ biểu đồ Elbow tìm số cụm k tối ưu ---
def plot_elbow(X, title, filename):
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(K_range, inertia, "bx-")
    plt.xlabel("Số lượng cụm K")
    plt.ylabel("Inertia (WCSS)")
    plt.title(title)
    plt.savefig(os.path.join(BASE_DIR,"..","report",filename), dpi=300, bbox_inches="tight")
    plt.show()

plot_elbow(X_normal_scaled, "Elbow cho Tập Số Đông (Normal)", "elbow_normal.png")
plot_elbow(X_outlier_scaled, "Elbow cho Tập Ngoại Lai (Outliers)", "elbow_outlier.png")


# --- 4. Huấn luyện K-Means độc lập theo số K đã chọn ---
# Tập Số Đông (Chọn K=3)
kmeans_normal = KMeans(n_clusters=3, random_state=42, n_init=10)
df_normal["Cluster_Raw"] = kmeans_normal.fit_predict(X_normal_scaled)
df_normal["Cluster"] = df_normal["Cluster_Raw"].map(
    {0: "Normal_1", 1: "Normal_2", 2: "Normal_3"}
)

# Tập Outliers (Chọn K=3)
kmeans_outlier = KMeans(n_clusters=3, random_state=42, n_init=10)
df_outlier["Cluster_Raw"] = kmeans_outlier.fit_predict(X_outlier_scaled)
df_outlier["Cluster"] = df_outlier["Cluster_Raw"].map(
    {0: "Outlier_1", 1: "Outlier_2", 2: "Outlier_3"}
)

# --- 5. Gộp 2 tập dữ liệu lại thành một bản đồ tổng thể ---
# Giữ lại toàn bộ các cột hành vi khác bằng cách xóa đúng cột phụ Cluster_Raw
df_final = pd.concat(
    [
        df_normal.drop(columns=["Cluster_Raw"]),
        df_outlier.drop(columns=["Cluster_Raw"]),
    ]
)

print("\nThống kê số lượng khách hàng sau khi Phân tầng:")
print(df_final["Cluster"].value_counts())

# Chuẩn hóa lại và chạy t-SNE
X_final = df_final.drop(columns=["Cluster"])
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

tsne = TSNE(n_components=3, perplexity=40, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_final_scaled)

# Thêm tọa độ t-SNE trực tiếp vào df_final để xuất file
df_final["t-SNE 1"] = X_tsne[:, 0]
df_final["t-SNE 2"] = X_tsne[:, 1]
df_final["t-SNE 3"] = X_tsne[:, 2]

# --- 6. Kiểm tra chân dung các nhóm khách hàng ---
# Tập trung kiểm tra cả các cột hành vi mua sắm xem chân dung có sắc nét không
marketing_cols = [
    "BALANCE",
    "PURCHASES",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "PURCHASES_TRX",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
]
cluster_profile = df_final.groupby("Cluster")[marketing_cols].mean()
print("\n=== BẢNG CHÂN DUNG KHÁCH HÀNG ===")
print(cluster_profile.round(2).T)

# --- 7. Đóng gói & lưu trữ file ---
# Lưu file dữ liệu tổng hợp (Chứa đủ 9 cột gốc + cột Cluster + 3 cột t-SNE)
df_final.to_csv(os.path.join(BASE_DIR,"..","report","credit_card_segmented_tsne.csv"), index=False)

# Lưu các mô hình và bộ scaler để sử dụng dự đoán trên Streamlit 
joblib.dump(scaler_n, os.path.join(BASE_DIR,"scaler_normal.pkl"))
joblib.dump(scaler_o, os.path.join(BASE_DIR,"scaler_outlier.pkl"))
joblib.dump(kmeans_normal, os.path.join(BASE_DIR,"kmeans_normal.pkl"))
joblib.dump(kmeans_outlier, os.path.join(BASE_DIR,"kmeans_outlier.pkl"))

# Tính toán ranh giới IQR để lưu lại phục vụ việc phân loại động trên Streamlit
iqr_bounds = {}
for col in core_financial_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_bounds[col] = {"lower": Q1 - 1.5 * IQR, "upper": Q3 + 1.5 * IQR}
joblib.dump(iqr_bounds, os.path.join(BASE_DIR,"iqr_bounds.pkl"))

