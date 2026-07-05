import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, make_scorer, f1_score
from xgboost import XGBClassifier
import joblib
    
def check_data(df, target):
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
        if col != target:
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

    print("KIỂM TRA ĐỘ MẤT CÂN BẰNG DỮ LIỆU:")
    class_counts = df[target].value_counts()
    print(class_counts)
    print("TỈ LỆ PHẦN TRĂM (%):")
    print(round(df[target].value_counts(normalize=True) * 100, 2))
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

# --- Lấy đường dẫn tuyệt đối của thư mục hiện tại ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Tải & Khám phá dữ liệu (EDA) ---
df = pd.read_csv(os.path.join(BASE_DIR, "..", "dataset", "loan_default_dataset.csv"))
target = "Default"
# check_data(df,target)
# visualize_data(df)
# check_outliers(df)

# --- 2. Tiền xử lý dữ liệu ---
# 2_1. Chia x, y
x = df.drop(target,axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=89, stratify=y)

# 2_2. Biến đổi dữ liệu
num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
ord_cols, ord_categories = ['Education'], [['High School', "Bachelor's", "Master's", 'PhD']]
nom_cols = ['EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

num_scaler = StandardScaler()
ord_encoder = OrdinalEncoder(categories=ord_categories)
nom_encoder = OneHotEncoder(sparse_output=False, drop='first')

preprocessor = ColumnTransformer(transformers=[
    ('num', num_scaler, num_cols),
    ('ord', ord_encoder, ord_cols),
    ('nom', nom_encoder, nom_cols)
])

# --- 3. Xây dựng mô hình ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(scale_pos_weight=2.1,max_depth=5,learning_rate=0.1,min_child_weight=120,reg_lambda=50.0, random_state=89))
])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# --- 4. Đánh giá mô hình theo ngưỡng threshold mặc định 0.5 ---
# 4_1. Evaluation Metrics
print(classification_report(y_test, y_pred))
# Save Classification_Report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).T.round(2)
report_df.to_csv(os.path.join(BASE_DIR, "..", "report", "loan_default_classification_report.csv"))

# 4_2. Confusion Matrix Plot
cm = confusion_matrix(y_test,y_pred,labels=[0,1])
cm_to_df = pd.DataFrame(cm, index=["Dự đoán Không vỡ nợ", "Dự đoán Vỡ nợ"], columns=["Thực tế Không vỡ nợ", "Thực tế Vỡ nợ"])
sns.heatmap(cm_to_df, annot=True, fmt="g")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, "..", "report", "loan_default_confusion_matrix.jpg"))
plt.show()

# --- 5. Đánh giá mô hình theo Chiến lược 3 Vùng Quyết định ---
# Dự đoán XÁC SUẤT (Probability) thay vì nhãn nhị phân
y_proba = model.predict_proba(x_test)[:, 1]

# Thiết lập cấu hình 2 Ngưỡng phân tách (Threshold Tuning)
T_LOW = 0.17   # Dưới mức này: Duyệt tự động (Ép Precision Class 0 lên cao)
T_HIGH = 0.75  # Trên mức này: Từ chối tự động (Ép Precision Class 1 lên cao)

# Hàm phân vùng quyết định tín dụng
def evaluate_three_zones(y_true, y_proba, t_low, t_high):
    # Tạo DataFrame để dễ truy vấn và thống kê
    results_df = pd.DataFrame({
        'Real_Label': y_true.values,
        'Proba_Default': y_proba
    })
    
    # Định nghĩa điều kiện lọc cho từng vùng
    vung_an_toan = results_df[results_df['Proba_Default'] < t_low]
    vung_tham_dinh = results_df[(results_df['Proba_Default'] >= t_low) & (results_df['Proba_Default'] <= t_high)]
    vung_tu_choi = results_df[results_df['Proba_Default'] > t_high]
    
    # --- Tính toán chỉ số cho VÙNG AN TOÀN ---
    total_safe = len(vung_an_toan)
    if total_safe > 0:
        # Precision Class 0 = Số người thực sự KHÔNG vỡ nợ / Tổng số người rơi vào vùng an toàn
        precision_c0_safe = (vung_an_toan['Real_Label'] == 0).sum() / total_safe
    else:
        precision_c0_safe = 0
        
    # --- Tính toán chỉ số cho VÙNG TỪ CHỐI (Auto-Reject) ---
    total_reject = len(vung_tu_choi)
    if total_reject > 0:
        # Precision Class 1 = Số người thực sự VỠ NỢ / Tổng số người bị từ chối thẳng
        precision_c1_reject = (vung_tu_choi['Real_Label'] == 1).sum() / total_reject
    else:
        precision_c1_reject = 0

    # --- In báo cáo chi tiết ---

    print(f"1. VÙNG AN TOÀN (Xác suất < {t_low}):")
    print(f"   - Số lượng hồ sơ giải ngân tự động: {total_safe} ({total_safe/len(results_df)*100:.2f}%)")
    print(f"   - ĐỘ CHÍNH XÁC: {precision_c0_safe*100:.2f}% (Mục tiêu: >= 95%)\n")
    
    print(f"2. VÙNG TỪ CHỐI (Xác suất > {t_high}):")
    print(f"   - Số lượng hồ sơ từ chối tự động : {total_reject} ({total_reject/len(results_df)*100:.2f}%)")
    print(f"   - ĐỘ CHÍNH XÁC: {precision_c1_reject*100:.2f}% (Mục tiêu: >= 70%)\n")
    
    print(f"3. VÙNG THẨM ĐỊNH LẠI (Từ {t_low} đến {t_high}):")
    print(f"   - Số lượng hồ sơ đẩy về duyệt tay : {len(vung_tham_dinh)} ({len(vung_tham_dinh)/len(results_df)*100:.2f}%)")
    print(f"   - Số ca nợ xấu thực tế cần lọc   : {(vung_tham_dinh['Real_Label'] == 1).sum()} ca")
    print("\n=====================================================================")

# Chạy hàm đánh giá hiệu suất thực tế trên tập Test
evaluate_three_zones(y_test, y_proba, t_low=T_LOW, t_high=T_HIGH)

# Lưu mô hình
joblib.dump(model, os.path.join(BASE_DIR, "loan_default_model.pkl"))