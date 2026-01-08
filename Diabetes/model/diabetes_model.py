import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

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
    df[num_cols].plot(kind='box', subplots=True, layout=(5, 5), sharex=False)
    plt.show()
    print("-------------------------------\n")

# --- Lấy đường dẫn tuyệt đối của thư mục hiện tại ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Tải & Khám phá Dataset ---
df = pd.read_csv(os.path.join(BASE_DIR, "..", "dataset", "diabetes_dataset.csv"))
target = "Diabetes_binary"
check_data(df,target)

# --- 2. Chia x, y ---
x = df.drop(target,axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# --- 3. Xây dựng mô hình ---
model = XGBClassifier(colsample_bytree= 0.8, gamma= 0, learning_rate= 0.05, max_depth= 5, min_child_weight= 3, n_estimators= 200, subsample= 0.8)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

            # parameters = {
            #     'n_estimators': [100, 200],
            #     'learning_rate': [0.01, 0.05, 0.1],
            #     'max_depth': [3, 5, 7],
            #     'min_child_weight': [1, 3, 5],
            #     'subsample': [0.8, 1.0],
            #     'colsample_bytree': [0.8, 1.0],
            #     'gamma': [0, 0.2, 0.4],
            # }
            # cls = GridSearchCV(XGBClassifier(verbose=-1,objective='binary'), param_grid=parameters, scoring="f1", cv=3, verbose=2, n_jobs=9)
            # cls.fit(x_train,y_train)
            # print("Best F1-score (CV): ",cls.best_score_)
            # print("Best parameters: ",cls.best_params_)
            # y_pred = cls.predict(x_test)

# --- 4. Đánh giá mô hình ---
# 4.1. Evaluation Metrics
print(classification_report(y_test, y_pred))
    # Save Classification_Report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).T.round(2)
report_df.to_csv(os.path.join(BASE_DIR, "..", "report", "diabetes_classification_report.csv"))

# 4.2. Confusion Matrix Plot
cm = confusion_matrix(y_test,y_pred,labels=[0,1])
cm_to_df = pd.DataFrame(cm, index=["Not Diabetes", "Diabetes"], columns=["Not Diabetes", "Diabetes"])
sn.heatmap(cm_to_df, annot=True, fmt="g")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, "..", "report", "diabetes_confusion_matrix.jpg"))
plt.show()

# 4.3. Phân tích feature quan trọng
feature_imp = model.feature_importances_
feature_imp_names = x.columns
feature_imp_df = pd.Series(feature_imp, index=feature_imp_names).sort_values(ascending=False)
print(feature_imp_df)

# 4.4. Đánh giá tính ổn định mô hình bằng Cross-Validation
scores = cross_val_score(model, x, y, cv=5)
print("Cross-val accuracy each fold:", scores)
print("Mean:", scores.mean(), "Std:", scores.std())
print("")

# 4.5. So sánh độ chính xác giữa tập train và test --> Kiểm tra Overfitting
print("Train Accuracy:", model.score(x_train, y_train))
print("Test Accuracy:", model.score(x_test, y_test))

# Lưu mô hình
joblib.dump(model, "diabetes_model.pkl")
