<h1 align="center">AI Health</h1>

[![Streamlit App](https://img.shields.io/badge/Truy%20cập%20ứng%20dụng%20trực%20tuyến-Click%20here-brightgreen)](https://ai-health.streamlit.app/)

AI Health là ứng dụng Web tương tác được xây dựng bằng Python & Streamlit nhằm hỗ trợ **dự đoán chi phí y tế** và **nguy cơ mắc bệnh** dựa trên thông tin cá nhân bằng các mô hình **Machine Learning**.

---

## Chức năng chính

- Ước tính **chi phí y tế hằng năm** mà bảo hiểm chi trả (Hoa Kỳ) – *Regression*
- Dự đoán **khả năng mắc bệnh tiểu đường** – *Classification*
- Dự đoán **nguy cơ mắc bệnh tim mạch** – *Classification*
- Trực quan hiệu suất mô hình bằng **biểu đồ hiệu suất mô hình**, **bảng chỉ số đánh giá** (Accuracy, Precision, Recall, F1, R2,...)
- Trực quan hóa dự đoán bằng giao diện người dùng đơn giản, dễ sử dụng

---

## Giao diện demo

<p align="center">
  <img src="https://github.com/PhungDinhQuangAnh/ai-health/blob/main/Demo/demo1.png" alt="Giao diện demo">
  <img src="https://github.com/PhungDinhQuangAnh/ai-health/blob/main/Demo/demo2.png" alt="Giao diện demo">
</p>

---

## Mô hình & Dữ liệu

| Bài toán             | Dataset                                                                                          | Mô hình sử dụng   |
|---------------------|--------------------------------------------------------------------------------------------------|-------------------|
| Dự đoán chi phí y tế| [Insurance Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)                | RandomForestRegressor |
| Tiểu đường           | [Diabetes Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | XGBClassifier |
| Bệnh tim             | [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)| RandomForestClassifier |

---

## Cấu trúc dự án
<pre>  
AI_Health/
├── app.py                      # App chính - giao diện chọn mô hình
├── requirements.txt            # Danh sách thư viện cần cài
├── README.md                   # Tài liệu mô tả dự án
├── LICENSE                     # Giấy phép sử dụng
├── Demo/                     
|    ├── demo1.png              # Hình ảnh demo giao diện web
│    └── demo2.png
│
├── Medical_Cost/
│   ├── medical_cost_app.py     # Ứng dụng Streamlit cho dự đoán chi phí y tế
│   ├── dataset/
│   │   └── medical_cost_dataset.csv   # Dữ liệu gốc
│   ├── model/
│   │   ├── medical_cost_model.py      # Code huấn luyện mô hình
│   │   └── medical_cost_model.pkl     # Mô hình đã lưu
│   └── report/
│       ├── actual_vs_predicted.png    # Biểu đồ giá trị dự đoán vs thực tế
│       ├── error_distribution.png     # Biểu đồ phân bố sai số
│       └── medical_cost_metrics.json  # Chỉ số đánh giá mô hình
│
├── Diabetes/
│   ├── diabetes_app.py         # Ứng dụng Streamlit cho dự đoán tiểu đường
│   ├── dataset/
│   │   └── diabetes_dataset.csv
│   ├── model/
│   │   ├── diabetes_model.py
│   │   └── diabetes_model.pkl
│   └── report/
│       ├── diabetes_classification_report.csv   # Chỉ số đánh giá
│       └── diabetes_confusion_matrix.jpg        # Ma trận nhầm lẫn
│
├── Heart/
│   ├── heart_app.py            # Ứng dụng Streamlit cho dự đoán bệnh tim
│   ├── dataset/
│   │   └── heart_dataset.csv
│   ├── model/
│   │   ├── heart_model.py
│   │   └── heart_model.pkl
│   └── report/
│       ├── heart_classification_report.csv
│       └── heart_confusion_matrix.jpg
</pre>

---

## Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Web UI:** streamlit
- **Xử lý dữ liệu:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Trực quan hóa:** matplotlib, seaborn, plotly

---

## Cách chạy ứng dụng

### Truy cập ứng dụng online

🔗 https://ai-health.streamlit.app/
