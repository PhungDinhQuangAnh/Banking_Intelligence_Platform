def run_diabetes_app():
    import streamlit as st
    import joblib
    import pandas as pd
    import plotly.graph_objects as go
    from datetime import datetime
    import os

    # Đường dẫn an toàn
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(APP_DIR, "model", "diabetes_model.pkl")
    CLASSIFICATION_REPORT_PATH = os.path.join(APP_DIR, "report", "diabetes_classification_report.csv")
    CONFUSION_MATRIX_PATH = os.path.join(APP_DIR, "report", "diabetes_confusion_matrix.jpg")

    # --- 1. Giao diện trang ---
    # st.set_page_config(page_title="Type 2 Diabetes Predictor", layout="wide", page_icon="🩺")
    st.markdown(
        "<h1 style='text-align:center; color:#2E86C1;'>🩺 Dự đoán nguy cơ bệnh tiểu đường loại 2</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- 2. Load mô hình ---
    model = joblib.load(MODEL_PATH)

    # --- 3. Form nhập liệu người dùng ---
    st.subheader("📋 Nhập thông tin sức khỏe:")
    with st.form("input_form"):
        with st.expander("‍🧍**Thông tin cá nhân**"):
            col1, col2 = st.columns(2)
            with col1:
                Sex = st.radio("Giới tính (Sex)", ["Nam", "Nữ"], help="Giới tính sinh học của bạn", horizontal=True)
                Education = st.selectbox("Trình độ học vấn", ["Không đi học hoặc chỉ học mẫu giáo","Tốt nghiệp lớp 1–8","Tốt nghiệp lớp 9–11","Tốt nghiệp lớp 12 hoặc có chứng chỉ tương đương THPT (GED)","Đã học đại học hoặc trường nghề, nhưng chưa có bằng","Tốt nghiệp đại học"])
            with col2:
                Age = st.selectbox("Tuổi", ["18–24 tuổi","25–29 tuổi","30–34 tuổi","35–39 tuổi","40–44 tuổi","45–49 tuổi","50–54 tuổi","55–59 tuổi","60–64 tuổi","65–69 tuổi","70–74 tuổi","75–79 tuổi","Trên 80 tuổi"])
                Income = st.selectbox("Thu nhập", ["Dưới 4 triệu VNĐ/tháng","4–6 triệu VNĐ/tháng","6–8 triệu VNĐ/tháng","8–10 triệu VNĐ/tháng","10–13 triệu VNĐ/tháng","13–18 triệu VNĐ/tháng","18–25 triệu VNĐ/tháng","Trên 25 triệu VNĐ/tháng"])

        with st.expander("‍ ❤️ **Tiền sử bệnh & sức khỏe tổng quát**"):
            col3, col4 = st.columns(2)
            with col3:
                HighBP = st.radio("Từng được chẩn đoán bị huyết áp cao?", ["Không", "Có"], horizontal=True)
                HighChol = st.radio("Từng được chẩn đoán mức Cholesterol trong máu cao?", ["Không", "Có"], horizontal=True)
                HeartDiseaseorAttack = st.radio("Từng bị bệnh tim?", ["Không", "Có"], horizontal=True)
                Stroke = st.radio("Từng bị đột quỵ?", ["Không", "Có"], horizontal=True)
            with col4:
                BMI = st.number_input("Chỉ số khối cơ thể (BMI)", min_value=10, max_value=100, value=25, step=1, help="BMI = Cân nặng (kg) / (Chiều cao (m))²")
                GenHlth = st.selectbox("Đánh giá tổng quát về sức khỏe", ["Rất tốt (Excellent)", "Tốt (Very good)", "Khá (Good)", "Kém (Fair)", "Rất kém (Poor)"])
                PhysHlth = st.number_input("Sức khỏe thể chất", min_value=0, max_value=30, value=0, step=1, help="Số ngày trong 30 ngày qua mà bạn cảm thấy thể chất không khỏe, như mệt mỏi, đau nhức, bệnh tật...")
                MentHlth = st.number_input("Sức khỏe tinh thần", min_value=0, max_value=30, value=0, step=1, help="Số ngày trong 30 ngày qua mà bạn cảm thấy tâm lý không ổn, lo lắng, căng thẳng, trầm cảm...")

        with st.expander("‍🏃‍♀️ **Hành vi sức khỏe**"):
            col5, col6 = st.columns(2)
            with col5:
                PhysActivity = st.radio("Hoạt động thể chất?", ["Không", "Có"], horizontal=True, help="Bạn có **tập thể dục/thể thao hoặc hoạt động thể chất ngoài công việc hằng ngày** trong vòng **30 ngày qua**.")
                Smoker = st.radio("Có hút thuốc?", ["Không","Có (hiện tại hoặc đã từng)"], horizontal=True)
                HvyAlcoholConsump = st.radio("Uống rượu nhiều?", ["Không", "Có"], horizontal=True,
                                             help=(
                                                "**Tiêu chí xác định**:\n"
                                                    "- Nam: ≥ 14 đơn vị/tuần\n"
                                                    "- Nữ: ≥ 7 đơn vị/tuần\n"
                                                "\n**Một 'đơn vị' rượu là**:\n"
                                                    "- 1 cốc bia (~355ml)\n"
                                                    "- hoặc 1 ly rượu vang (~150ml)\n"                                                
                                                    "- hoặc 1 shot rượu mạnh (~44ml)\n")
                                             )
            with col6:
                Fruits = st.radio("Ăn trái cây hàng ngày?", ["Không", "Có"], horizontal=True)
                Veggies = st.radio("Ăn rau, củ hàng ngày?", ["Không", "Có"], horizontal=True)
                DiffWalk = st.radio("Khó khăn khi đi lại?", ["Không", "Có"], horizontal=True, help="**Gặp khó khăn hoặc không thể đi lại** do vấn đề về thể chất, sức khỏe hoặc bệnh tật.")

        with st.expander("‍🏥 **Tiếp cận dịch vụ y tế**"):
            col7, col8, col9 = st.columns(3)
            with col7:
                CholCheck = st.radio("Đã kiểm tra cholesterol trong 5 năm qua?", ["Không", "Có"])
            with col8:
                NoDocbcCost = st.radio("Từng không khám vì chi phí?", ["Không", "Có"], help="Trong **12 tháng qua**, bạn **cần gặp bác sĩ nhưng đã không đi vì lý do chi phí quá cao**.")
            with col9:
                AnyHealthcare = st.radio("Có bảo hiểm y tế hoặc nguồn thanh toán chăm sóc sức khỏe?", ["Không", "Có"])

        submitted = st.form_submit_button("🔍 **Dự đoán**")

    # --- 4. Dự đoán ---
    if submitted:
        # 4_1. Mapping từng nhóm
        mappings = {
            "Sex": {"Nữ": 0, "Nam": 1},
            "Education": {
                "Không đi học hoặc chỉ học mẫu giáo": 1,
                "Tốt nghiệp lớp 1–8": 2,
                "Tốt nghiệp lớp 9–11": 3,
                "Tốt nghiệp lớp 12 hoặc có chứng chỉ tương đương THPT (GED)": 4,
                "Đã học đại học hoặc trường nghề, nhưng chưa có bằng": 5,
                "Tốt nghiệp đại học": 6
            },
            "Age": {
                "18–24 tuổi": 1, "25–29 tuổi": 2, "30–34 tuổi": 3, "35–39 tuổi": 4,
                "40–44 tuổi": 5, "45–49 tuổi": 6, "50–54 tuổi": 7, "55–59 tuổi": 8,
                "60–64 tuổi": 9, "65–69 tuổi": 10, "70–74 tuổi": 11,
                "75–79 tuổi": 12, "Trên 80 tuổi": 13
            },
            "Income": {
                "Dưới 4 triệu VNĐ/tháng": 1, "4–6 triệu VNĐ/tháng": 2, "6–8 triệu VNĐ/tháng": 3,
                "8–10 triệu VNĐ/tháng": 4, "10–13 triệu VNĐ/tháng": 5, "13–18 triệu VNĐ/tháng": 6,
                "18–25 triệu VNĐ/tháng": 7, "Trên 25 triệu VNĐ/tháng": 8
            },
            "GenHlth": {"Rất tốt (Excellent)": 1, "Tốt (Very good)": 2, "Khá (Good)": 3,
                        "Kém (Fair)": 4, "Rất kém (Poor)": 5},
            # Áp dụng cho nhóm Có/Không
            "binary": {"Không": 0, "Có": 1, "Có (hiện tại hoặc đã từng)": 1}
        }

        # 4_2. Tạo dictionary cho dữ liệu đầu vào
        input_data = {
            "HighBP": mappings["binary"][HighBP],
            "HighChol": mappings["binary"][HighChol],
            "CholCheck": mappings["binary"][CholCheck],
            "BMI": BMI,
            "Smoker": mappings["binary"][Smoker],
            "Stroke": mappings["binary"][Stroke],
            "HeartDiseaseorAttack": mappings["binary"][HeartDiseaseorAttack],
            "PhysActivity": mappings["binary"][PhysActivity],
            "Fruits": mappings["binary"][Fruits],
            "Veggies": mappings["binary"][Veggies],
            "HvyAlcoholConsump": mappings["binary"][HvyAlcoholConsump],
            "AnyHealthcare": mappings["binary"][AnyHealthcare],
            "NoDocbcCost": mappings["binary"][NoDocbcCost],
            "GenHlth": mappings["GenHlth"][GenHlth],
            "MentHlth": MentHlth,
            "PhysHlth": PhysHlth,
            "DiffWalk": mappings["binary"][DiffWalk],
            "Sex": mappings["Sex"][Sex],
            "Age": mappings["Age"][Age],
            "Education": mappings["Education"][Education],
            "Income": mappings["Income"][Income]
        }
        input_df = pd.DataFrame([input_data])

        # 4_3. Dự đoán xác suất
        proba = model.predict_proba(input_df)[0][1]

        # --- 5. Gauge hiển thị phần trăm nguy cơ ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Kết quả dự đoán:")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 2),
            title={'text': "Nguy cơ mắc bệnh tiểu đường (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'bar': {'color': "crimson"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. Hiển thị kết quả ---
        if proba < 0.4:
            result = "✅ Nguy cơ THẤP mắc bệnh tiểu đường"
            st.success(f"{result} ({proba * 100:.2f}%)")
        elif proba < 0.7:
            result = "⚠️ Nguy cơ TRUNG BÌNH mắc bệnh tiểu đường"
            st.warning(f"{result} ({proba * 100:.2f}%)")
        else:
            result = "❗ Nguy cơ CAO mắc bệnh tiểu đường"
            st.error(f"{result} ({proba * 100:.2f}%)")

        # --- 7. Lịch sử dự đoán ---
        # 7_1. Khởi tạo session_state nếu chưa có
        if "diabetes_history" not in st.session_state:
            st.session_state["diabetes_history"] = []

        # 7_2. Nếu form đã submit, thêm dòng mới vào lịch sử
        record = {
            "🕒 Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "📈 Xác suất (%)": round(proba * 100, 2),
            "👤 Giới tính": Sex,
            "👵 Tuổi": Age,
            "🎓 Trình độ học vấn": Education,
            "💰 Thu nhập": Income,
            "💓 Huyết áp cao": HighBP,
            "🧬 Mỡ máu cao": HighChol,
            "❤️ Bệnh tim": HeartDiseaseorAttack,
            "🧠 Đột quỵ": Stroke,
            "⚖️ BMI": BMI,
            "🩺 Sức khỏe tổng quát": GenHlth,
            "🤒 Ngày thể chất không khỏe": PhysHlth,
            "😟 Ngày tâm lý không ổn": MentHlth,
            "🏃 Hoạt động thể chất": PhysActivity,
            "🚬 Hút thuốc": Smoker,
            "🍺 Uống rượu nhiều": HvyAlcoholConsump,
            "🍎 Ăn trái cây": Fruits,
            "🥦 Ăn rau củ": Veggies,
            "🚶‍♂️ Khó khăn đi lại": DiffWalk,
            "🧪 Đã kiểm tra Cholesterol": CholCheck,
            "💸 Không đi khám vì chi phí": NoDocbcCost,
            "🏥 Có bảo hiểm y tế": AnyHealthcare
        }
        st.session_state["diabetes_history"].append(record)

        # 7_3. Hiển thị lịch sử dự đoán (gồm cả bản mới nhất)
        if st.session_state["diabetes_history"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📜 Lịch sử dự đoán")
            df_history = pd.DataFrame(st.session_state["diabetes_history"])

            # Tô đậm dòng cuối (mới nhất)
            def highlight_last(s):
                return ['background-color: #e0f7fa' if i == len(s) - 1 else '' for i in range(len(s))]

            st.dataframe(
                df_history.style.apply(highlight_last, axis=0),
                use_container_width=True
            )

        # --- 8. Đánh giá hiệu suất mô hình ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🚀 Hiệu suất mô hình")

        # 8_1. Confusion Matrix
        with st.expander("📊 **Confusion Matrix**"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(CONFUSION_MATRIX_PATH)
            with col2:
                st.markdown("""
                    <h5>📋 Diễn giải kết quả:</h5>
                    <ul>
                      <li>✅ Dự đoán đúng người KHÔNG mắc bệnh: <b>5.049 người</b>
                      <li>🧍 Dự đoán đúng người CÓ bệnh: <b>5.659 người</b>
                      <li>⚠️ Dự đoán nhầm người khỏe là có bệnh: <b>2.041 người</b>
                      <li>❌ Bỏ sót người bệnh (dự đoán là khỏe): <b>1.390 người</b>
                    </ul>
    
                    <h5>🧠 Kết luận:</h5>
                    <ul>
                        <li>📈 Tổng số dự đoán đúng: <b>10.708 / 14.139 → ~76% chính xác</b>
                        <li>🔍 Mô hình <b>phát hiện khá tốt người bệnh</b>, giúp cảnh báo sớm nguy cơ.
                        <li>👩‍⚕️ Tuy nhiên, vẫn có sai sót nên <b>người dùng nên đi khám để xác nhận</b> nếu kết quả là <b>“nguy cơ cao”</b>.
                    </ul>
                    """, unsafe_allow_html=True)

        # 8_2. Classification Report
        with st.expander("📋 **Classification Report**"):
            # Đọc dữ liệu từ file CSV
            report_df = pd.read_csv(CLASSIFICATION_REPORT_PATH, index_col=0)
            report_df.rename(index={
                "0.0": "Không mắc bệnh (Class 0)",
                "1.0": "Có bệnh (Class 1)",
                "accuracy": "Độ chính xác (Accuracy)",
                "macro avg": "Trung bình cộng (Macro Avg)",
                "weighted avg": "Trung bình có trọng số (Weighted Avg)"
            }, inplace=True)

            # Định dạng bảng đẹp
            styled_df = report_df.style.format("{:.2f}").set_properties(**{
                'text-align': 'center'
            }).set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]}
            ])

            # Hiển thị bảng
            st.dataframe(styled_df, use_container_width=True)

            # Phân tích dễ hiểu
            st.markdown("""
                <h5>📊 Phân tích chi tiết:</h5>
                <ul>
                    <li>👤 <b>Không mắc bệnh (Class 0):</b>
                    <ul>
                        <li>📏 Precision: {:.0f}% → Trong số dự đoán là <i>không mắc bệnh</i>, có {:.0f}% là đúng.
                        <li>🎯 Recall: {:.0f}% → Trong số <i>thực sự không mắc bệnh</i>, mô hình phát hiện đúng {:.0f}%.
                    </ul>
                    <li>❤️ <b>Có bệnh (Class 1):</b>
                    <ul>
                        <li>📏 Precision: {:.0f}% → Trong số dự đoán là <i>có bệnh</i>, có {:.0f}% là đúng.
                        <li>🎯 Recall: {:.0f}% → Trong số <i>thực sự mắc bệnh</i>, mô hình phát hiện đúng {:.0f}%.
                    </ul>
                </ul>
                <h5>📈 Tổng thể:</h5>
                <ul>
                    <li>✅ Accuracy (độ chính xác tổng thể): {:.0f}%
                    <li>⚖️ Mô hình có độ cân bằng tốt (F1-score trung bình ≈ {:.0f}%)
                    <li>🔍 Ưu tiên phát hiện đúng người có bệnh hơn → phù hợp cho mục tiêu sàng lọc nguy cơ.
                </ul>
                """.format(
                report_df.loc["Không mắc bệnh (Class 0)", "precision"] * 100,
                report_df.loc["Không mắc bệnh (Class 0)", "precision"] * 100,
                report_df.loc["Không mắc bệnh (Class 0)", "recall"] * 100,
                report_df.loc["Không mắc bệnh (Class 0)", "recall"] * 100,

                report_df.loc["Có bệnh (Class 1)", "precision"] * 100,
                report_df.loc["Có bệnh (Class 1)", "precision"] * 100,
                report_df.loc["Có bệnh (Class 1)", "recall"] * 100,
                report_df.loc["Có bệnh (Class 1)", "recall"] * 100,

                report_df.loc["Độ chính xác (Accuracy)", "precision"] * 100,
                report_df.loc["Trung bình cộng (Macro Avg)", "f1-score"] * 100
            ), unsafe_allow_html=True)

        # --- 9. Thông tin thêm ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📘 Thông tin thêm")
        with st.expander("📖 **Xem chi tiết**"):
            st.markdown("""
            <div style='font-size: 16px; line-height: 1.7; color: #333;'>
    
            <h5>🧠 Về ứng dụng này</h5> 
            <p>Ứng dụng giúp bạn <b>ước lượng nguy cơ mắc bệnh tiểu đường loại 2</b> dựa trên thông tin sức khỏe cá nhân bạn cung cấp.</p>
            <p>Kết quả là <b>một tỉ lệ phần trăm (%)</b> — càng cao thì nguy cơ mắc bệnh càng lớn.</p>
    
            <div style="background-color: #fff8e1; padding: 10px; border-left: 5px solid #f39c12; margin-top:10px;">
            ⚠️ <b>Lưu ý:</b> Ứng dụng chỉ mang tính chất tham khảo, không thay thế cho việc khám hoặc chẩn đoán bởi bác sĩ.
            </div>
    
            <hr>
    
            <h5>📚 Bộ dữ liệu</h5>
            <ul>
                <li>🌐 Bộ dữ liệu mô hình sử dụng trên nền tảng Kaggle: 
                    <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset" target="_blank"><b>Xem tại đây</b></a>.</li>
                <li>📊 Dữ liệu sử dụng là từ file <b>diabetes_binary_5050split_health_indicators_BRFSS2015.csv</b> trong bộ dữ liệu này.</li>
                <li>🔍 Dữ liệu được thu thập từ khảo sát <b>BRFSS 2015</b> – chương trình giám sát các yếu tố rủi ro hành vi qua điện thoại do <b>Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh Hoa Kỳ (CDC)</b> thực hiện.</li>
                <div style="background-color: #f0f9ff; padding: 12px; border-left: 5px solid #3498db; border-radius: 5px; margin-top: 10px;">
                💱 <b>Chú thích:</b> Biến dữ liệu <b>thu nhập</b> đã được điều chỉnh theo <b>sức mua tương đương (Purchasing Power Parity - PPP)</b> trên form nhập liệu trên để phản ánh mức sống tại Việt Nam. Việc chuẩn hóa này giúp mô hình dự đoán chính xác hơn trong bối cảnh kinh tế – xã hội địa phương.
                </div>
            </ul>
    
            <hr>
    
            <h5>⚙️ Ứng dụng hoạt động thế nào?</h5>
            <ol>
                <li>Bạn nhập các thông tin về tuổi, chỉ số cơ thể, hành vi sức khỏe,...</li>
                <li>Mô hình <b>XGBClassifier</b> được huấn luyện với <b>bộ dữ liệu</b> sẽ phân tích và dự đoán nguy cơ.</li>
                <li>Hiển thị kết quả bằng màu sắc và phần trăm dễ hiểu.</li>
            </ol>
    
            <hr>
    
            <h5>🎯 Mục tiêu</h5>
            <p>Giúp bạn <b>nhận biết nguy cơ sớm hơn</b>, từ đó:</p>
            <ul>
                <li>🧘‍♂️ Chủ động thay đổi lối sống</li>
                <li>🏥 Tăng cường khám sức khỏe định kỳ</li>
                <li>🛡️ Góp phần phòng ngừa bệnh tiểu đường một cách hiệu quả</li>
            </ul>
    
            </div>
            """, unsafe_allow_html=True)



