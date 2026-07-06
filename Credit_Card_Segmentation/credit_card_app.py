import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

def run_credit_card_app():
    # LẤY CÁC ĐƯỜNG DẪN
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    IQR_BOUNDS_PATH = os.path.join(APP_DIR, "model", "iqr_bounds.pkl")
    KMEANS_NORMAL_PATH = os.path.join(APP_DIR, "model", "kmeans_normal.pkl")
    KMEANS_OUTLIER_PATH = os.path.join(APP_DIR, "model", "kmeans_outlier.pkl")
    SCALER_NORMAL_PATH = os.path.join(APP_DIR, "model", "scaler_normal.pkl")
    SCALER_OUTLIER_PATH = os.path.join(APP_DIR, "model", "scaler_outlier.pkl")
    CSV_PATH = os.path.join(APP_DIR, "report", "credit_card_segmented_tsne.csv")
    ELBOW_NORMAL_PATH = os.path.join(APP_DIR, "report", "elbow_normal.png")
    ELBOW_OUTLIER_PATH = os.path.join(APP_DIR, "report", "elbow_outlier.png")

    # CẤU HÌNH CSS
    st.markdown("""
        <style>
        /* Container bọc ngoài */
        .data-source-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-bottom: 32px;
        }
        /* Thẻ con - Thiết kế đối xứng viền 2 cạnh Trái & Phải */
        .premium-info-card {
            display: flex;
            align-items: center;
            background: #FFFFFF !important;
            border-top: 1px solid #E2E8F0 !important;
            border-bottom: 1px solid #E2E8F0 !important;
            /* Tạo viền màu nổi bật đối xứng ở cả 2 cạnh bên trái và phải */
            border-left: 5px solid #2563EB !important; 
            border-right: 5px solid #2563EB !important; 
            border-radius: 16px !important;
            padding: 18px 24px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04), 0 8px 16px -6px rgba(15, 23, 42, 0.04);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        /* Hiệu ứng Hover */
        .premium-info-card:hover {
            transform: translateY(-5px);
            /* Khi di chuột qua, viền 2 bên sáng rực lên tông xanh đậm hơn */
            border-left-color: #1D4ED8 !important;
            border-right-color: #1D4ED8 !important;
            border-top-color: #BFDBFE !important;
            border-bottom-color: #BFDBFE !important;
            box-shadow: 0 25px 35px -5px rgba(37, 99, 235, 0.15), 0 12px 20px -8px rgba(37, 99, 235, 0.12);
        }
        /* Khối chứa Material Icon bo tròn */
        .card-icon-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 44px;
            height: 44px;
            border-radius: 12px;
            margin-right: 18px;
        }
        /* Phân loại màu nền nhẹ cho từng Icon để tiệp màu web */
        .badge-blue { background: rgba(37, 99, 235, 0.08); color: #2563EB; }
        .badge-shield { background: rgba(2, 132, 199, 0.08); color: #0284C7; }
        /* Ép font chữ Material Icons hoạt động chuẩn xác */
        .card-icon-badge .material-icons {
            font-family: 'Material Icons' !important;
            font-size: 22px !important;
        }
        /* Vùng văn bản nội dung & Điểm nhấn màu sắc */
        .card-content-text {
            font-size: 13.5px !important;
            line-height: 1.6;
            color: #334155 !important;
        }
        .text-highlight {
            color: #2563EB !important;
            font-weight: 700;
        }
        /* Đường dẫn Kaggle */
        .premium-banner-link {
            color: #1D4ED8 !important;
            text-decoration: none;
            font-weight: 700;
            border-bottom: 1.5px solid rgba(29, 78, 216, 0.2);
            transition: all 0.2s;
        }
        .premium-banner-link:hover {
            color: #2563EB !important;
            border-bottom-color: #2563EB;
        }
        /* Mặc định trên PC màn hình lớn: Hiện đủ 6 cột thẳng hàng chằn chặn */
        .kpi-grid {
            display: grid !important;
            grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
            gap: 16px !important;
            width: 100% !important;
            margin-bottom: 28px !important;
        }
        
        /* Cấu hình thẻ KPI (Đậm đà + Đổ bóng sâu + Hover) */
        .kpi-card {
            background: #FFFFFF !important;
            border-radius: 16px !important;
            padding: 16px !important;
            box-sizing: border-box !important;
            box-shadow: 0 10px 20px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.08) !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
        }
        
        .kpi-card:hover {
            transform: translateY(-6px) !important;
            box-shadow: 0 25px 30px -5px rgba(15, 23, 42, 0.16), 0 12px 16px -6px rgba(15, 23, 42, 0.16) !important;
        }
        
        .kpi-label {
            font-size: 13px !important;
            font-weight: 800 !important; 
            letter-spacing: 0.5px !important;
        }
        
        .kpi-num {
            font-size: 28px !important;
            font-weight: 800 !important;
            color: #0F172A !important;
            margin: 8px 0 4px 0 !important;
            line-height: 1.1 !important;
            white-space: nowrap !important;
        }
        
        .kpi-text {
            font-size: 14px !important;
            font-weight: 800 !important;
            line-height: 1.4 !important;
        }

        /* ========================================================================= */
        /* KHU VỰC ĐIỀU CHỈNH KHI BẮT BUỘC PHẢI XUỐNG DÒNG (RESPONSIVE BREAKPOINTS)   */
        /* ========================================================================= */

        /* [Trường hợp 1] Màn hình Laptop nhỏ / Máy tính bảng nằm ngang: Tự rớt thành 2 hàng, mỗi hàng 3 ô đều tăm tắp */
        @media (max-width: 1300px) {
            .kpi-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            }
        }

        /* [Trường hợp 2] Máy tính bảng đứng / Trình duyệt thu nhỏ sâu: Tự rớt thành 3 hàng, mỗi hàng 2 ô đều tăm tắp */
        @media (max-width: 850px) {
            .kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
                gap: 12px !important;
            }
            .kpi-card {
                padding: 12px !important;
            }
            .kpi-num {
                font-size: 24px !important; /* Hạ nhẹ size số để tránh tràn viền trên màn nhỏ */
            }
            .kpi-text {
                font-size: 12.5px !important;
            }
        }
        /* Khung Chân dung Đặc quyền */
        .premium-profile-card {
            background: #FFFFFF;
            border-radius: 24px;
            padding: 26px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.06);
            margin-bottom: 20px;
        }
        .group-badge {
            display: inline-block;
            padding: 6px 14px;
            font-size: 12px !important;
            font-weight: 800;
            border-radius: 30px;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .action-gradient-box {
            background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
            border-radius: 16px;
            padding: 20px;
            border: 2px dashed #CBD5E1;
            margin-top: 20px;
        }
        .action-box-title { 
            font-size: 13px !important; 
            font-weight: 800; 
            color: #0F172A; 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
            margin-bottom: 6px; 
        }
        .action-box-text { 
            line-height: 1.6; 
            font-weight: 600; 
            color: #334155;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)
   
    # CHUẨN BỊ THÔNG TIN CÁC NHÓM KHÁCH HÀNG
    STRATEGIES = {
        "Nhóm 1: Khách Hàng Thụ Động": {
            "short_label": "Nhóm 1", "old_id": "Normal_2", "icon": "snooze", "type": "Phổ thông",
            "color": "#10B981", "bg_light": "rgba(16, 185, 129, 0.12)",
            "desc": "Chiếm số đông nhưng rất ít khi phát sinh giao dịch quẹt thẻ chi tiêu mặc dù tài khoản luôn có số dư ổn định.",
            "action": "Gửi chiến dịch kích cầu tự động qua thông báo App: 'Quẹt thẻ chi tiêu từ 1 triệu đồng nhận ngay voucher mua sắm giảm 100k' vào khung giờ vàng cuối tuần."
        },
        "Nhóm 2: Tiêu Dùng Thông Minh": {
            "short_label": "Nhóm 2", "old_id": "Normal_1", "icon": "shopping_bag", "type": "Năng động",
            "color": "#3B82F6", "bg_light": "rgba(59, 130, 246, 0.12)",
            "desc": "Mức nợ tích lũy duy trì ở ngưỡng rất an toàn. Tần suất quẹt thẻ sắm sửa hàng hóa, đăng ký trả góp diễn ra cực kỳ đều đặn và khoa học.",
            "action": "Tặng đặc quyền hoàn tiền cố định (Cashback) từ 2% - 5% khi chi tiêu mua sắm, đồng thời chủ động gợi ý nâng hạn mức tín dụng trực tuyến."
        },
        "Nhóm 3: Vay Tiêu Dùng Nhỏ": {
            "short_label": "Nhóm 3", "old_id": "Normal_3", "icon": "payments", "type": "Dòng tiền",
            "color": "#8B5CF6", "bg_light": "rgba(139, 92, 246, 0.12)",
            "desc": "Lười quẹt thẻ tại các điểm bán hàng thông thường, tuy nhiên tần suất và giá trị rút tiền mặt trực tiếp tại các cây ATM lại rất lớn.",
            "action": "Gửi gợi ý chuyển đổi tính năng rút tiền mặt ngắn hạn sang sản phẩm 'Trả góp tiền mặt qua thẻ' với lãi suất cố định thấp để quản trị rủi ro."
        },
        "Nhóm 4: VIP Tiêu Dùng Cao Cấp": {
            "short_label": "Nhóm 4", "old_id": "Outlier_2", "icon": "military_tech", "type": "Thượng lưu",
            "color": "#D97706", "bg_light": "rgba(217, 119, 6, 0.12)",
            "desc": "Được cấp hạn mức lớn, quẹt món to thẳng tay cho các dịch vụ xa xỉ, giải trí cao cấp và luôn hoàn trả dư nợ sòng phẳng, đúng hạn.",
            "action": "Tích điểm thưởng nhân hệ số (X2, X3) để đổi quà tặng giá trị cao, phát hành thẻ đồng thương hiệu ưu đãi tại các trung tâm thương mại lớn."
        },
        "Nhóm 5: Siêu VIP Chi Tiêu Khủng": {
            "short_label": "Nhóm 5", "old_id": "Outlier_1", "icon": "diamond", "type": "Cá mập (Whale)",
            "color": "#DC2626", "bg_light": "rgba(220, 38, 38, 0.12)",
            "desc": "Nhóm khách hàng siêu cấp. Sức chi tiêu khổng lồ, tất toán dư nợ minh bạch tuyệt đối, đem lại nguồn lợi nhuận lớn nhất cho ngân hàng.",
            "action": "Cung cấp chuỗi đặc quyền tối thượng: Tặng thẻ đen quyền lực, miễn phí phòng chờ thương gia sân bay quốc tế, lượt chơi Golf và trợ lý VIP 1-1."
        },
        "Nhóm 6: Tín Dụng Rút Tiền Mặt Cao": {
            "short_label": "Nhóm 6", "old_id": "Outlier_3", "icon": "report_problem", "type": "Rủi ro cao",
            "color": "#EA580C", "bg_light": "rgba(234, 88, 12, 0.12)",
            "desc": "Dư nợ chạm trần, coi thẻ tín dụng như khoản vay thấu chi để liên tục rút tiền mặt. Đem lại doanh thu phí lớn nhưng tiềm ẩn rủi ro nợ xấu.",
            "action": "Thắt chặt kiểm soát điểm tín dụng chặt chẽ, áp trần giới hạn rút tiền mặt tối đa hàng tháng và hướng khách hàng sang các gói trả góp 0%."
        }  
    }
    REVERSE_MAP = {v["old_id"]: k for k, v in STRATEGIES.items()}

    COLOR_HTML_MAP = {k: f"<span style='color:{v['color']}; font-weight:800;'>{v['short_label']}</span>" for k, v in STRATEGIES.items()}
    COLOR_MAP_SHORT = {f"<span style='color:{v['color']}; font-weight:800;'>{v['short_label']}</span>": v["color"] for k, v in STRATEGIES.items()}

    # TẢI DỮ LIỆU
    @st.cache_data
    def load_data():
        df = pd.read_csv(CSV_PATH)
        df["Full_Nhóm"] = df["Cluster"].map(REVERSE_MAP).fillna(df["Cluster"])
        df["Nhóm"] = df["Full_Nhóm"].map(COLOR_HTML_MAP)
        return df
    df_final = load_data()

    # --- 1. Tiêu đề trang ---
    st.markdown(
    """
    <h1 style="text-align: center; background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #1E3A8A 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">
        Phân Khúc Khách Hàng Thẻ Tín Dụng
    </h1>
    """, 
    unsafe_allow_html=True
    )

    #  --- 2. Banner nguồn gốc dữ liệu & rào cản bảo mật ---
    st.markdown("""
    <div class="data-source-container">
        <div class="premium-info-card">
            <div class="card-icon-badge badge-blue">
                <span class="material-icons">bar_chart</span>
            </div>
            <div class="card-content-text">
                <b>Hệ thống phân khúc hành vi tiêu dùng của </b><span class="text-highlight">8.950 chủ thẻ tín dụng</span><b> thành </b><span class="text-highlight">6 nhóm khách hàng</span><b> từ bộ dữ liệu uy tín </b> 
                <a class="premium-banner-link" href="https://www.kaggle.com/datasets/arjunbhasin2013/ccdata" target="_blank">Credit Card Dataset for Clustering</a> 
                <b>do tác giả <i>Arjun Bhasin</i> thu thập và đăng tải.</b>
            </div>
        </div>
        <div class="premium-info-card">
            <div class="card-icon-badge badge-shield">
                <span class="material-icons">verified_user</span>
            </div>
            <div class="card-content-text">
                <b>Do rào cản bảo mật nghiêm ngặt tại Việt Nam </b>(<span class="text-highlight">Nghị định 13/2023/NĐ-CP</span>)<b>, việc khai thác dữ liệu giao dịch thực tế từ các ngân hàng nội địa là không thể công khai. Bộ dữ liệu chuẩn hóa quốc tế này được lựa chọn vì phản ánh tốt các chỉ số cốt lõi tương đồng với hệ thống </b><span class="text-highlight">Core Banking</span>: <b><i>Số dư nợ, Thói quen mua sắm, Tần suất rút tiền và Hạn mức tín dụng.</i></b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 3. Chia 3 Tab (Trực quan - Dự đoán - Thông tin) ---
    tab_dashboard, tab_prediction, tab_infomation = st.tabs(["**Tổng Quan & Trực Quan Hóa**", "**Phân Khúc Khách Hàng**", "**Thông Tin Thêm**"])
    
    # 3_1. TAB Trực quan
    with tab_dashboard:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px; margin-top: 20px;">
            <div style="background: #E0F2FE; padding: 4px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                <span class="material-icons" style="color: #2563EB; font-size: 24px; font-family: 'Material Icons' !important; line-height: 1;">groups</span>
            </div>
            <span style="font-size: 20px; font-weight: 800; color: #0F172A; line-height: 1;">Số Lượng Khách Hàng Mỗi Nhóm</span>
        </div>
        """, unsafe_allow_html=True)
        
        counts = df_final["Full_Nhóm"].value_counts()
        kpi_html = '<div class="kpi-grid">'

        for group_id in STRATEGIES.keys():
            icon_name = STRATEGIES[group_id]["icon"]
            color = STRATEGIES[group_id]["color"]
            bg_light = STRATEGIES[group_id]["bg_light"]
            short_label = STRATEGIES[group_id]["short_label"]
            group_name = group_id.split(": ")[1]
            value_str = f"{counts.get(group_id, 0):,}"
            
            card_item = f"""
            <div class="kpi-card" style="border: 2.5px solid {color}; background: linear-gradient(180deg, #FFFFFF 50%, {bg_light} 100%);">
                <div class="kpi-label" style="color: {color}; display: flex; align-items: center; gap: 6px;">
                    <span class="material-icons" style="font-size: 18px; font-family: 'Material Icons' !important;">{icon_name}</span>
                    <span>{short_label}</span>
                </div>
                <div class="kpi-num">{value_str}</div>
                <div class="kpi-text" style="color: {color};">{group_name}</div>
            </div>
            """
            kpi_html += card_item.replace("\n", "").strip()

        kpi_html += '</div>'

        # Xuất ra màn hình và quét sạch dấu Enter thừa lần cuối
        st.markdown(kpi_html.replace("\n", ""), unsafe_allow_html=True)
        
        # =========================================================================
        # NHÚNG CSS RESPONSIVE CHO HỆ THỐNG COLUMNS CỦA STREAMLIT
        # =========================================================================
        st.markdown("""
            <style>
            /* Ep các cột Streamlit xếp dọc */
            @media (max-width: 1400px) {
                /* Tìm đến container chứa các cột của Streamlit và ép nó chuyển sang flex-direction dọc */
                div[data-testid="stHorizontalBlock"] {
                    flex-direction: column !important;
                    gap: 24px !important;
                }
                
                /* Ép các cột con (c_left và c_right) chiếm trọn 100% bề ngang màn hình */
                div[data-testid="stHorizontalBlock"] > div {
                    width: 100% !important;
                    max-width: 100% !important;
                    flex: 1 1 100% !important;
                }
                
                /* Điều chỉnh nhẹ padding của thẻ premium-profile-card trên mobile nếu có */
                .premium-profile-card {
                    padding: 16px !important;
                }
            }
            </style>
        """, unsafe_allow_html=True)

        c_left, c_right = st.columns([55, 45])

        # --- Biểu đồ không gian phân khúc khách hàng ---
        with c_left:
            # Sử dụng biến chuỗi riêng và xử lý sạch khoảng trắng bằng .replace("\n", "")
            title_left = """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px; margin-top: 10px;">
                <div style="background: #E0F2FE; padding: 4px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span class="material-icons" style="color: #2563EB; font-size: 21px; font-family: 'Material Icons' !important; line-height: 1;">hub</span>
                </div>
                <span style="font-size: 20px; font-weight: 800; color: #0F172A; line-height: 1;">Trực Quan Không Gian Phân Khúc Khách Hàng</span>
            </div>
            """
            st.markdown(title_left.replace("\n", ""), unsafe_allow_html=True)
            
            chart_type = st.radio(
                "Chọn kiểu hiển thị bản đồ t-SNE:",
                options=["Bản đồ Mặt phẳng 2D", "Bản đồ Không gian 3D"],
                horizontal=True,
                label_visibility="collapsed"
            )

            df_sorted = df_final.sort_values(by="Nhóm")

            tsne_custom_data = ["Nhóm", "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "PURCHASES_TRX", "PURCHASES_FREQUENCY", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]

            tsne_hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "Balance: $%{customdata[1]:,.0f}<br>"
                "Purchases: $%{customdata[2]:,.0f}<br>"
                "Oneoff Purchases: $%{customdata[3]:,.0f}<br>"
                "Installments Purchases: $%{customdata[4]:,.0f}<br>"
                "Purchase Transactions: %{customdata[5]}<br>"
                "Purchase Frequency: %{customdata[6]:.2f}<br>"
                "Cash Advance: $%{customdata[7]:,.0f}<br>"
                "Credit Limit: $%{customdata[8]:,.0f}<br>"
                "Payments: $%{customdata[9]:,.0f}"
                "<extra></extra>"
            )

            if chart_type == "Bản đồ Không gian 3D":
                fig = px.scatter_3d(
                    df_sorted, x="t-SNE 1", y="t-SNE 2", z="t-SNE 3", color="Nhóm",
                    custom_data=tsne_custom_data,
                    color_discrete_map=COLOR_MAP_SHORT, opacity=0.55, height=540
                )
                fig.update_traces(marker=dict(size=2.5), hovertemplate=tsne_hovertemplate)
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene=dict(
                        xaxis=dict(visible=True, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"), 
                        yaxis=dict(visible=True, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"), 
                        zaxis=dict(visible=True, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"),
                        bgcolor="#FFFFFF"  
                    ),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, title_text="", font=dict(size=14, weight="bold"))
                )
                    
            else:
                fig = px.scatter(
                    df_sorted, x="t-SNE 1", y="t-SNE 2", color="Nhóm",
                    custom_data=tsne_custom_data,
                    color_discrete_map=COLOR_MAP_SHORT, opacity=0.6, height=540
                )
                fig.update_traces(marker=dict(size=4.5), hovertemplate=tsne_hovertemplate)
                fig.update_layout(
                    margin=dict(l=10, r=0, b=0, t=10),
                    xaxis=dict(visible=True, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"),
                    yaxis=dict(visible=True, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, title_text="", font=dict(size=14, weight="bold"))
                )
                
            st.plotly_chart(fig, width='stretch') # Đổi thành width='stretch' để biểu đồ tự bám sát theo độ rộng của cột sau responsive

        # --- Thông tin chi tiết từng nhóm khách hàng ---
        with c_right:
            title_right = """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px; margin-top: 8px;">
                <div style="background: #E0F2FE; padding: 4px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span class="material-icons" style="color: #2563EB; font-size: 24px; font-family: 'Material Icons' !important; line-height: 1;">ads_click</span>
                </div>
                <span style="font-size: 20px; font-weight: 800; color: #0F172A; line-height: 1;">Chiến Lược Khai Thác Nhóm Khách Hàng</span>
            </div>
            """
            st.markdown(title_right.replace("\n", ""), unsafe_allow_html=True)
            
            selected_view = st.selectbox("Chọn cụm nghiệp vụ:", list(STRATEGIES.keys()), label_visibility="collapsed")
            current_data = STRATEGIES[selected_view]
            
            profile_html = f"""
            <div class="premium-profile-card" style="border-top: 6px solid {current_data["color"]};">
                <span class="group-badge" style="background-color: {current_data["bg_light"]}; color: {current_data["color"]}; display: inline-flex; align-items: center; gap: 4px;">
                    <span class="material-icons" style="font-size: 14px; font-family: 'Material Icons' !important;">label</span>
                    Phân khúc: {current_data["type"]}
                </span>
                <div style="font-size: 22px; font-weight: 800; color: #0F172A; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; line-height: 1;">
                    <span class="material-icons" style="color: {current_data["color"]}; font-size: 26px; font-family: 'Material Icons' !important;">{current_data["icon"]}</span>
                    <span>{selected_view}</span>
                </div>
                <div class="profile-desc" style="color: #1E293B; line-height: 1.6; margin-bottom: 6px;">
                    <b>Đặc trưng hành vi tài chính:</b> <i>{current_data["desc"]}</i>
                </div> 
                <div class="action-gradient-box" style="border-color: {current_data["color"]};">
                    <div class="action-box-title" style="color: {current_data["color"]}; font-weight: 800; display: flex; align-items: center; gap: 6px; line-height: 1;">
                        <span class="material-icons" style="font-size: 18px; font-family: 'Material Icons' !important;">ads_click</span>
                        Gợi ý chiến lược / Giải pháp hành động
                    </div>
                    <div class="action-box-text">{current_data["action"]}</div>
                </div>
            </div>
            """
            st.markdown(profile_html.replace("\n", ""), unsafe_allow_html=True)
            
            sub_df = df_final[df_final["Full_Nhóm"] == selected_view]
            columns_mapping = {"BALANCE": "Số dư nợ hiện tại", "PURCHASES": "Tổng tiền mua sắm", "CASH_ADVANCE": "Rút tiền mặt từ thẻ", "CREDIT_LIMIT": "Hạn mức tối đa thẻ", "PAYMENTS": "Số tiền đã trả lại"}

            mean_values = [sub_df[col].mean() for col in columns_mapping.keys()]
            chart_data = pd.DataFrame({"Tên Tiếng Anh": list(columns_mapping.keys()), "Tên Tiếng Việt": list(columns_mapping.values()), "Giá trị trung bình ($)": mean_values})

            mini_fig = px.bar(chart_data, x="Tên Tiếng Việt", y="Giá trị trung bình ($)", color="Tên Tiếng Việt", custom_data=["Tên Tiếng Anh", "Tên Tiếng Việt"], color_discrete_sequence=[current_data["color"]], height=240)
            mini_fig.update_traces(hovertemplate="<b>Chỉ báo tài chính:</b> %{customdata[0]}<br><b>%{customdata[1]}:</b> %{y:,.2f} $<extra></extra>")

            short_labels = ["Số dư nợ<br>hiện tại", "Tổng tiền<br>mua sắm", "Rút tiền mặt<br>từ thẻ", "Hạn mức<br>tối đa thẻ", "Số tiền<br>đã trả lại"]

            mini_fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=50), 
                xaxis=dict(tickmode='array', tickvals=chart_data["Tên Tiếng Việt"], ticktext=short_labels, tickfont=dict(size=11, color="#334155"), tickangle=0),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                xaxis_title=None, yaxis_title="Giá trị trung bình ($)"
            )

            st.plotly_chart(mini_fig, width='stretch')
        
    # 3_2. TAB Dự đoán
    with tab_prediction:
        # Tạo form nhập liệu
        with st.form("input_form"):
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px; margin-top: 20px;">
                <div style="background: #E0F2FE; padding: 4px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span class="material-icons" style="color: #2563EB; font-size: 22px; font-family: 'Material Icons' !important; line-height: 1;">person_add</span>
                </div>
                <span style="font-size: 22px; font-weight: 800; color: #0F172A; line-height: 1;">Nhập thông tin khách hàng</span>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                balance = st.number_input("**Số dư nợ hiện tại (Balance)**", min_value=0.0, value = 3000.0, step=0.1, icon=":material/attach_money:", help="Số nợ hiện tại chủ thẻ chưa hoàn trả")
                payments = st.number_input("**Tổng số tiền đã trả cho ngân hàng (Payments)**", min_value=0.0, value=2000.0, step=0.1, icon=":material/attach_money:", help="Tổng số tiền khách hàng đã thanh toán/chuyển khoản lại cho ngân hàng để trả nợ.")
                credit_limit = st.number_input("**Hạn mức tín dụng được cấp (Credit Limit)**", min_value=0.0, value=4000.0, step=0.1, icon=":material/attach_money:", help="Số tiền tối đa ngân hàng cấp cho chủ thẻ tiêu dùng.")  
            with col2:
                cash_advance = st.number_input("**Tổng giá trị rút tiền mặt (Cash Advance)**", min_value=0.0, value=0.0, step=0.1, icon=":material/attach_money:", help="Tổng số tiền mặt chủ thẻ đã ứng trực tiếp tại cây ATM hoặc quầy giao dịch.")
                purchases = st.number_input("**Tổng giá trị mua sắm (Purchases)**", min_value=0.0, value=1000.0, step=0.1, icon=":material/attach_money:", help="Tổng số tiền quẹt thẻ thanh toán hàng hóa, dịch vụ tại các máy POS/Online.")
                tenure = st.number_input("**Giai đoạn quan sát (Tenure - Tháng)**", min_value=6, max_value=12, value=12, step=1, icon=":material/calendar_month:", help="Số tháng tích lũy dữ liệu hành vi gần đây của khách hàng (Từ 6 - 12 tháng).")
            submitted = st.form_submit_button("**Tiến hành phân khúc khách hàng**")
            
            if submitted:
                # Tải các mô hình và bộ chuẩn hóa
                iqr_bounds = joblib.load(IQR_BOUNDS_PATH)
                scaler_normal = joblib.load(SCALER_NORMAL_PATH)
                scaler_outlier = joblib.load(SCALER_OUTLIER_PATH)
                kmeans_normal = joblib.load(KMEANS_NORMAL_PATH)
                kmeans_outlier = joblib.load(KMEANS_OUTLIER_PATH)
                
                # Kiểm tra xem có phải Outlier hay không dựa trên ranh giới IQR cũ
                user_features = {
                    "BALANCE": balance, 
                    "PURCHASES": purchases, 
                    "CASH_ADVANCE": cash_advance, 
                    "CREDIT_LIMIT": credit_limit, 
                    "PAYMENTS": payments
                }
                
                is_user_outlier = False
                for col, val in user_features.items():
                    if val > iqr_bounds[col]["upper"] or val < iqr_bounds[col]["lower"]:
                        is_user_outlier = True
                        break
                
                # Thứ tự mảng: [BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS]
                input_df = pd.DataFrame([{
                "BALANCE": balance,
                "PURCHASES": purchases,
                "CASH_ADVANCE": cash_advance,
                "CREDIT_LIMIT": credit_limit,
                "PAYMENTS": payments
                }])
                
                if is_user_outlier:
                    scaled_vector = scaler_outlier.transform(input_df)
                    pred_raw = kmeans_outlier.predict(scaled_vector)[0]
                    cluster_mapping = {
                        0: "Nhóm 5: Siêu VIP Chi Tiêu Khủng", 
                        1: "Nhóm 4: VIP Tiêu Dùng Cao Cấp", 
                        2: "Nhóm 6: Tín Dụng Rút Tiền Mặt Cao"
                    } 
                    final_cluster_key = cluster_mapping.get(pred_raw)
                else:
                    scaled_vector = scaler_normal.transform(input_df)
                    pred_raw = kmeans_normal.predict(scaled_vector)[0]
                    cluster_mapping = {
                        0: "Nhóm 2: Tiêu Dùng Thông Minh", 
                        1: "Nhóm 1: Khách Hàng Thụ Động", 
                        2: "Nhóm 3: Vay Tiêu Dùng Nhỏ"
                    }
                    final_cluster_key = cluster_mapping.get(pred_raw)

                matched_strategy = None
                strategy_label = ""
                
                for key, info in STRATEGIES.items():
                    if info.get("old_id") == final_cluster_key or key == final_cluster_key:
                        matched_strategy = info
                        strategy_label = key
                        break
                
                if matched_strategy:
                    brand_color = matched_strategy.get('color', '#2563EB')
                    bg_light = matched_strategy.get('bg_light', 'rgba(37, 99, 235, 0.1)')
                    group_icon = matched_strategy.get('icon', 'person')
                    group_type = matched_strategy.get('type', 'Tổng quan')
                    
                    st.markdown(f"""
                    <style>
                        .predict-profile-card {{
                            background: #FFFFFF;
                            border: 2.5px solid {brand_color} !important;
                            padding: 24px;
                            border-radius: 16px;
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                            margin-bottom: 20px;
                            transition: all 0.3s ease-in-out; /* Tạo độ mượt khi hover */
                        }}
                        .predict-profile-card:hover {{
                            transform: translateY(-4px); /* Nhấc card lên 4px */
                            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); /* Đổ bóng đậm hơn */
                            border-color: {brand_color};
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    profile_html = f"""
                    <div class="predict-profile-card">
                        <span class="group-badge" style="
                            background-color: {bg_light}; 
                            color: {brand_color}; 
                            display: inline-flex; 
                            align-items: center; 
                            gap: 4px;
                            font-size: 13px;
                            font-weight: 700;
                            padding: 4px 12px;
                            border-radius: 20px;
                            margin-bottom: 14px;
                            text-transform: uppercase;
                        ">
                            <span class="material-icons" style="font-size: 14px; font-family: 'Material Icons' !important;">label</span>
                            Phân khúc: {group_type}
                        </span>
                        <div style="font-size: 22px; font-weight: 800; color: #0F172A; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; line-height: 1;">
                            <span class="material-icons" style="color: {brand_color}; font-size: 28px; font-family: 'Material Icons' !important;">{group_icon}</span>
                            <span>{strategy_label}</span>
                        </div> 
                        <div class="profile-desc" style="color: #1E293B; font-size: 15px; line-height: 1.6; margin-bottom: 16px;">
                            <b>Đặc trưng hành vi tài chính:</b> <i>{matched_strategy["desc"]}</i>
                        </div> 
                        <div class="action-gradient-box" style="
                            border: 1px dashed {brand_color};
                            background: {brand_color}04;
                            padding: 16px;
                            border-radius: 12px;
                        ">
                            <div class="action-box-title" style="color: {brand_color}; font-size: 14.5px; font-weight: 800; display: flex; align-items: center; gap: 6px; margin-bottom: 6px; line-height: 1;">
                                <span class="material-icons" style="font-size: 18px; font-family: 'Material Icons' !important;">ads_click</span>
                                Gợi ý chiến lược / Giải pháp hành động
                            </div>
                            <div class="action-box-text" style="font-size: 14.5px; color: #334155; line-height: 1.5; font-weight: 500;">
                                {matched_strategy["action"]}
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(profile_html, unsafe_allow_html=True)

    # 3_3. TAB Thông tin
    with tab_infomation:
        # Cấu hình CSS
        st.markdown("""
            <style>
                /* Định dạng font chữ nội dung đồng bộ 15px */
                .custom-content {
                    font-size: 15px !important;
                    color: #334155;
                    line-height: 1.65;
                }
                
                /* Khối hộp lớn chứa toàn bộ nội dung */
                .premium-card {
                    background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-radius: 14px;
                    padding: 24px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
                    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
                }
                .premium-card:hover {
                    transform: translateY(-5px);
                    border-color: #2563EB;
                    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
                }
                
                /* MẶC ĐỊNH TRÊN PC: Lưới hiển thị 3 cột chằn chặn */
                .data-card-grid {
                    display: grid !important;
                    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                    gap: 16px !important;
                    margin-bottom: 10px !important;
                    margin-top: 15px !important;
                }
                
                /* Thẻ con hiển thị tên cột */
                .data-card {
                    background: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-radius: 10px;
                    padding: 16px;
                    transition: all 0.3s ease;
                    box-sizing: border-box !important;
                }
                .data-card:hover {
                    background: #FFFFFF;
                    border-color: #3B82F6;
                    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.06);
                }
                
                .card-title {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-weight: 700;
                    font-size: 15px;
                    color: #0F172A;
                    margin-bottom: 8px;
                }

                /* ========================================================================= */
                /* CẤU HÌNH RESPONSIVE                             */
                /* ========================================================================= */
                
                /* Màn hình Laptop nhỏ / Máy tính bảng nằm ngang: Chia 2 cột + Căn giữa hàng cuối */
                @media (max-width: 1300px) {
                    .data-card-grid {
                        display: flex !important;
                        flex-wrap: wrap !important;
                        justify-content: center !important; /* Căn giữa hàng cuối */
                    }
                    .data-card {
                        /* Tính toán kích thước để xếp vừa khít 2 cột (trừ đi gap) */
                        flex: 0 0 calc(50% - 8px) !important;
                        max-width: calc(50% - 8px) !important;
                    }
                }
                
                /* Màn hình Điện thoại: Thu về 1 cột dọc toàn bộ */
                @media (max-width: 700px) {
                    .premium-card {
                        padding: 16px !important;
                    }
                    .data-card-grid {
                        display: flex !important;
                        flex-direction: column !important;
                        gap: 12px !important;
                    }
                    .data-card {
                        flex: 0 0 100% !important;
                        max-width: 100% !important;
                        padding: 14px !important;
                    }
                }
            </style>
        """, unsafe_allow_html=True)

        # PHẦN 1: DỮ LIỆU PHÂN TÍCH
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; margin-top: 10px;">
            <div style="background: #E0F2FE; padding: 5px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                <span class="material-icons" style="color: #2563EB; font-size: 23px; font-family: 'Material Icons' !important;">storage</span>
            </div>
            <span style="font-size: 23px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">Dữ liệu phân tích</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="premium-card">
            <p class="custom-content" style="margin: 0;">
                Tập dữ liệu thô ban đầu của ngân hàng quản lý gồm <b>18 trường thông tin</b> khác nhau của chủ thẻ. Nhằm tối ưu hóa hiệu năng tính toán và tập trung giải quyết bài toán cốt lõi là <b>Tăng trưởng doanh số & Kích cầu chi tiêu</b>, hệ thống tiến hành sàng lọc và trích xuất ra <b>9 cột chỉ báo hành vi</b> quan trọng nhất:
            </p>
            <div class="data-card-grid">
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">account_balance_wallet</span> BALANCE</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Số dư tài khoản:</b> Số tiền nợ tín dụng hiện tại mà khách hàng chưa thanh toán cho ngân hàng.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">shopping_bag</span> PURCHASES</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Tổng giá trị mua sắm:</b> Toàn bộ số tiền tài khoản đã quẹt thẻ chi tiêu mua sắm hàng hóa dịch vụ.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">update</span> PURCHASES_FREQUENCY</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Tần suất quẹt thẻ:</b> Mức độ thường xuyên mua sắm của chủ thẻ (Chỉ số từ 0 đến 1).</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">payments</span> ONEOFF_PURCHASES</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Mua sắm thanh toán ngay:</b> Số tiền chi tiêu cho các giao dịch quẹt thẻ trả thẳng 1 lần.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">credit_card</span> INSTALLMENTS_PURCHASES</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Mua sắm trả góp:</b> Giá trị tiền chi tiêu phục vụ cho các dịch vụ đăng ký trả góp hàng tháng.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">receipt_long</span> PURCHASES_TRX</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Số lượt giao dịch mua:</b> Tổng số lần phát sinh hóa đơn chi tiêu mua sắm thành công.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">local_atm</span> CASH_ADVANCE</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Rút tiền mặt:</b> Tổng số tiền mặt mà chủ thẻ đã rút trực tiếp tại cây ATM qua thẻ tín dụng.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">speed</span> CREDIT_LIMIT</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Hạn mức tín dụng:</b> Ngưỡng tiêu tiền tối đa được ngân hàng phê duyệt và cấp cho chủ thẻ.</p>
                </div>
                <div class="data-card">
                    <div class="card-title"><span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; color: #2563EB; vertical-align: middle;">fact_check</span> PAYMENTS</div>
                    <p class="custom-content" style="font-size: 14px !important; color: #475569; margin: 0;"><b>Số tiền đã trả:</b> Tổng tiền khách hàng đã nộp lại cho ngân hàng để thanh toán dư nợ kỳ trước.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # PHẦN 2: KỸ THUẬT PHÂN TÍCH
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; margin-top: 40px;">
            <div style="background: #E0F2FE; padding: 5px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                <span class="material-icons" style="color: #2563EB; font-size: 23px; font-family: 'Material Icons' !important;">alt_route</span>
            </div>
            <span style="font-size: 23px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">Kỹ thuật phân tích</span>
        </div>
        """, unsafe_allow_html=True)

       # Bước 1
        st.markdown("""
        <div class="premium-card" style="margin-bottom: 15px;">
            <div style="background: #F8FAFC; padding: 20px; border-radius: 12px; border-left: 4px solid #60A5FA; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 12px; color: #2563EB; font-weight: 700; font-size: 15px;">
                    <span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; vertical-align: middle;">psychology</span> Bước 1: Huấn luyện học máy trên 5 cột cốt lõi
                </div>
                <p class="custom-content" style="margin: 0; color: #475569; line-height: 1.6;">
                    Nhằm tránh nhiễu toán học, thuật toán K-Means chỉ sử dụng <b>5 biến dòng tiền chính</b> để phân cụm:<br>
                    <span style="display: inline-block; background: #EFF6FF; color: #3B82F6; font-weight: 600; font-size: 13px; padding: 2px 8px; border-radius: 6px; margin: 4px 2px; border: 1px solid #DBEAFE;">BALANCE</span>
                    <span style="display: inline-block; background: #EFF6FF; color: #3B82F6; font-weight: 600; font-size: 13px; padding: 2px 8px; border-radius: 6px; margin: 4px 2px; border: 1px solid #DBEAFE;">PURCHASES</span>
                    <span style="display: inline-block; background: #EFF6FF; color: #3B82F6; font-weight: 600; font-size: 13px; padding: 2px 8px; border-radius: 6px; margin: 4px 2px; border: 1px solid #DBEAFE;">CASH_ADVANCE</span>
                    <span style="display: inline-block; background: #EFF6FF; color: #3B82F6; font-weight: 600; font-size: 13px; padding: 2px 8px; border-radius: 6px; margin: 4px 2px; border: 1px solid #DBEAFE;">CREDIT_LIMIT</span>
                    <span style="display: inline-block; background: #EFF6FF; color: #3B82F6; font-weight: 600; font-size: 13px; padding: 2px 8px; border-radius: 6px; margin: 4px 2px; border: 1px solid #DBEAFE;">PAYMENTS</span><br>
                    Dữ liệu được chia thành tập <b>Số Đông (6,746 dòng)</b> và tập <b>Ngoại Lai (2,204 dòng)</b> để tiến hành tìm điểm gãy tối ưu (Elbow Method) và gán nhãn độc lập (Mỗi tập chia thành 3 cụm).
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Đồ thị elbow method
        col_el1, col_el2 = st.columns(2)

        with col_el1:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px; justify-content: center;">
                <span class="material-icons" style="font-family: 'Material Icons' !important; font-size: 18px; color: #2563EB; vertical-align: middle;">trending_down</span>
                <span style="font-weight: 700; color: #475569; font-size: 14px;">Elbow Method - Tập Số Đông (Normal) ➔ K = 3</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(ELBOW_NORMAL_PATH, width='stretch')

        with col_el2:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px; justify-content: center;">
                <span class="material-icons" style="font-family: 'Material Icons' !important; font-size: 18px; color: #2563EB; vertical-align: middle;">trending_down</span>
                <span style="font-weight: 700; color: #475569; font-size: 14px;">Elbow Method - Tập Ngoại Lai (Outliers) ➔ K = 3</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(ELBOW_OUTLIER_PATH, width='stretch')

        # Bước 2
        st.markdown("""
        <div class="premium-card" style=" margin-bottom: 15px;">
            <div style="background: #F8FAFC; padding: 20px; border-radius: 12px; border-left: 4px solid #60A5FA; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 12px; color: #2563EB; font-weight: 700; font-size: 15px;">
                    <span class="material-icons" style="font-family: 'Material Icons'; font-size: 18px; vertical-align: middle;">analytics</span> Bước 2: Phân tích đặc trưng dựa trên 9 cột
                </div>
                <p class="custom-content" style="margin: 0; color: #475569; line-height: 1.6;">
                    <b>Sau khi đã phân tách thành công thành 6 cụm khách hàng rõ rệt</b>, hệ thống phân tích hành vi từng cụm từ trung bình <b>9 cột thuộc tính ban đầu</b>.<br>
                    Lúc này, hệ thống phân tích thêm các chỉ số mở rộng (tần suất, hình thức mua sắm, số lượt giao dịch) để khắc họa trọn vẹn chân dung đặc trưng của từng nhóm khách hàng.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


       
