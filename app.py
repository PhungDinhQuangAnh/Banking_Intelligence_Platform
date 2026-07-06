from streamlit_option_menu import option_menu
import streamlit as st
from Loan_Default_Prediction.loan_default_app import run_loan_default_app
from Credit_Card_Segmentation.credit_card_app import run_credit_card_app

st.set_page_config(page_title="Banking Intelligence Platform", page_icon=":bank:", layout="wide", initial_sidebar_state="expanded")
st.markdown('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">', unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">', unsafe_allow_html=True)

# --- Cấu hình CSS cho phần Sidebar ---
st.markdown("""
    <style>           
        .section-title {
            font-size: 19px;
            font-weight: 600;
            color: #2E86C1;
            margin: 0px 0 10px;
            border-bottom: 1px solid #D6EAF8;
        }
        .info-card {
            animation: fadeIn 1s ease-in-out;
            background-color: #fefefe;
            padding: 14px 18px;
            border-left: 4px solid #5DADE2;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            font-size: 15px;
            line-height: 1.65;
            margin-bottom: 25px;
            text-align: justify;
        }
        .info-card:hover {
            background-color: #E6F7FF;
            transform: scale(1.003);
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        }
        .author-card {
            background-color: #fefefe;
            border-left: 4px solid #5DADE2;
            padding: 14px 18px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            font-size: 15px;
            line-height: 1.65;
            font-weight: 500;
        }
        .author-card:hover {
            background-color: #E6F7FF;
            transform: scale(1.003);
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)
    
# --- Cấu hình CSS cân đối các Tab ---
st.markdown("""
    <style>
        /* --- CHỈ ÁP DỤNG TRÊN MÀN HÌNH MÁY TÍNH --- */
        @media (min-width: 650px) {
            /* Ép container chứa tab mở rộng 100%*/
            div[data-testid="stTabs"] [role="tablist"] {
                display: flex !important;
                width: 100% !important;
                justify-content: space-between !important;
            }
            /* Ép các nút Tab chia đều không gian 25% */
            div[data-testid="stTabs"] button[role="tab"] {
                flex: 1 1 0% !important;
                width: 100% !important;
                max-width: 25% !important;
                text-align: center !important;
                padding: 12px 2px !important;
                font-weight: 500 !important;
                transition: all 0.25s ease-in-out !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# --- Thiết kế phần Sidebar ---
with st.sidebar:
    # --- Header nổi bật ---
    st.markdown("""
        <div style="
            font-size: 24px;
            font-weight: bold;
            color: #1B4F72;
            border-bottom: 3px solid #2980B9;
            text-align: center;
            animation: slideIn 1s ease;
        ">
            <span class='material-icons' style='vertical-align: middle; font-size: 22px;'>insights</span> Banking Intelligence
        </div>

        <style>
            @keyframes slideIn {
            from {opacity: 0; transform: translateX(-20px);}
            to {opacity: 1; transform: translateX(0);}
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""""")

    # --- Menu chọn mô hình ---
    st.markdown("""
    <div class='section-title' style='display: flex; align-items: center; gap: 5px;'>
        <span class='material-icons' style='color: #2E86C1; font-size: 24px; vertical-align: middle;'>search</span>
        <span>Chọn mô hình</span>
    </div>
    """, unsafe_allow_html=True)
    selected_app = option_menu(
        menu_title = None,
        options = ["Phân Khúc Khách Hàng Thẻ Tín Dụng", "Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay vốn"],
        icons = ["shield-check", "people"],
        default_index = 0,
        styles = {
            "container": {
                "padding": "6px !important", 
                "background-color": "#fefefe",
                "border-left": "4px solid #5DADE2", 
                "border-radius": "10px", 
                "box-shadow": "0 2px 6px rgba(0,0,0,0.04);",
            },
            "icon": {
                "color": "#2E86C1", 
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "font-weight": "500",
                "color": "#566573",
                "padding": "10px 12px",
                "margin": "4px 0",
                "--hover-color": "#E6F7FF",
                "transition": "all 0.3s ease-in-out",
                "border-radius": "10px"
            },
            "nav-link-selected": {
                "background-color": "#AED6F1",
                "font-weight": "700",
                "color": "#1B4F72",
                "border-radius": "10px",
                "box-shadow": "0 2px 6px rgba(41, 128, 185, 0.15)"
            },
        }
    )

    # --- Giới thiệu ---
    st.markdown("""
    <div class='section-title' style='display: flex; align-items: center; gap: 9px;'>
        <span class='material-icons' style='font-size: 24px; vertical-align: middle;'>info</span>
        <span>Giới thiệu</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <b>Banking Intelligence Platform</b> là nền tảng tích hợp <b>thuật toán AI</b> với <b>mục tiêu</b> hỗ trợ doanh nghiệp:
            <ul style="padding-left:5px;">
                <li>Phân khúc chủ thẻ tín dụng dựa trên hành vi tài chính, nhằm tăng trưởng doanh số & kích cầu chi tiêu.</li>
                <li>Thẩm định rủi ro, giảm tải thủ công, tự động duyệt hồ sơ vay vốn an toàn & từ chối hồ sơ vay vốn rủi ro cao.</li>
            </ul>
            <span style="font-size:13px; color:#777;"><b><i>⚠️ Kết quả mang tính tham khảo, hỗ trợ ra quyết định!</i></b></span>
        </div>
    """, unsafe_allow_html=True)

    # --- Tác giả ---
    st.markdown("""
    <div class='section-title' style='display: flex; align-items: center; gap: 5px;'>
        <span class='material-icons'; font-size: 24px; vertical-align: middle;'>person</span>
        <span>Tác giả</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="author-card">
            <div style="color:#1B4F72; font-weight:600;">Phùng Đình Quang Anh</div>
            <div style="font-size:15px; color:#999; margin-top:2px;">© 2026 All rights reserved.</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Nút GitHub ---
    st.markdown("""
        <style>
            a.github-button {
                display: inline-block;
                background-color: #24292e;
                color: white !important;
                text-decoration: none !important;
                padding: 10px 22px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.15);
                transition: all 0.25s ease;
            }
            a.github-button:hover {
                background-color: #2f363d;
                transform: translateY(-3px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.25);
            }
        </style>

        <div style="text-align: center; margin-top: 16px;">
            <a href="https://github.com/PhungDinhQuangAnh/Banking_Intelligence_Platform" target="_blank" class="github-button">
                <i class="fa-brands fa-github" style="font-size: 20px; vertical-align: middle;"></i> Mã nguồn GitHub
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- Nội dung tương ứng với từng mô hình ---
if selected_app == "Phân Khúc Khách Hàng Thẻ Tín Dụng":
    run_credit_card_app()

elif selected_app == "Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay vốn":
    run_loan_default_app()

