# -*- coding: utf-8 -*-
"""
Automobile Price Predictor
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="🚗 Automobile Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(99, 102, 241, 0.16), transparent 42%),
            radial-gradient(circle at 88% 18%, rgba(34, 211, 238, 0.12), transparent 45%),
            radial-gradient(circle at 50% 100%, rgba(168, 85, 247, 0.10), transparent 50%),
            #0b0f19;
    }

    section[data-testid="stSidebar"] {
        background: #0e1320;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.6rem 1.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background:
            radial-gradient(circle at top left, rgba(99, 102, 241, 0.25), transparent 55%),
            radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.18), transparent 55%),
            rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
        animation: fadeInUp 0.6s ease;
    }
    .hero h1 {
        font-size: 2.9rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #818cf8, #22d3ee 60%, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        opacity: 0.72;
        font-size: 1.08rem;
        font-weight: 400;
        margin: 0;
        color: #e7e9f5;
    }
    .hero .badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #e7e9f5;
    }

    /* Gradient divider */
    .divider {
        height: 1px;
        border: none;
        margin: 1.8rem 0;
        background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.6), rgba(34, 211, 238, 0.6), transparent);
    }

    /* Input group card */
    .input-card {
        border-radius: 18px;
        padding: 1.5rem 1.6rem 0.7rem 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
        transition: border-color 0.25s ease;
    }
    .input-card:hover {
        border-color: rgba(129, 140, 248, 0.35);
    }

    /* Metric cards */
    .metric-card {
        padding: 1.5rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(14px);
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        border-color: rgba(129, 140, 248, 0.4);
        box-shadow: 0 14px 30px rgba(99, 102, 241, 0.2);
    }
    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        opacity: 0.6;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 0;
        color: #e7e9f5;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.3rem 0 0 0;
        color: #ffffff;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #22d3ee 120%);
        padding: 2.4rem 1.5rem;
        border-radius: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 45px rgba(99, 102, 241, 0.35);
        animation: fadeInUp 0.5s ease;
    }
    .result-card h2 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
        opacity: 0.9;
    }
    .result-value {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.6rem 0;
        text-shadow: 2px 2px 12px rgba(0,0,0,0.25);
    }
    .result-card p {
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.9;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.45);
    }

    /* Misc widget polish */
    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
    }
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    iframe {
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """โหลดโมเดลและ scaler จากไฟล์"""
    try:
        model = joblib.load('model_files/rf_model.pkl')
        scaler = joblib.load('model_files/scaler.pkl')
        feature_names = joblib.load('model_files/feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

model, scaler, feature_names = load_model()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🚗 Automobile")
    st.markdown("### Price Predictor")
    st.markdown("---")
    st.markdown("""
    **โมเดล:** Random Forest Regressor
    **Dataset:** Automobile (1985 Auto Imports)
    **Accuracy:** R² = 0.94
    """)

    # Feature descriptions
    with st.expander("📖 คำอธิบาย Features"):
        st.markdown("""
        - **engine-size**: ขนาดเครื่องยนต์ (ลูกบาศก์นิ้ว)
        - **horsepower**: กำลังเครื่องยนต์ (แรงม้า)
        - **curb-weight**: น้ำหนักตัวรถเปล่า (ปอนด์)
        - **city-mpg**: อัตราสิ้นเปลืองในเมือง (ไมล์/แกลลอน)
        - **highway-mpg**: อัตราสิ้นเปลืองทางหลวง (ไมล์/แกลลอน)
        - **wheel-base**: ระยะฐานล้อ (นิ้ว)
        - **length**: ความยาวรถ (นิ้ว)
        - **width**: ความกว้างรถ (นิ้ว)
        """)

# ==================== Main Content ====================
st.markdown("""
<div class='hero'>
    <h1>🚗 Automobile Price Predictor</h1>
    <p>ทำนายราคารถยนต์ด้วย Machine Learning</p>
    <span class='badge'>🌲 Random Forest · R² = 0.94</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ==================== Input Section ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚙️ Engine & Performance</div>", unsafe_allow_html=True)
    engine_size = st.slider("🔧 Engine Size (cu-in)", 60.0, 330.0, 130.0, 1.0)
    horsepower = st.slider("🐎 Horsepower", 45.0, 265.0, 105.0, 1.0)
    city_mpg = st.slider("⛽ City MPG", 12.0, 50.0, 25.0, 1.0)
    highway_mpg = st.slider("🛣️ Highway MPG", 15.0, 55.0, 31.0, 1.0)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📐 Body & Dimensions</div>", unsafe_allow_html=True)
    curb_weight = st.slider("⚖️ Curb Weight (lbs)", 1480.0, 4100.0, 2560.0, 10.0)
    wheel_base = st.slider("🛞 Wheel Base (in)", 86.0, 121.0, 99.0, 0.1)
    length = st.slider("📏 Length (in)", 140.0, 210.0, 174.0, 0.1)
    width = st.slider("↔️ Width (in)", 60.0, 73.0, 66.0, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== Prediction Button ====================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 ทำนายราคารถยนต์", use_container_width=True)

# ==================== Prediction Logic ====================
if predict_button:
    with st.spinner("🔮 กำลังทำนาย..."):
        # สร้าง input array (ต้องเรียงลำดับตาม feature_names)
        input_data = np.array([[
            engine_size, horsepower, curb_weight, city_mpg,
            highway_mpg, wheel_base, length, width
        ]])

        # Scale ข้อมูล
        input_scaled = scaler.transform(input_data)

        # ทำนาย
        prediction = model.predict(input_scaled)[0]

    # แสดงผลลัพธ์
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='result-card'>
        <h2>💰 ราคารถยนต์ที่ทำนายได้</h2>
        <div class='result-value'>${prediction:,.0f}</div>
        <p>ประเมินจากคุณลักษณะของรถที่กรอก</p>
    </div>
    """, unsafe_allow_html=True)

    # แสดงรายละเอียด
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 รายละเอียดการทำนาย</div>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>💵</div>
            <p class='metric-label'>Predicted Price</p>
            <p class='metric-value'>${prediction:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        price_per_hp = prediction / horsepower if horsepower > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>🐎</div>
            <p class='metric-label'>Price / Horsepower</p>
            <p class='metric-value'>${price_per_hp:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        avg_mpg = (city_mpg + highway_mpg) / 2
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>⛽</div>
            <p class='metric-label'>Avg. MPG</p>
            <p class='metric-value'>{avg_mpg:,.1f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        confidence = 94  # R² score
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>🎯</div>
            <p class='metric-label'>Confidence</p>
            <p class='metric-value'>{confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Feature Importance</div>", unsafe_allow_html=True)

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#6366f1', '#8b5cf6', '#22d3ee'],
            title='🔍 ความสำคัญของ Features ในการทำนาย'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, family='Inter, sans-serif', color='#e7e9f5'),
            title_font=dict(size=16, color='#e7e9f5'),
            margin=dict(l=10, r=10, t=60, b=10),
            coloraxis_showscale=False
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # Input Summary
    with st.expander("📋 ดูข้อมูล Input ที่ใช้ทำนาย"):
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown("<p style='text-align:center;color:#8a90a8;margin-top:30px;'>Made with Streamlit · Machine Learning Projects (Automobile Dataset)</p>", unsafe_allow_html=True)
