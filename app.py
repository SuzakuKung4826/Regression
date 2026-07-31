# -*- coding: utf-8 -*-
"""
California Housing Price Predictor
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
    page_title="🏠 California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        border-radius: 22px;
        border: 1px solid rgba(239, 108, 0, 0.28);
        background:
            radial-gradient(circle at top left, rgba(255, 179, 0, 0.22), transparent 55%),
            linear-gradient(135deg, rgba(239, 108, 0, 0.16), rgba(255, 179, 0, 0.10));
        box-shadow: 0 10px 30px rgba(239, 108, 0, 0.12);
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ef6c00, #ffb300);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        opacity: 0.82;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0;
    }
    .hero .badge {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #ef6c00, #e65100);
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Gradient divider */
    .divider {
        height: 3px;
        border: none;
        border-radius: 3px;
        margin: 1.5rem 0;
        background: linear-gradient(90deg, transparent, #ef6c00, #ffb300, transparent);
    }

    /* Input group card */
    .input-card {
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        border: 1px solid rgba(239, 108, 0, 0.25);
        background: rgba(255, 179, 0, 0.04);
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        padding: 1.4rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(239, 108, 0, 0.22);
        background: rgba(255, 179, 0, 0.05);
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 25px rgba(239, 108, 0, 0.18);
    }
    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        opacity: 0.7;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.3rem 0 0 0;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #ef6c00 0%, #ffb300 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 35px rgba(239, 108, 0, 0.3);
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
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    .result-card p {
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.9;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ef6c00, #e65100);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 25px rgba(239, 108, 0, 0.3);
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
    st.markdown("## 🏠 California Housing")
    st.markdown("### Price Predictor")
    st.markdown("---")
    st.markdown("""
    **โมเดล:** Random Forest Regressor
    **Dataset:** California Housing
    **Accuracy:** R² = 0.81
    """)

    # Feature descriptions
    with st.expander("📖 คำอธิบาย Features"):
        st.markdown("""
        - **MedInc**: รายได้เฉลี่ย (×$10,000)
        - **HouseAge**: อายุบ้านเฉลี่ย
        - **AveRooms**: จำนวนห้องเฉลี่ย
        - **AveBedrms**: จำนวนห้องนอนเฉลี่ย
        - **Population**: จำนวนประชากร
        - **AveOccup**: จำนวนคนต่อครัวเรือน
        - **Latitude**: ละติจูด
        - **Longitude**: ลองจิจูด
        """)

# ==================== Main Content ====================
st.markdown("""
<div class='hero'>
    <h1>🏠 California Housing Price Predictor</h1>
    <p>ทำนายราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning</p>
    <span class='badge'>🌲 Random Forest · R² = 0.81</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ==================== Input Section ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📍 Location & Demographics</div>", unsafe_allow_html=True)
    latitude = st.slider("🌐 Latitude", 32.0, 42.0, 35.0, 0.01)
    longitude = st.slider("🌐 Longitude", -124.0, -114.0, -119.0, 0.01)
    population = st.slider("👥 Population", 1.0, 30000.0, 1500.0, 10.0)
    ave_occup = st.slider("👨‍👩‍👧‍👦 Avg. Occupancy", 1.0, 10.0, 3.0, 0.1)
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}), zoom=4, height=180)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏡 House Characteristics</div>", unsafe_allow_html=True)
    med_inc = st.slider("💰 Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("🏚️ House Age (years)", 1.0, 52.0, 20.0, 1.0)
    ave_rooms = st.slider("🚪 Avg. Rooms", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.slider("🛏️ Avg. Bedrooms", 0.5, 5.0, 1.1, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== Prediction Button ====================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 ทำนายราคาบ้าน", use_container_width=True)

# ==================== Prediction Logic ====================
if predict_button:
    with st.spinner("🔮 กำลังทำนาย..."):
        # สร้าง input array
        input_data = np.array([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude
        ]])

        # Scale ข้อมูล
        input_scaled = scaler.transform(input_data)

        # ทำนาย
        prediction = model.predict(input_scaled)[0]
        prediction_usd = prediction * 100000

    # แสดงผลลัพธ์
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='result-card'>
        <h2>💰 ราคาบ้านที่ทำนายได้</h2>
        <div class='result-value'>${prediction_usd:,.0f}</div>
        <p>หรือประมาณ {prediction:,.2f} × $100,000</p>
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
            <p class='metric-value'>${prediction_usd:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        price_per_room = prediction_usd / ave_rooms if ave_rooms > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>🚪</div>
            <p class='metric-label'>Price / Room</p>
            <p class='metric-value'>${price_per_room:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        price_per_person = prediction_usd / population if population > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>👥</div>
            <p class='metric-label'>Price / Person</p>
            <p class='metric-value'>${price_per_person:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        confidence = 81  # R² score
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
            color_continuous_scale=['#e65100', '#ef6c00', '#ffb300'],
            title='🔍 ความสำคัญของ Features ในการทำนาย'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, family='Poppins, sans-serif', color='#2c3e50'),
            title_font=dict(size=16, color='#2c3e50'),
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
st.markdown("<p style='text-align:center;color:#5a6b8c;margin-top:30px;'>Made with Streamlit · Machine Learning Projects</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#5a6b8c;margin-top:4px;'>Develop By tpp72</p>", unsafe_allow_html=True)
