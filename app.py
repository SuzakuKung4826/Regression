# -*- coding: utf-8 -*-
"""
Automobile Price Predictor
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
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
        font-size: 2.7rem;
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
        font-size: 1.05rem;
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

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.4rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #e7e9f5;
    }

    .divider {
        height: 1px;
        border: none;
        margin: 1.8rem 0;
        background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.6), rgba(34, 211, 238, 0.6), transparent);
    }

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

    .info-card {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        color: #d7dae8;
        line-height: 1.7;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }
    .info-card b, .info-card strong { color: #f1f2fb; }
    .info-card h4 { color: #a5b4fc; margin-top: 0; }

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
    .metric-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
    .metric-label {
        opacity: 0.6; font-size: 0.82rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.6px; margin: 0; color: #e7e9f5;
    }
    .metric-value { font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0 0 0; color: #ffffff; }

    .result-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #22d3ee 120%);
        padding: 2.4rem 1.5rem;
        border-radius: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 45px rgba(99, 102, 241, 0.35);
        animation: fadeInUp 0.5s ease;
    }
    .result-card h2 { margin: 0; font-size: 1.3rem; font-weight: 600; opacity: 0.9; }
    .result-value { font-size: 3.2rem; font-weight: 800; margin: 0.6rem 0; text-shadow: 2px 2px 12px rgba(0,0,0,0.25); }
    .result-card p { font-size: 1.05rem; margin: 0; opacity: 0.9; }

    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white; border: none; padding: 0.75rem 2rem; border-radius: 999px;
        font-weight: 700; font-size: 1.05rem; letter-spacing: 0.3px;
        transition: all 0.25s ease; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 12px 28px rgba(99, 102, 241, 0.45); }

    div[data-testid="stExpander"] {
        border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.08); background: rgba(255, 255, 255, 0.03);
    }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    iframe { border-radius: 14px; }

    /* Developer profile card */
    .dev-card {
        text-align: center;
        padding: 1.2rem 1rem 1.4rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.035);
        margin-bottom: 1.2rem;
    }
    .dev-card img {
        border-radius: 50%;
        border: 3px solid rgba(129, 140, 248, 0.55);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
    }
    .dev-name { color: #f1f2fb; font-weight: 700; font-size: 1.02rem; margin-top: 0.8rem; }
    .dev-detail { color: #9aa0c3; font-size: 0.82rem; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# ==================== Load Model & Comparison Results ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_files/rf_model.pkl')
        scaler = joblib.load('model_files/scaler.pkl')
        feature_names = joblib.load('model_files/feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

@st.cache_data
def load_comparison():
    try:
        with open('model_files/comparison_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

model, scaler, feature_names = load_model()
comparison_results = load_comparison()

# ==================== Sidebar: Developer Info ====================
with st.sidebar:
    st.markdown(
        """
        <div class="dev-card">
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.image("assets/profile_circle.png", width=140)
    st.markdown(
        """
        <div style="text-align:center; margin-top:0.6rem;">
            <div class="dev-name">นายจิรภัทร จันทร์มล</div>
            <div class="dev-detail">รหัสนักศึกษา 664245026</div>
            <div class="dev-detail">หมู่เรียน 66/43</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## 🚗 Automobile")
    st.markdown("### Price Predictor")
    st.markdown("---")
    st.markdown("""
    **โมเดลที่ deploy:** Random Forest Regressor
    **Dataset:** Automobile (1985 Auto Imports)
    **Accuracy:** R² = 0.94
    """)

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

# ==================== Main Header ====================
st.markdown("""
<div class='hero'>
    <h1>🚗 Automobile Price Predictor</h1>
    <p>ทำนายราคารถยนต์ด้วย Machine Learning</p>
    <span class='badge'>🌲 Random Forest · R² = 0.94</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ==================== Tabs (5 หัวข้อตามเกณฑ์การให้คะแนน) ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 ปัญหา & Dataset",
    "🔧 Data Preprocessing",
    "🤖 ทฤษฎีโมเดล",
    "📊 ประเมิน & เปรียบเทียบ",
    "🔮 ทำนายราคา",
])

# ---------------------------------------------------------------------------
# TAB 1: Problem Definition & Dataset
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("<div class='section-title'>🎯 การกำหนดปัญหา</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    <b>โจทย์ปัญหา:</b> ทำนาย "ราคารถยนต์มือหนึ่ง" (ณ ปี 1985) จากสเปกทางเทคนิคของรถ เช่น ขนาดเครื่องยนต์
    แรงม้า น้ำหนักตัวรถ อัตราสิ้นเปลืองเชื้อเพลิง และขนาดตัวถัง โดยไม่ต้องรู้ยี่ห้อหรือรุ่นล่วงหน้า<br><br>
    ปัญหานี้เป็น <b>ปัญหาการถดถอย (Regression)</b> เพราะค่าที่ต้องการทำนาย (ราคา) เป็นตัวเลขต่อเนื่อง
    ไม่ใช่การจัดกลุ่มหรือจำแนกประเภท เหมาะสำหรับการฝึกและเปรียบเทียบโมเดล Machine Learning หลายแบบ
    บนปัญหาเดียวกัน
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📦 เหตุผลที่เลือก Dataset นี้</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    ใช้ <b>Automobile Dataset (1985 Auto Imports Database)</b> ซึ่งเป็นชุดข้อมูลมาตรฐานที่รู้จักกันดีในวงการ
    Machine Learning ด้วยเหตุผลดังนี้:
    <ul>
        <li><b>มีฟีเจอร์ตัวเลขหลากหลายและสัมพันธ์กับราคาโดยตรง</b> เช่น ขนาดเครื่องยนต์ แรงม้า น้ำหนักตัวรถ
        ทำให้เหมาะกับโจทย์ Regression และสามารถอธิบายผลลัพธ์ได้อย่างมีเหตุผล (interpretable)</li>
        <li><b>ใกล้เคียงสถานการณ์จริง</b> — การประเมินราคารถยนต์จากสเปกเป็นปัญหาที่พบได้จริงในธุรกิจซื้อขายรถมือสอง</li>
        <li><b>ขนาดกำลังพอดี</b> (205 แถว) เพียงพอสำหรับฝึกโมเดลและแบ่ง train/test ได้อย่างมีนัยสำคัญ แต่ไม่ใหญ่จนใช้เวลาประมวลผลนาน</li>
        <li><b>ต่างจากชุดข้อมูล California Housing ที่ใช้เดิม</b> — เปลี่ยนมาใช้ข้อมูลที่ทดลองเก็บ/เลือกเอง เพื่อให้ได้ฝึกกระบวนการทำความสะอาดข้อมูลจริง (มีค่าขาดหายที่ต้องจัดการ)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card"><div class="metric-icon">📄</div>
        <p class="metric-label">จำนวนข้อมูล</p><p class="metric-value">205 แถว</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card"><div class="metric-icon">📊</div>
        <p class="metric-label">Features ที่ใช้</p><p class="metric-value">8 ตัวแปร</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card"><div class="metric-icon">🎯</div>
        <p class="metric-label">ตัวแปรเป้าหมาย</p><p class="metric-value">Price ($)</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card"><div class="metric-icon">🧩</div>
        <p class="metric-label">ประเภทปัญหา</p><p class="metric-value">Regression</p></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2: Data Preprocessing
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("<div class='section-title'>🔧 ขั้นตอนการเตรียมข้อมูล (Data Preprocessing)</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>① การคัดเลือกฟีเจอร์ (Feature Selection)</h4>
    Dataset ต้นฉบับมี 26 คอลัมน์ (ทั้งตัวเลขและข้อความ เช่น ยี่ห้อ ประเภทเชื้อเพลิง รูปแบบตัวถัง) แต่เลือกใช้เฉพาะ
    <b>8 ฟีเจอร์ตัวเลข</b> ที่มีความสัมพันธ์ทางกลศาสตร์/วิศวกรรมกับราคาโดยตรง ได้แก่ engine-size, horsepower,
    curb-weight, city-mpg, highway-mpg, wheel-base, length, width เพื่อให้โมเดลเรียนรู้ได้ตรงประเด็นและตีความ
    ผลได้ง่าย (ตัดคอลัมน์ประเภทข้อความที่ต้องเข้ารหัสเพิ่มออกไปก่อน)

    <h4>② การจัดการค่าที่ขาดหาย (Missing Values)</h4>
    คอลัมน์ <b>horsepower</b> มีค่าที่บันทึกเป็นเครื่องหมาย "?" อยู่ 2 แถว (ข้อมูลเก็บไม่ครบ) จึงแปลงค่าดังกล่าว
    เป็น NaN ด้วย <code>pd.to_numeric(errors="coerce")</code> แล้วเติมค่าด้วย <b>ค่ามัธยฐาน (Median Imputation)</b>
    ของคอลัมน์นั้น เพื่อไม่ให้ค่าผิดปกติ (outlier) ดึงค่าเฉลี่ยให้เพี้ยนไป

    <h4>③ การแบ่งชุดข้อมูล (Train/Test Split)</h4>
    แบ่งข้อมูลเป็น <b>Train 80% / Test 20%</b> (random_state=42 เพื่อให้ผลทดลองซ้ำได้) โดย Test set จะไม่ถูกใช้
    ในการฝึกโมเดลเลย เก็บไว้วัดความแม่นยำแบบไม่ลำเอียงเท่านั้น

    <h4>④ การปรับมาตรฐานข้อมูล (Feature Scaling)</h4>
    ฟีเจอร์แต่ละตัวมีสเกลต่างกันมาก เช่น curb-weight อยู่ในหลักพันปอนด์ ในขณะที่ city-mpg อยู่ในหลักสิบ
    จึงใช้ <b>StandardScaler</b> แปลงทุกฟีเจอร์ให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1 โดย <b>fit บน train set
    เท่านั้น</b> แล้วนำพารามิเตอร์เดียวกันไป transform ทั้ง train และ test (ป้องกัน data leakage)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📐 สรุปฟีเจอร์ที่ใช้</div>", unsafe_allow_html=True)
    feat_table = pd.DataFrame({
        "Feature": feature_names,
        "ความหมาย": [
            "ขนาดเครื่องยนต์ (ลูกบาศก์นิ้ว)",
            "กำลังเครื่องยนต์ (แรงม้า)",
            "น้ำหนักตัวรถเปล่า (ปอนด์)",
            "อัตราสิ้นเปลืองในเมือง (ไมล์/แกลลอน)",
            "อัตราสิ้นเปลืองทางหลวง (ไมล์/แกลลอน)",
            "ระยะฐานล้อ (นิ้ว)",
            "ความยาวรถ (นิ้ว)",
            "ความกว้างรถ (นิ้ว)",
        ],
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3: Model Theory
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("<div class='section-title'>🤖 ทฤษฎีของโมเดลที่ใช้เปรียบเทียบ</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>1. Linear Regression</h4>
    หาสมการเส้นตรง (ไฮเปอร์เพลน) ที่ลากผ่านข้อมูลโดยให้ผลรวมกำลังสองของค่าคลาดเคลื่อน (Sum of Squared Residuals)
    น้อยที่สุด สมมติว่าความสัมพันธ์ระหว่างฟีเจอร์และราคาเป็นเส้นตรง ข้อดีคือเข้าใจง่ายและตีความค่าสัมประสิทธิ์ได้ตรงไปตรงมา
    แต่จะทำงานได้ไม่ดีถ้าความสัมพันธ์จริงมีความซับซ้อนแบบไม่เป็นเส้นตรง

    <h4>2. Decision Tree Regressor</h4>
    แบ่งพื้นที่ฟีเจอร์ออกเป็นส่วนย่อยๆ ซ้ำไปเรื่อยๆ (recursive splitting) โดยเลือกจุดแบ่งที่ทำให้ค่าความแปรปรวน
    (variance/MSE) ภายในแต่ละกลุ่มย่อยลดลงมากที่สุด แล้วทำนายด้วยค่าเฉลี่ยของกลุ่มนั้น จับความสัมพันธ์ที่ไม่เป็นเส้นตรง
    ได้ดีกว่า Linear Regression แต่มีแนวโน้ม overfit ถ้าปล่อยให้ต้นไม้ลึกเกินไป

    <h4>3. Random Forest Regressor <span style="color:#a5b4fc;">(โมเดลที่นำไป deploy จริง)</span></h4>
    เป็นเทคนิค <b>Ensemble แบบ Bagging</b> — สร้าง Decision Tree จำนวนมาก (ในที่นี้ 300 ต้น) โดยแต่ละต้นฝึกจาก
    ข้อมูลที่สุ่มแบบ bootstrap (สุ่มแบบใส่คืน) และสุ่มเลือกฟีเจอร์บางส่วนในแต่ละจุดแบ่ง จากนั้นนำผลทำนายของทุกต้นมา
    เฉลี่ยกัน ช่วยลดความแปรปรวน (variance) และลด overfitting เมื่อเทียบกับต้นไม้เดี่ยว ให้ผลลัพธ์ที่แม่นยำและเสถียรกว่า

    <h4>4. Gradient Boosting Regressor</h4>
    เป็นเทคนิค <b>Ensemble แบบ Boosting</b> — สร้าง Decision Tree ทีละต้นตามลำดับ โดยต้นไม้ต้นถัดไปจะพยายามแก้ไข
    ค่าความคลาดเคลื่อน (residual error) ที่เหลือจากต้นก่อนหน้า แล้วรวมผลทำนายแบบถ่วงน้ำหนักด้วยอัตราการเรียนรู้
    (learning rate) มักให้ความแม่นยำสูงที่สุดในบรรดาทั้ง 4 โมเดล แต่ฝึกช้ากว่าและอ่อนไหวต่อการปรับพารามิเตอร์มากกว่า
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>เหตุผลที่เลือก Random Forest เป็นโมเดลที่ deploy ใช้งานจริง</h4>
    จากผลการเปรียบเทียบในแท็บถัดไป Gradient Boosting ให้ค่า R² สูงสุดเพียงเล็กน้อย (ต่างกัน ~0.5%) แต่ Random Forest
    มีความเสี่ยง overfitting ต่ำกว่า ปรับจูนพารามิเตอร์ง่ายกว่า และให้ผลลัพธ์ที่เสถียรกว่าเมื่อข้อมูล input มีความหลากหลาย
    จึงเลือกใช้ Random Forest เป็นโมเดลหลักของแอปพลิเคชันนี้
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 4: Evaluation & Comparison
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("<div class='section-title'>📊 การประเมินและเปรียบเทียบโมเดล</div>", unsafe_allow_html=True)

    if comparison_results:
        comp_df = pd.DataFrame(comparison_results).rename(columns={
            "model": "โมเดล", "r2": "R² Score", "mae": "MAE ($)", "rmse": "RMSE ($)"
        })
        st.markdown("<div class='section-title' style='font-size:1rem;'>📋 ตารางเปรียบเทียบ (วัดจาก Test set)</div>", unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-card" style="margin-top: 0.5rem;">
        <b>ความหมายของตัวชี้วัด:</b><br>
        • <b>R² Score</b> — สัดส่วนความแปรปรวนของราคาที่โมเดลอธิบายได้ ยิ่งใกล้ 1 ยิ่งดี<br>
        • <b>MAE (Mean Absolute Error)</b> — ค่าคลาดเคลื่อนเฉลี่ยแบบสัมบูรณ์ (หน่วยเป็นดอลลาร์) ยิ่งน้อยยิ่งดี<br>
        • <b>RMSE (Root Mean Squared Error)</b> — คล้าย MAE แต่ให้น้ำหนักกับความผิดพลาดขนาดใหญ่มากกว่า ยิ่งน้อยยิ่งดี
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = px.bar(
                comp_df, x="โมเดล", y="R² Score", text="R² Score",
                color="R² Score", color_continuous_scale=['#6366f1', '#8b5cf6', '#22d3ee'],
                title="เปรียบเทียบ R² Score ระหว่างโมเดล",
            )
            fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_r2.update_layout(
                height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, family='Inter, sans-serif', color='#e7e9f5'),
                title_font=dict(size=14, color='#e7e9f5'), coloraxis_showscale=False,
                yaxis_range=[0, 1.05],
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            fig_mae = px.bar(
                comp_df, x="โมเดล", y="MAE ($)", text="MAE ($)",
                color="MAE ($)", color_continuous_scale=['#22d3ee', '#8b5cf6', '#6366f1'],
                title="เปรียบเทียบ MAE ($) ระหว่างโมเดล (ยิ่งต่ำยิ่งดี)",
            )
            fig_mae.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_mae.update_layout(
                height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, family='Inter, sans-serif', color='#e7e9f5'),
                title_font=dict(size=14, color='#e7e9f5'), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        best_model = comp_df.loc[comp_df["R² Score"].idxmax(), "โมเดล"]
        st.success(f"🏆 โมเดลที่ให้ผลลัพธ์แม่นยำที่สุดในการทดสอบนี้คือ **{best_model}** (R² สูงสุด) — แต่แอปนี้เลือก deploy ด้วย Random Forest ตามเหตุผลในแท็บ 'ทฤษฎีโมเดล'")
    else:
        st.warning("ไม่พบไฟล์ผลการเปรียบเทียบโมเดล (model_files/comparison_results.json)")

# ---------------------------------------------------------------------------
# TAB 5: Prediction (Streamlit Application)
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("<div class='section-title'>🔮 ทำนายราคารถยนต์จากสเปกที่กรอก</div>", unsafe_allow_html=True)

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

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 ทำนายราคารถยนต์", use_container_width=True)

    if predict_button:
        with st.spinner("🔮 กำลังทำนาย..."):
            input_data = np.array([[
                engine_size, horsepower, curb_weight, city_mpg,
                highway_mpg, wheel_base, length, width
            ]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='result-card'>
            <h2>💰 ราคารถยนต์ที่ทำนายได้</h2>
            <div class='result-value'>${prediction:,.0f}</div>
            <p>ประเมินจากคุณลักษณะของรถที่กรอก (โดยโมเดล Random Forest)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 รายละเอียดการทำนาย</div>", unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""<div class='metric-card'><div class='metric-icon'>💵</div>
            <p class='metric-label'>Predicted Price</p><p class='metric-value'>${prediction:,.0f}</p></div>""", unsafe_allow_html=True)
        with col_m2:
            price_per_hp = prediction / horsepower if horsepower > 0 else 0
            st.markdown(f"""<div class='metric-card'><div class='metric-icon'>🐎</div>
            <p class='metric-label'>Price / Horsepower</p><p class='metric-value'>${price_per_hp:,.0f}</p></div>""", unsafe_allow_html=True)
        with col_m3:
            avg_mpg = (city_mpg + highway_mpg) / 2
            st.markdown(f"""<div class='metric-card'><div class='metric-icon'>⛽</div>
            <p class='metric-label'>Avg. MPG</p><p class='metric-value'>{avg_mpg:,.1f}</p></div>""", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""<div class='metric-card'><div class='metric-icon'>🎯</div>
            <p class='metric-label'>Confidence (R²)</p><p class='metric-value'>94%</p></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Feature Importance</div>", unsafe_allow_html=True)

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names, 'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig = px.bar(
                importance_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale=['#6366f1', '#8b5cf6', '#22d3ee'],
                title='🔍 ความสำคัญของ Features ในการทำนาย'
            )
            fig.update_layout(
                height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, family='Inter, sans-serif', color='#e7e9f5'),
                title_font=dict(size=16, color='#e7e9f5'), margin=dict(l=10, r=10, t=60, b=10),
                coloraxis_showscale=False
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 ดูข้อมูล Input ที่ใช้ทำนาย"):
            input_df = pd.DataFrame({'Feature': feature_names, 'Value': input_data[0]})
            st.dataframe(input_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 ปรับค่าสเปกรถยนต์ด้านบนแล้วกดปุ่ม 'ทำนายราคารถยนต์' เพื่อดูผลลัพธ์")

# ==================== Footer ====================
st.markdown("---")
st.markdown("<p style='text-align:center;color:#8a90a8;margin-top:30px;'>Made with Streamlit · Machine Learning Projects (Automobile Dataset)</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#8a90a8;margin-top:4px;'>โดย นายจิรภัทร จันทร์มล · 664245026 · หมู่เรียน 66/43</p>", unsafe_allow_html=True)
