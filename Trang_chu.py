import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title='AIDEOM-VN',
    page_icon='🇻🇳',
    layout='wide'
)

st.title('🇻🇳 AIDEOM-VN')
st.markdown('### *AI-Driven Decision Optimization Model for Vietnam*')
st.markdown('Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế '
            'Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020-2025.')

st.divider()

# KPI trang chủ
col1, col2, col3, col4 = st.columns(4)
col1.metric('GDP 2025',          '514.0 tỷ USD',  '+8.02%')
col2.metric('Kinh tế số / GDP',  '≈19.5%',        '+1.2 dpt')
col3.metric('FDI giải ngân 2025','27.6 tỷ USD',   '+8.9%')
col4.metric('GDP/người 2025',    '5.026 USD',      '+6.9%')

st.divider()

# Danh sách 12 bài
st.markdown('## 📋 12 bài toán theo 4 cấp độ')

with st.expander('🟢 Cấp độ DỄ — Làm quen mô hình', expanded=True):
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 1**')
    col2.markdown('Hàm sản xuất **Cobb-Douglas mở rộng + AI** — Growth accounting, dự báo GDP 2030')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 2**')
    col2.markdown('**LP phân bổ ngân sách** 4 hạng mục — scipy.optimize, shadow price')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 3**')
    col2.markdown('**Chỉ số ưu tiên 10 ngành** — Min-max norm, weighted scoring, sensitivity')

with st.expander('🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển'):
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 4**')
    col2.markdown('**LP ngành-vùng** — 24 biến, ràng buộc công bằng vùng miền')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 5**')
    col2.markdown('**MIP 15 dự án** — Binary variables, precedence, knapsack')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 6**')
    col2.markdown('**TOPSIS 6 vùng** — Entropy weights, ideal solution ranking')

with st.expander('🟠 Cấp độ KHÁ KHÓ — Tối ưu nâng cao'):
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 7**')
    col2.markdown('**NSGA-II Pareto** — 4 mục tiêu xung đột, Pareto frontier 3D')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 8**')
    col2.markdown('**Tối ưu động 2026-2035** — SLSQP, Cobb-Douglas, cú sốc 2028')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 9**')
    col2.markdown('**Lao động & AI** — NetJob, ngưỡng đào tạo, dịch chuyển lao động')

with st.expander('🔴 Cấp độ KHÓ — AI & Bất định'):
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 10**')
    col2.markdown('**Stochastic LP** — Two-stage, VSS, EVPI, Robust optimization')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 11**')
    col2.markdown('**Q-learning RL** — MDP, epsilon-greedy, DQN so sánh')
    col1, col2 = st.columns([1, 3])
    col1.markdown('**Bài 12**')
    col2.markdown('**AIDEOM-VN tích hợp** — 6 module, 5 kịch bản, dashboard')

st.divider()
st.caption('📊 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025')