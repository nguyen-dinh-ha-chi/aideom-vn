import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

st.set_page_config(page_title='Bài 5 — MIP 15 dự án', layout='wide')
st.title('📋 Bài 5 — MIP Lựa chọn dự án chuyển đổi số')
st.markdown('15 dự án, biến nhị phân, ràng buộc tiên quyết và loại trừ.')

names = {1:'TTDL Hòa Lạc',2:'TTDL phía Nam',3:'5G toàn quốc',
         4:'VNeID 2.0',5:'Cổng DVC v3',6:'Y tế số',
         7:'Giáo dục số K-12',8:'AI quốc gia',9:'Sandbox fintech',
         10:'Logistics thông minh',11:'Nông nghiệp số ĐBSCL',
         12:'Đào tạo 50k kỹ sư AI',13:'Khu CN bán dẫn',
         14:'An ninh mạng SOC',15:'Open Data quốc gia'}
C  = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,
      8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,
      8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
B  = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,
      8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}

budget = st.slider('Ngân sách tổng (tỷ VND)', 60000, 120000, 80000, 10000)

if st.button('▶ Chạy tối ưu MIP', type='primary'):
    projects = list(range(1, 16))
    m = pulp.LpProblem('MIP_B5', pulp.LpMaximize)
    y = pulp.LpVariable.dicts('y', projects, cat='Binary')

    m += pulp.lpSum(B[i]*y[i] for i in projects)
    m += pulp.lpSum(C[i]*y[i] for i in projects)  <= budget
    m += pulp.lpSum(C1[i]*y[i] for i in projects) <= budget*0.5
    m += y[1]+y[2] <= 1
    m += y[8]  <= y[12]
    m += y[13] <= y[12]
    m += y[4]+y[5] >= 1
    m += y[14] >= 1
    m += pulp.lpSum(y[i] for i in projects) >= 7
    m += pulp.lpSum(y[i] for i in projects) <= 11
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    selected = [i for i in projects if y[i].value() > 0.5]
    Z        = pulp.value(m.objective)
    tot_cost = sum(C[i] for i in selected)

    col1, col2, col3 = st.columns(3)
    col1.metric('Z* NPV',         f'{Z/1000:.1f} nghìn tỷ')
    col2.metric('Số dự án chọn',  f'{len(selected)}/15')
    col3.metric('Tổng chi phí',   f'{tot_cost/1000:.1f} nghìn tỷ')

    st.divider()
    st.subheader('Dự án được chọn')
    rows = []
    for i in selected:
        rows.append({'Mã':f'P{i}', 'Tên':names[i],
                     'Chi phí (tỷ)':C[i], 'NPV (tỷ)':B[i],
                     'ROI':round(B[i]/C[i],2)})
    df_sel = pd.DataFrame(rows)
    st.dataframe(df_sel.style.background_gradient(
                 subset=['ROI'], cmap='Greens'),
                 use_container_width=True)

    st.subheader('ROI từng dự án được chọn')
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(df_sel['Tên'], df_sel['ROI'],
           color='#4CAF50', edgecolor='white', alpha=0.85)
    ax.axhline(1.0, color='red', linestyle='--', label='ROI=1')
    ax.set_xticklabels(df_sel['Tên'], rotation=20, ha='right')
    ax.set_ylabel('NPV / Chi phí')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)

    st.divider()
    st.subheader('📌 Nhận xét')
    avg_roi = sum(B[i] for i in selected) / sum(C[i] for i in selected)
    not_selected = [i for i in range(1,16) if i not in selected]
    st.markdown(f"""
    - **{len(selected)} dự án được chọn** với tổng NPV = {Z/1000:.1f} nghìn tỷ, chi phí = {tot_cost/1000:.1f} nghìn tỷ (ROI = {avg_roi:.2f}).
    - **Dự án bắt buộc:** P14 (An ninh mạng SOC) luôn được chọn theo ràng buộc y₁₄ ≥ 1.
    - **Tiên quyết:** P8 (AI quốc gia) và P13 (Bán dẫn) chỉ được chọn nếu P12 (Đào tạo kỹ sư) được chọn — phản ánh phụ thuộc nhân lực.
    - **Dự án không được chọn:** {', '.join([f'P{i}' for i in not_selected[:5]])}{'...' if len(not_selected)>5 else ''} — do ràng buộc ngân sách hoặc tiên quyết.
    - **Hàm ý:** Tăng ngân sách lên 100,000 tỷ sẽ cho phép chọn thêm dự án hạ tầng lớn (P1/P2/P3).
    """)
else:
    st.info('Nhấn **▶ Chạy tối ưu MIP** để xem kết quả.')