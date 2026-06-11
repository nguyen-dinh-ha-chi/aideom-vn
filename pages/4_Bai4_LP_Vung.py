import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pulp

st.set_page_config(page_title='Bài 4 — LP Ngành-Vùng', layout='wide')
st.title('🗺️ Bài 4 — LP Phân bổ ngân sách số theo ngành-vùng')
st.markdown('24 biến quyết định, ràng buộc công bằng vùng miền.')

regions      = ['NMM','RRD','NCC','CH','SE','MD']
region_names = ['Trung du MN phía Bắc','Đồng bằng sông Hồng',
                'Bắc TB + DH Trung Bộ','Tây Nguyên',
                'Đông Nam Bộ','Đồng bằng sông CL']
items        = ['I','D','AI','H']
item_names   = ['Hạ tầng số','CĐS DN','AI','Nhân lực']
beta_mat     = {
    ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
    ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
    ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
    ('CH', 'I'):1.20,('CH', 'D'):0.75,('CH', 'AI'):0.45,('CH', 'H'):1.35,
    ('SE', 'I'):0.90,('SE', 'D'):1.30,('SE', 'AI'):1.55,('SE', 'H'):1.00,
    ('MD', 'I'):1.10,('MD', 'D'):0.85,('MD', 'AI'):0.65,('MD', 'H'):1.25,
}

df_r  = pd.read_csv('Data/vietnam_regions_2024.csv')
D0    = dict(zip(regions, df_r['digital_index_0_100'].values))

budget   = st.slider('Ngân sách tổng (tỷ VND)', 30000, 70000, 50000, 5000)
equity   = st.checkbox('Có ràng buộc công bằng vùng (C5)', value=True)

if st.button('▶ Chạy tối ưu', type='primary'):
    m = pulp.LpProblem('LP_B4', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)
    m += pulp.lpSum(beta_mat[(r,j)]*x[r][j]
                    for r in regions for j in items)
    m += pulp.lpSum(x[r][j]
                    for r in regions for j in items) <= budget
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in items) >= 5000
        m += pulp.lpSum(x[r][j] for j in items) <= 12000
    m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000

    if equity:
        M_var = pulp.LpVariable('Dmax', lowBound=0)
        for r in regions:
            m += D0[r] + 0.002*x[r]['D'] <= M_var
            m += D0[r] + 0.002*x[r]['D'] >= 0.7*M_var

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    Z = pulp.value(m.objective)
    result = np.array([[x[r][j].value() for j in items]
                       for r in regions])
    df_res = pd.DataFrame(result, index=region_names, columns=item_names)
    df_res['TỔNG'] = df_res.sum(axis=1)

    st.metric('Z* GDP Gain', f'{Z:,.2f} tỷ VND')
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('Phân bổ tối ưu (tỷ VND)')
        st.dataframe(df_res.style.format('{:.1f}')
                     .background_gradient(cmap='YlOrRd'),
                     use_container_width=True)
    with col_b:
        st.subheader('Heatmap phân bổ')
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_res.iloc[:,:4], annot=True, fmt='.0f',
                    cmap='YlOrRd', ax=ax, linewidths=0.5)
        st.pyplot(fig)

    st.subheader('Tổng ngân sách mỗi vùng')
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    df_res['TỔNG'].plot(kind='barh', ax=ax2,
                        color='#2196F3', alpha=0.85)
    ax2.axvline(5000,  color='green', linestyle='--', label='Sàn 5,000')
    ax2.axvline(12000, color='red',   linestyle='--', label='Trần 12,000')
    ax2.legend()
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    st.pyplot(fig2)

    st.divider()
    st.subheader('📌 Nhận xét')
    
    max_val = df_res['TỔNG'].max()
    min_val = df_res['TỔNG'].min()
    top_vungs = df_res[df_res['TỔNG'] == max_val].index.tolist()
    bot_vungs = df_res[df_res['TỔNG'] == min_val].index.tolist()
    top_item  = df_res.iloc[:,:4].sum().idxmax()

    st.markdown(f"""
    - **Vùng nhận nhiều nhất ({max_val:,.0f} tỷ):** {', '.join(top_vungs)} — do hệ số tác động AI/D cao nhất.
    - **Vùng nhận ít nhất ({min_val:,.0f} tỷ):** {', '.join(bot_vungs)} — ở mức sàn tối thiểu 5,000 tỷ.
    - **Hạng mục ưu tiên nhất:** {top_item} — chiếm tỷ trọng lớn nhất trong tổng phân bổ.
    - {'**Ràng buộc công bằng C5** (λ=0.6) đang hoạt động — buộc các vùng yếu đạt ít nhất 60% mức số hóa của vùng dẫn đầu.' if equity else '**Không có ràng buộc công bằng** — ngân sách tập trung vào vùng có hệ số cao nhất.'}
    - **Z* = {Z:,.2f} tỷ VND** — GDP gain kỳ vọng từ toàn bộ ngân sách phân bổ.
        """)
else:
    st.info('Nhấn **▶ Chạy tối ưu** để xem kết quả.')