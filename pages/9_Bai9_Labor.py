import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

st.set_page_config(page_title='Bài 9 - Lao động & AI', layout='wide')
st.title('👷 Bài 9 - Tác động AI tới thị trường lao động')
st.markdown('NetJob, ngưỡng đào tạo tối thiểu, dịch chuyển lao động.')

sector_names=['Nông-Lâm-Thủy sản','CN chế biến chế tạo','Xây dựng',
              'Bán buôn-bán lẻ','Tài chính-Ngân hàng','Logistics-Vận tải',
              'CNTT-Truyền thông','Giáo dục-Đào tạo']
N=8
L   =np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
risk=np.array([18,42,25,38,52,35,28,22])/100
a1  =np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1  =np.array([45.,28.,35.,32.,22.,30.,20.,55.])
c1  =np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
d1  =np.array([50.,32.,42.,38.,26.,36.,24.,62.])
nc  =a1-c1*risk

budget=st.slider('Ngân sách (tỷ VND)',10000,50000,30000,5000)

if st.button('▶ Chạy tối ưu lao động', type='primary'):
    m=pulp.LpProblem('Labor_B9',pulp.LpMaximize)
    xA=pulp.LpVariable.dicts('xAI',range(N),lowBound=0)
    xH=pulp.LpVariable.dicts('xH', range(N),lowBound=0)
    # Ràng buộc Min/Max mỗi ngày để kết quả có ý nghĩa kinh tế hơn
    min_per_sector = 500
    max_per_sector = 12000
    for i in range(N):
        m += xA[i] + xH[i] >= min_per_sector, f'Min_{i}'
        m += xA[i] + xH[i] <= max_per_sector, f'Max_{i}'
    m+=pulp.lpSum(nc[i]*xA[i]+b1[i]*xH[i] for i in range(N))
    m+=pulp.lpSum(xA[i]+xH[i] for i in range(N))<=budget
    for i in range(N):
        m+=nc[i]*xA[i]+b1[i]*xH[i]>=0
        m+=d1[i]*xH[i]>=c1[i]*risk[i]*xA[i]
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    xA_o=np.array([xA[i].value() or 0 for i in range(N)])
    xH_o=np.array([xH[i].value() or 0 for i in range(N)])
    nj  =a1*xA_o+b1*xH_o-c1*risk*xA_o
    disp=c1*risk*xA_o
    new =a1*xA_o

    col1,col2,col3=st.columns(3)
    col1.metric('Tổng NetJob',    f'{nj.sum():,.0f} việc làm')
    col2.metric('Việc làm mới',   f'{new.sum():,.0f} việc làm')
    col3.metric('Bị dịch chuyển', f'{disp.sum():,.0f} việc làm',
                delta=f'-{disp.sum():,.0f}', delta_color='inverse')

    st.divider()
    col_a,col_b=st.columns(2)
    with col_a:
        st.subheader('NetJob ròng từng ngành')
        fig,ax=plt.subplots(figsize=(6,4))
        colors=['#4CAF50' if v>=0 else '#F44336' for v in nj]
        ax.barh(sector_names,nj,color=colors,edgecolor='white')
        ax.axvline(0,color='black',linewidth=0.8)
        ax.set_xlabel('Nghìn việc làm')
        ax.grid(axis='x',linestyle='--',alpha=0.4)
        st.pyplot(fig)

    with col_b:
        st.subheader('Phân bổ x_AI vs x_H')
        fig2,ax2=plt.subplots(figsize=(6,4))
        x_pos=np.arange(N); w=0.35
        ax2.bar(x_pos-w/2,xA_o,w,label='x_AI',
                color='#2196F3',alpha=0.85)
        ax2.bar(x_pos+w/2,xH_o,w,label='x_H',
                color='#FF9800',alpha=0.85)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s[:8] for s in sector_names],
                             rotation=20,ha='right')
        ax2.legend()
        ax2.grid(axis='y',linestyle='--',alpha=0.4)
        st.pyplot(fig2)

    st.divider()
    st.subheader('Bảng chi tiết')
    df_res=pd.DataFrame({
        'Ngành'    :sector_names,
        'x_AI'     :xA_o.round(1),
        'x_H'      :xH_o.round(1),
        'Việc làm mới'   :new.round(0),
        'Nâng cấp kỹ năng'  :(b1*xH_o).round(0),
        'Bị dịch chuyển':disp.round(0),
        'NetJob'   :nj.round(0),
    })
    st.dataframe(df_res.style.background_gradient(
                 subset=['NetJob'],cmap='RdYlGn'),
                 use_container_width=True)
    
    st.subheader('📌 Nhận xét')
    top_ai_idx  = np.argmax(xA_o)
    top_h_idx   = np.argmax(xH_o)
    top_nj_idx  = np.argmax(nj)

    st.markdown(f"""
    - Ngân sách **{budget:,.0f} tỷ VND** tạo ra tổng **{nj.sum():,.0f} việc làm**.
    - Ngành nhận đầu tư AI nhiều nhất: **{sector_names[top_ai_idx]}** ({xA_o[top_ai_idx]:,.0f} tỷ)
    - Ngành nhận đầu tư đào tạo nhiều nhất: **{sector_names[top_h_idx]}** ({xH_o[top_h_idx]:,.0f} tỷ)
    - Ngành tạo NetJob nhiều nhất: **{sector_names[top_nj_idx]}** ({nj[top_nj_idx]:,.0f} việc)
    - Số ngành có x_AI > 0: **{(xA_o > 0).sum()}** / 8 ngành
    """)
else:
    st.info('Nhấn **▶ Chạy tối ưu lao động** để xem kết quả.')