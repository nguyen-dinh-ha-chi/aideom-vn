import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title='Bài 6 — TOPSIS 6 vùng', layout='wide')
st.title('🏆 Bài 6 — TOPSIS Xếp hạng 6 vùng theo AI Readiness')
st.markdown('TOPSIS + Entropy weights, bản đồ Digital Index và AI Readiness.')

df_r = pd.read_csv('vietnam_regions_2024.csv')
criteria     = ['grdp_per_capita_million_VND','fdi_registered_billion_USD',
                'digital_index_0_100','ai_readiness_0_100',
                'trained_labor_pct','rd_intensity_pct',
                'internet_penetration_pct','gini_coef']
is_benefit   = [True,True,True,True,True,True,True,False]
w_expert     = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
rnames       = df_r['region_name_en'].values
X            = df_r[criteria].values.astype(float)

def topsis_fn(X, w, ib):
    R      = X / np.sqrt((X**2).sum(axis=0))
    V      = R * w
    A_star = np.array([V[:,j].max() if ib[j] else V[:,j].min()
                       for j in range(V.shape[1])])
    A_neg  = np.array([V[:,j].min() if ib[j] else V[:,j].max()
                       for j in range(V.shape[1])])
    S_star = np.sqrt(((V-A_star)**2).sum(axis=1))
    S_neg  = np.sqrt(((V-A_neg )**2).sum(axis=1))
    return S_neg/(S_star+S_neg)

def entropy_w(X):
    P = X/X.sum(axis=0)
    k = 1/np.log(len(X))
    E = -k*np.nansum(P*np.log(P+1e-12), axis=0)
    d = 1-E
    return d/d.sum()

C_exp = topsis_fn(X, w_expert, is_benefit)
C_ent = topsis_fn(X, entropy_w(X), is_benefit)
ranks = np.argsort(np.argsort(-C_exp))+1
top3  = rnames[np.argsort(C_exp)[::-1][:3]]

col1, col2, col3 = st.columns(3)
col1.metric('🥇 #1', top3[0])
col2.metric('🥈 #2', top3[1])
col3.metric('🥉 #3', top3[2])

st.divider()

# Bản đồ
st.subheader('🗺️ Bản đồ Digital Index + AI Readiness')
df_map = pd.DataFrame({
    'Vùng'         : rnames,
    'lat'          : [21.8,21.0,16.5,13.5,11.0,10.0],
    'lon'          : [104.5,106.0,107.5,108.0,107.0,105.5],
    'Digital_Index': df_r['digital_index_0_100'].values,
    'AI_Readiness' : df_r['ai_readiness_0_100'].values,
    'TOPSIS_Score' : C_exp,
})

col_a, col_b = st.columns(2)
with col_a:
    fig1 = px.scatter_geo(df_map, lat='lat', lon='lon',
                          size='Digital_Index', color='Digital_Index',
                          hover_name='Vùng', color_continuous_scale='Blues',
                          size_max=40, title='Digital Index', scope='asia')
    fig1.update_geos(center=dict(lat=16.0,lon=106.0), projection_scale=8,
                     showland=True, landcolor='lightgray',
                     showocean=True, oceancolor='lightblue')
    fig1.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    fig2 = px.scatter_geo(df_map, lat='lat', lon='lon',
                          size='AI_Readiness', color='AI_Readiness',
                          hover_name='Vùng', color_continuous_scale='Reds',
                          size_max=40, title='AI Readiness', scope='asia')
    fig2.update_geos(center=dict(lat=16.0,lon=106.0), projection_scale=8,
                     showland=True, landcolor='lightgray',
                     showocean=True, oceancolor='lightblue')
    fig2.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
col_c, col_d = st.columns(2)
with col_c:
    st.subheader('So sánh TOPSIS: Chuyên gia vs Entropy')
    fig3, ax3 = plt.subplots(figsize=(6,4))
    x_pos = np.arange(len(rnames))
    ax3.barh(x_pos-0.2, C_exp[np.argsort(C_exp)],
             0.35, label='Chuyên gia', color='#2196F3', alpha=0.85)
    ax3.barh(x_pos+0.2, C_ent[np.argsort(C_exp)],
             0.35, label='Entropy',    color='#FF9800', alpha=0.85)
    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(rnames[np.argsort(C_exp)])
    ax3.set_xlabel('TOPSIS Score')
    ax3.legend()
    ax3.grid(axis='x', linestyle='--', alpha=0.4)
    st.pyplot(fig3)

with col_d:
    st.subheader('Bảng xếp hạng')
    df_rank = pd.DataFrame({
        'Vùng'           : rnames,
        'Score CG'        : C_exp.round(4),
        'Score Entropy'   : C_ent.round(4),
        'Hạng CG'         : ranks,
        'Hạng Entropy'    : np.argsort(np.argsort(-C_ent))+1,
    }).sort_values('Hạng CG')
    st.dataframe(df_rank.style.background_gradient(
                 subset=['Score CG'], cmap='Greens'),
                 use_container_width=True)
    
st.divider()
st.subheader('📌 Nhận xét')
rank_change = np.abs(np.argsort(np.argsort(-C_exp)) - np.argsort(np.argsort(-C_ent)))
most_changed = rnames[np.argmax(rank_change)]
st.markdown(f"""
- **Top 3 (trọng số chuyên gia):** {', '.join(top3)} — Đông Nam Bộ và Đồng bằng sông Hồng dẫn đầu nhờ vượt trội về GRDP, FDI và AI Readiness.
- **Vùng thay đổi xếp hạng nhiều nhất khi dùng Entropy:** {most_changed} — khi trọng số khách quan, các tiêu chí có độ biến động cao được ưu tiên hơn.
- **Khoảng cách lớn** giữa top 2 và các vùng còn lại phản ánh sự phân hóa số rõ rệt giữa các vùng kinh tế Việt Nam.
- **Khuyến nghị:** Xây dựng 3 trung tâm AI ưu tiên tại {top3[0]}, {top3[1]}, và {top3[2]} — phù hợp mục tiêu Quyết định 127/QĐ-TTg.
""")