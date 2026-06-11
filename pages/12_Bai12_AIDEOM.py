import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title='Bài 12 — AIDEOM-VN', layout='wide')
st.title('🇻🇳 Bài 12 — AIDEOM-VN Tích hợp')
st.markdown('So sánh 5 kịch bản chính sách phát triển kinh tế số Việt Nam 2026–2030.')

# Load KPI
try:
    df_kpi = pd.read_csv('Data/aideom_kpi_summary.csv', index_col=0)
    st.success('✅ Đã tải KPI từ aideom_kpi_summary.csv')
except:
    st.error('❌ Chưa có file KPI — hãy chạy cell bài 12 trong notebook trước')
    st.stop()

st.divider()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    '📊 Tổng quan (M1-M2)',
    '🗺️ Phân bổ (M3)',
    '📋 5 Kịch bản (M6)',
    '⚠️ Cảnh báo rủi ro (M4-M5)'
])

with tab1:
    st.header('M1 — Dự báo kinh tế')
    gdp_2025 = float(df_kpi.iloc[0]['GDP_2025 (nghìn tỷ)'])
    gdp_2030 = float(df_kpi.iloc[0]['GDP_2030 dự báo'])
    mape_val = float(df_kpi.iloc[0]['MAPE (%)'])
    tfp_val  = float(df_kpi.iloc[0]['TFP_2025'])

    col1,col2,col3,col4 = st.columns(4)
    col1.metric('GDP 2025', f'{gdp_2025:,.1f} ng.tỷ')
    col2.metric('GDP 2030 dự báo', f'{gdp_2030:,.1f} ng.tỷ',
                f'+{(gdp_2030/gdp_2025-1)*100:.1f}%')
    col3.metric('TFP 2025', f'{tfp_val:.4f}')
    col4.metric('MAPE dự báo', f'{mape_val:.2f}%')

    st.divider()
    st.header('M2 — Sẵn sàng số')
    top_vung = df_kpi.iloc[0]['Top vùng AI']
    st.info(f'🏆 Vùng ưu tiên triển khai AI đầu tiên: **{top_vung}**')
    # Dữ liệu 6 vùng với tọa độ trung tâm
    # Tính lại TOPSIS ngay trong file này
    df_r = pd.read_csv('Data/vietnam_regions_2024.csv')
    criteria_m2   = ['grdp_per_capita_million_VND','fdi_registered_billion_USD',
                    'digital_index_0_100','ai_readiness_0_100',
                    'trained_labor_pct','rd_intensity_pct',
                    'internet_penetration_pct','gini_coef']
    is_benefit_m2 = [True,True,True,True,True,True,True,False]
    w_exp         = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    X_m2          = df_r[criteria_m2].values.astype(float)

    # Hàm TOPSIS
    def topsis_local(X, weights, is_benefit):
        norm   = np.sqrt((X**2).sum(axis=0))
        R      = X / norm
        V      = R * weights
        A_star = np.array([V[:,j].max() if is_benefit[j]
                        else V[:,j].min() for j in range(V.shape[1])])
        A_neg  = np.array([V[:,j].min() if is_benefit[j]
                        else V[:,j].max() for j in range(V.shape[1])])
        S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
        S_neg  = np.sqrt(((V - A_neg )**2).sum(axis=1))
        return S_neg / (S_star + S_neg)

    topsis_scores = topsis_local(X_m2, w_exp, is_benefit_m2)

    # Dùng scores trong df_map
    df_map = pd.DataFrame({
        'Vùng'         : ['Trung du MN phía Bắc','Đồng bằng sông Hồng',
                        'Bắc TB + DH Trung Bộ','Tây Nguyên',
                        'Đông Nam Bộ','Đồng bằng sông CL'],
        'lat'          : [21.8, 21.0, 16.5, 13.5, 11.0, 10.0],
        'lon'          : [104.5,106.0,107.5,108.0,107.0,105.5],
        'Digital_Index': [38,   78,   55,   32,   82,   48],
        'AI_Readiness' : [22,   68,   40,   18,   75,   30],
        'TOPSIS_Score' : topsis_scores.tolist()
    })

    st.subheader('🗺️ Bản đồ Digital Index theo vùng')

    col_a, col_b = st.columns(2)

    with col_a:
        fig_map1 = px.scatter_geo(
            df_map,
            lat='lat', lon='lon',
            size='Digital_Index',
            color='Digital_Index',
            hover_name='Vùng',
            hover_data={'Digital_Index': True, 'lat': False, 'lon': False},
            color_continuous_scale='Blues',
            size_max=40,
            title='Digital Index theo vùng',
            scope='asia',
        )
        fig_map1.update_geos(
            center=dict(lat=16.0, lon=106.0),
            projection_scale=8,
            showland=True, landcolor='lightgray',
            showocean=True, oceancolor='lightblue',
            showcoastlines=True
        )
        fig_map1.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_map1, use_container_width=True)

    with col_b:
        fig_map2 = px.scatter_geo(
            df_map,
            lat='lat', lon='lon',
            size='AI_Readiness',
            color='AI_Readiness',
            hover_name='Vùng',
            hover_data={'AI_Readiness': True, 'lat': False, 'lon': False},
            color_continuous_scale='Reds',
            size_max=40,
            title='AI Readiness theo vùng',
            scope='asia',
        )
        fig_map2.update_geos(
            center=dict(lat=16.0, lon=106.0),
            projection_scale=8,
            showland=True, landcolor='lightgray',
            showocean=True, oceancolor='lightblue',
            showcoastlines=True
        )
        fig_map2.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_map2, use_container_width=True)

    # Bảng xếp hạng TOPSIS kèm bản đồ
    st.subheader('📊 Xếp hạng TOPSIS + Entropy Weight')
    df_map['Xếp hạng'] = np.argsort(np.argsort(-df_map['TOPSIS_Score'])) + 1
    df_map_show = df_map[['Vùng','Digital_Index','AI_Readiness',
                        'TOPSIS_Score','Xếp hạng']].sort_values('Xếp hạng')
    st.dataframe(
        df_map_show.style
                .background_gradient(subset=['Digital_Index'], cmap='Blues')
                .background_gradient(subset=['AI_Readiness'],  cmap='Reds')
                .format({'TOPSIS_Score': '{:.4f}'}),
        use_container_width=True
    )

    top1 = df_map_show.iloc[0]['Vùng']
    st.success(f'🏆 Vùng ưu tiên triển khai AI đầu tiên: **{top1}**')

with tab2:
    st.header('M3 — Phân bổ ngân sách theo vùng')
    sc_sel = st.selectbox('Chọn kịch bản:', df_kpi.index)
    z_val  = float(df_kpi.loc[sc_sel, 'Z* GDP gain (tỷ)'])
    st.metric(f'Z* GDP Gain — {sc_sel}', f'{z_val:,.1f} tỷ VND')

    labels_alloc = ['Hạ tầng số', 'AI & Dữ liệu', 'Nhân lực số', 'R&D']
    SCENARIOS_SHARE = {
        'S1 Truyền thống' : [0.70, 0.10, 0.10, 0.10],
        'S2 Số hóa nhanh' : [0.25, 0.45, 0.15, 0.15],
        'S3 AI dẫn dắt'   : [0.20, 0.20, 0.45, 0.15],
        'S4 Bao trùm số'  : [0.30, 0.20, 0.10, 0.40],
        'S5 Tối ưu cân bằng': [0.25, 0.25, 0.25, 0.25],
    }
    shares = SCENARIOS_SHARE.get(sc_sel, [0.25]*4)
    fig, ax = plt.subplots(figsize=(7,3))
    colors  = ['#2196F3','#FF9800','#9C27B0','#4CAF50']
    ax.bar(labels_alloc, [s*100 for s in shares],
           color=colors, edgecolor='white')
    ax.set_ylabel('Tỷ trọng (%)')
    ax.set_title(f'Cơ cấu phân bổ — {sc_sel}', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)

with tab3:
    st.header('So sánh 5 kịch bản chính sách')
    st.dataframe(
        df_kpi.style
              .highlight_max(axis=0, color='#C8E6C9')
              .highlight_min(axis=0, color='#FFCDD2')
              .format('{:.1f}', subset=df_kpi.select_dtypes('number').columns),
        use_container_width=True
    )
    st.caption('🟢 Cao nhất  🔴 Thấp nhất trong từng chỉ tiêu')

    col_sel = st.selectbox('Chọn chỉ tiêu để so sánh:',
                           df_kpi.select_dtypes('number').columns)
    fig, ax = plt.subplots(figsize=(9,4))
    colors5 = ['#2196F3','#FF9800','#9C27B0','#4CAF50','#F44336']
    bars = ax.bar(df_kpi.index, df_kpi[col_sel],
                  color=colors5, edgecolor='white', alpha=0.85)
    for bar, v in zip(bars, df_kpi[col_sel]):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()*1.01,
                f'{v:.1f}', ha='center', fontsize=9)
    ax.set_title(col_sel, fontweight='bold')
    ax.set_xticklabels(df_kpi.index, rotation=15, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)

    csv = df_kpi.to_csv(encoding='utf-8-sig')
    st.download_button('⬇️ Tải bảng KPI (CSV)',
                       data=csv,
                       file_name='aideom_kpi.csv',
                       mime='text/csv')

with tab4:
    st.header('⚠️ Cảnh báo rủi ro (M4-M5)')

    col1, col2, col3 = st.columns(3)

    # KPI cards
    best_env  = df_kpi['Rủi ro MT'].idxmin()
    best_job  = df_kpi['Tổng NetJob (nghìn)'].idxmax()
    best_comp = df_kpi['Điểm thỏa hiệp'].idxmax()

    col1.metric('Ít rủi ro MT nhất',     best_env)
    col2.metric('Nhiều việc làm nhất',   best_job)
    col3.metric('Thỏa hiệp tốt nhất',    best_comp)

    st.divider()

    # Biểu đồ 4 mục tiêu (bài 7)
    st.subheader('📊 4 Mục tiêu Pareto theo kịch bản (Bài 7)')
    col_a, col_b = st.columns(2)

    with col_a:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        colors5   = ['#2196F3','#FF9800','#9C27B0','#4CAF50','#F44336']
        ax1.bar(df_kpi.index, df_kpi['Rủi ro MT'],
                color=colors5, alpha=0.85, edgecolor='white')
        ax1.set_title('f3 — Rủi ro môi trường', fontweight='bold')
        ax1.set_xticklabels(df_kpi.index, rotation=15, ha='right')
        ax1.set_ylabel('Chỉ số phát thải')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(df_kpi.index, df_kpi['Rủi ro AN'],
                color=colors5, alpha=0.85, edgecolor='white')
        ax2.set_title('f4 — Rủi ro an ninh dữ liệu', fontweight='bold')
        ax2.set_xticklabels(df_kpi.index, rotation=15, ha='right')
        ax2.set_ylabel('Chỉ số rủi ro')
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig2)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.bar(df_kpi.index, df_kpi['Bất bình đẳng'],
                color=colors5, alpha=0.85, edgecolor='white')
        ax3.set_title('f2 — Bất bình đẳng vùng', fontweight='bold')
        ax3.set_xticklabels(df_kpi.index, rotation=15, ha='right')
        ax3.set_ylabel('MAD ngân sách vùng')
        ax3.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig3)

    with col_d:
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.bar(df_kpi.index, df_kpi['Điểm thỏa hiệp'],
                color=colors5, alpha=0.85, edgecolor='white')
        ax4.set_title('Điểm thỏa hiệp Pareto', fontweight='bold')
        ax4.set_xticklabels(df_kpi.index, rotation=15, ha='right')
        ax4.set_ylabel('Compromise score')
        ax4.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig4)

    st.divider()

    # Stochastic LP (bài 10)
    st.subheader('🎲 Bất định kinh tế toàn cầu (Bài 10)')
    col_e, col_f = st.columns(2)

    with col_e:
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        ax5.bar(df_kpi.index, df_kpi['Z* SP (tỷ)'],
                color=colors5, alpha=0.85, edgecolor='white')
        ax5.set_title('Z* Stochastic theo kịch bản', fontweight='bold')
        ax5.set_xticklabels(df_kpi.index, rotation=15, ha='right')
        ax5.set_ylabel('Tỷ VND')
        ax5.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig5)

    with col_f:
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        ax6.bar(df_kpi.index, df_kpi['EVPI (tỷ)'],
                color=colors5, alpha=0.85, edgecolor='white')
        ax6.set_title('EVPI theo kịch bản', fontweight='bold')
        ax6.set_xticklabels(df_kpi.index, rotation=15, ha='right')
        ax6.set_ylabel('Tỷ VND')
        ax6.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig6)

    evpi_best = df_kpi['EVPI (tỷ)'].idxmin()
    st.info(f'💡 **{evpi_best}** có EVPI thấp nhất — '
            f'ít phụ thuộc vào thông tin tương lai nhất, '
            f'phù hợp khi môi trường kinh tế bất định.')
    
    st.divider()

    # Khuyến nghị chính sách
    st.subheader('📌 Khuyến nghị chính sách')
    st.markdown(f'''
    | Tiêu chí | Kịch bản tốt nhất |
    |----------|-------------------|
    | Tối đa GDP gain | **{df_kpi["Z* GDP gain (tỷ)"].idxmax()}** |
    | Tối đa việc làm | **{df_kpi["Tổng NetJob (nghìn)"].idxmax()}** |
    | Tối thiểu rủi ro MT | **{df_kpi["Rủi ro MT"].idxmin()}** |
    | Tối thiểu dịch chuyển LĐ | **{df_kpi["Dịch chuyển (nghìn)"].idxmin()}** |
    | Điểm thỏa hiệp cao nhất | **{df_kpi["Điểm thỏa hiệp"].idxmax()}** |
    ''')