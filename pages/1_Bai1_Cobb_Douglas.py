import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title='Bài 1 — Cobb-Douglas + AI', layout='wide')
st.title('📈 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI')
st.markdown('Ước lượng TFP, phân rã tăng trưởng, dự báo GDP 2030.')

# Dữ liệu
df = pd.read_csv('Data/vietnam_macro_2020_2025.csv')
years = df['year'].values
Y     = df['GDP_trillion_VND'].values
D     = df['digital_economy_share_GDP_pct'].values
K     = np.array([16500,17800,19600,21300,23500,25900])
L     = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
AI    = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
H     = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07

# Tính TFP
mau  = (K**alpha)*(L**beta)*(D**gamma)*(AI**delta)*(H**theta)
A    = Y / mau
A_mean = A.mean()
Y_hat  = A_mean * mau
MAPE   = np.mean(np.abs((Y-Y_hat)/Y))*100

# KPI
col1, col2, col3 = st.columns(3)
col1.metric('MAPE (Cobb-Douglas)', f'{MAPE:.2f}%')
col2.metric('Ā (TFP trung bình)',  f'{A_mean:.4f}')

# Dự báo 2030
A_2030 = A[-1]*(1.012**5)
K_2030 = K[-1]*(1.06**5)
L_2030 = L[-1]*(1.06**5)
Y_2030 = A_2030*(K_2030**alpha)*(L_2030**beta)*\
         (30.0**gamma)*(100.0**delta)*(35.0**theta)
col3.metric('Y 2030 dự báo', f'{Y_2030:,.0f} ng.tỷ')

st.divider()

# Phân rã tăng trưởng
dY = np.diff(np.log(Y))
contribs = {
    'TFP (A)'     : np.diff(np.log(A)).mean()*100,
    'Vốn (K)'     : (alpha*np.diff(np.log(K))).mean()*100,
    'Lao động (L)': (beta *np.diff(np.log(L))).mean()*100,
    'Số hóa (D)'  : (gamma*np.diff(np.log(D))).mean()*100,
    'AI'          : (delta*np.diff(np.log(AI))).mean()*100,
    'Nhân lực (H)': (theta*np.diff(np.log(H))).mean()*100,
}

col_a, col_b = st.columns(2)

with col_a:
    st.subheader('TFP Việt Nam 2020–2025')
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(years, A, marker='o', color='steelblue', linewidth=2)
    for yr, a in zip(years, A):
        ax.annotate(f'{a:.2f}', xy=(yr,a),
                    xytext=(0,8), textcoords='offset points',
                    ha='center', fontsize=8)
    ax.set_xlabel('Năm'); ax.set_ylabel('TFP (A_t)')
    ax.set_xticks(years); ax.grid(linestyle='--', alpha=0.4)
    st.pyplot(fig)

with col_b:
    st.subheader('Phân rã đóng góp tăng trưởng')
    fig, ax = plt.subplots(figsize=(6,3))
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4']
    vals   = list(contribs.values())
    bars   = ax.bar(list(contribs.keys()), vals,
                    color=colors, edgecolor='white')

    # Nhãn giá trị trên/dưới mỗi cột
    for bar, val in zip(bars, vals):
        offset = 0.05 if val >= 0 else -0.15
        va     = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + offset,
                f'{val:.3f}%', ha='center', va=va,
                fontsize=8, fontweight='bold')

    ax.set_ylabel('Điểm phần trăm/năm')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Mở rộng trục y để hiện cả giá trị âm
    y_min = min(vals)
    y_max = max(vals)
    ax.set_ylim(y_min - 0.5, y_max + 0.8)

    plt.xticks(rotation=15, ha='right', fontsize=8)
    st.pyplot(fig)

st.divider()
st.subheader('GDP thực tế vs Dự báo')
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(years, Y,     'b-o', label='Thực tế', linewidth=2)
ax.plot(years, Y_hat, 'r--s', label=f'Dự báo (MAPE={MAPE:.2f}%)', linewidth=2)
ax.set_xlabel('Năm'); ax.set_ylabel('GDP (nghìn tỷ VND)')
ax.set_xticks(years); ax.legend(); ax.grid(linestyle='--', alpha=0.4)
st.pyplot(fig)

st.info(f'📌 **Kết luận:** TFP tăng đều từ {A[0]:.2f} lên {A[-1]:.2f} '
        f'(+{(A[-1]/A[0]-1)*100:.1f}% trong 5 năm). '
        f'GDP dự báo 2030: **{Y_2030:,.0f} nghìn tỷ VND** '
        f'(≈{Y_2030/23:.0f} tỷ USD).')

st.divider()
st.subheader('📌 Nhận xét')
top_factor = max(contribs, key=contribs.get)
st.markdown(f"""
- **TFP** tăng từ {A[0]:.2f} lên {A[-1]:.2f} (+{(A[-1]/A[0]-1)*100:.1f}% trong 5 năm) — động lực tăng trưởng lớn nhất ({contribs['TFP (A)']:.2f} điểm%/năm).
- **Lao động (L)** đóng góp âm ({contribs['Lao động (L)']:.3f}%) do COVID-19 làm lao động giảm mạnh năm 2021 — tăng trưởng GDP không nhờ tăng số lượng lao động mà nhờ năng suất.
- **Số hóa và AI** đóng góp {contribs['Số hóa (D)']:.2f} + {contribs['AI']:.2f} điểm%/năm — còn nhỏ nhưng đang nổi lên, phù hợp định hướng Nghị quyết 57-NQ/TW.
- **GDP 2030** dự báo đạt **{Y_2030:,.0f} nghìn tỷ VND** (≈{Y_2030/23:.0f} tỷ USD) nếu duy trì đà tăng trưởng TFP 1.2%/năm và đạt mục tiêu kinh tế số 30% GDP.
""")