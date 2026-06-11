import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Bài 3 — Priority 10 ngành', layout='wide')
st.title('🏭 Bài 3 — Chỉ số ưu tiên ngành Việt Nam 2024')
st.markdown('Chuẩn hóa min-max, tính Priority score, phân tích độ nhạy trọng số.')

df = pd.read_csv('Data/vietnam_sectors_2024.csv')

cols_good = ['growth_rate_2024_pct','spillover_coef_0_1',
             'export_billion_USD','labor_million','ai_readiness_0_100']
col_bad   = 'automation_risk_pct'

def norm_good(x): return (x - x.min()) / (x.max() - x.min())
def norm_bad(x):  return (x.max() - x) / (x.max() - x.min())

Xg = df[cols_good].apply(norm_good)
Xb = norm_bad(df[col_bad])

# Sidebar trọng số
st.sidebar.header('⚙️ Điều chỉnh trọng số')
w_growth    = st.sidebar.slider('Tăng trưởng',  0.05, 0.40, 0.20, 0.05)
w_spillover = st.sidebar.slider('Lan tỏa',       0.05, 0.40, 0.20, 0.05)
w_export    = st.sidebar.slider('Xuất khẩu',     0.05, 0.40, 0.15, 0.05)
w_labor     = st.sidebar.slider('Việc làm',      0.05, 0.40, 0.10, 0.05)
w_ai        = st.sidebar.slider('AI Readiness',  0.05, 0.40, 0.20, 0.05)
w_risk      = st.sidebar.slider('Rủi ro TĐH',   0.05, 0.40, 0.15, 0.05)

total_w = w_growth+w_spillover+w_export+w_labor+w_ai+w_risk
if abs(total_w - 1.0) > 0.01:
    st.sidebar.warning(f'⚠️ Tổng trọng số = {total_w:.2f} (nên = 1.00)')

w = np.array([w_growth, w_spillover, w_export, w_labor, w_ai])
priority = Xg.values @ w - w_risk * Xb.values
df['Priority'] = priority

df_ranked = df[['sector_name_en','Priority']].sort_values(
    'Priority', ascending=False).reset_index(drop=True)
df_ranked.index += 1

col1, col2, col3 = st.columns(3)
col1.metric('🥇 Ngành ưu tiên #1', df_ranked.iloc[0]['sector_name_en'])
col2.metric('🥈 Ngành ưu tiên #2', df_ranked.iloc[1]['sector_name_en'])
col3.metric('🥉 Ngành ưu tiên #3', df_ranked.iloc[2]['sector_name_en'])

st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader('Xếp hạng Priority Score')
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#FF5722' if i < 3 else '#2196F3'
              for i in range(len(df_ranked))]
    ax.barh(df_ranked['sector_name_en'][::-1],
            df_ranked['Priority'][::-1],
            color=colors[::-1], edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Priority Score')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    st.pyplot(fig)

with col_b:
    st.subheader('Ma trận chuẩn hóa')
    df_norm = Xg.copy()
    df_norm.index = df['sector_name_en']
    df_norm.columns = ['Tăng trưởng','Lan tỏa','Xuất khẩu','Việc làm','AI Ready']
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax2, linewidths=0.5)
    ax2.set_title('Giá trị chuẩn hóa min-max', fontweight='bold')
    st.pyplot(fig2)

st.divider()
st.subheader('Phân tích độ nhạy theo trọng số AI Readiness')
ai_range = np.arange(0.05, 0.45, 0.05)
rank_mat  = []
for w_ai_new in ai_range:
    remaining = 1.0 - w_ai_new - w_risk
    w_tmp = np.array([
        w_growth    / (1-w_ai-w_risk) * remaining,
        w_spillover / (1-w_ai-w_risk) * remaining,
        w_export    / (1-w_ai-w_risk) * remaining,
        w_labor     / (1-w_ai-w_risk) * remaining,
        w_ai_new
    ])
    sc_tmp = Xg.values @ w_tmp - w_risk * Xb.values
    rank_mat.append(pd.Series(sc_tmp).rank(ascending=False).astype(int).values)

rank_df = pd.DataFrame(rank_mat,
                       index=[f'{w:.2f}' for w in ai_range],
                       columns=df['sector_name_en'].values)
fig3, ax3 = plt.subplots(figsize=(12, 4))
sns.heatmap(rank_df.astype(int), annot=True, fmt='d',
            cmap='RdYlGn_r', ax=ax3, linewidths=0.5,
            cbar_kws={'label': 'Xếp hạng'})
ax3.set_title('Độ nhạy xếp hạng theo w_AI', fontweight='bold')
ax3.set_xlabel('Ngành')
ax3.set_ylabel('Trọng số w_AI')
plt.xticks(rotation=30, ha='right')
st.pyplot(fig3)

st.divider()
st.subheader('📌 Nhận xét')
bottom1 = df_ranked.iloc[-1]['sector_name_en']
st.markdown(f"""
- **Top 3 ưu tiên:** {df_ranked.iloc[0]['sector_name_en']}, {df_ranked.iloc[1]['sector_name_en']}, {df_ranked.iloc[2]['sector_name_en']} — phù hợp với Nghị quyết 57-NQ/TW ưu tiên CNTT và công nghiệp chế biến.
- **Ngành thấp nhất:** {bottom1} — thường do AI Readiness thấp hoặc rủi ro tự động hóa cao.
- **Độ nhạy w_AI:** Khi tăng trọng số AI Readiness, thứ hạng các ngành có AI Readiness cao (CNTT-TT) tăng lên — top 3 thay đổi khi w_AI > 0.30.
- **Lưu ý:** Kết quả phụ thuộc vào bộ trọng số — nên kết hợp cả hai góc nhìn "tăng trưởng" và "bao trùm" khi đưa ra quyết định chính sách.
""")