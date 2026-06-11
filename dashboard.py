
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳", layout="wide")
st.title("🇻🇳 AIDEOM-VN Dashboard")
st.markdown("**Mô hình Ra Quyết Định Kinh Tế Việt Nam | 2026–2030**")

# Load KPI từ file đã chạy
try:
    df_kpi = pd.read_csv("aideom_kpi_summary.csv", index_col=0)
    st.success("✅ Đã tải dữ liệu KPI từ aideom_kpi_summary.csv")
except:
    st.error("❌ Chưa có file KPI — hãy chạy notebook bài 12 trước")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan", "📋 So sánh kịch bản",
    "🗺️ Phân bổ vùng", "⚠️ Cảnh báo rủi ro"
])

with tab1:
    st.header("Tổng quan KPI")
    gdp_2025 = df_kpi.iloc[0]["GDP_2025 (nghìn tỷ)"]
    gdp_2030 = df_kpi.iloc[0]["GDP_2030 dự báo"]
    mape_val = df_kpi.iloc[0]["MAPE (%)"]

    col1, col2, col3 = st.columns(3)
    col1.metric("GDP 2025",       f"{gdp_2025:.1f} nghìn tỷ")
    col2.metric("GDP 2030 dự báo",f"{gdp_2030:.1f} nghìn tỷ")
    col3.metric("MAPE dự báo",    f"{mape_val:.2f}%")
    st.dataframe(df_kpi, use_container_width=True)

with tab2:
    st.header("So sánh 5 kịch bản")
    col = st.selectbox("Chọn chỉ tiêu:", df_kpi.select_dtypes("number").columns)
    fig, ax = plt.subplots(figsize=(9,4))
    colors = ["#2196F3","#FF9800","#9C27B0","#4CAF50","#F44336"]
    ax.bar(df_kpi.index, df_kpi[col], color=colors, alpha=0.85)
    ax.set_title(col, fontweight="bold")
    ax.set_xticklabels(df_kpi.index, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

with tab3:
    st.header("Phân bổ ngân sách theo vùng")
    region_names_vn = ["Trung du MN phía Bắc","Đồng bằng sông Hồng",
                        "Bắc TB + DH Trung Bộ","Tây Nguyên",
                        "Đông Nam Bộ","Đồng bằng sông CL"]
    selected_sc = st.selectbox("Kịch bản:", df_kpi.index)
    z_val = df_kpi.loc[selected_sc, "Z* GDP gain (tỷ)"]
    st.metric(f"Z* GDP Gain — {selected_sc}", f"{z_val:,.1f} tỷ VND")

with tab4:
    st.header("⚠️ Cảnh báo rủi ro")
    st.subheader("Rủi ro theo kịch bản")
    fig2, ax2 = plt.subplots(figsize=(9,4))
    ax2.bar(df_kpi.index, df_kpi["Rủi ro MT"],
            color="#F44336", alpha=0.8, label="Môi trường")
    ax2.set_title("Rủi ro môi trường theo kịch bản", fontweight="bold")
    ax2.set_xticklabels(df_kpi.index, rotation=15, ha="right")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig2)

    best_sc = df_kpi["Rủi ro MT"].idxmin()
    st.info(f"✅ Kịch bản ít rủi ro MT nhất: **{best_sc}**")
    worst_sc = df_kpi["Tổng NetJob (nghìn)"].idxmax()
    st.success(f"🏆 Kịch bản tạo nhiều việc làm nhất: **{worst_sc}**")
