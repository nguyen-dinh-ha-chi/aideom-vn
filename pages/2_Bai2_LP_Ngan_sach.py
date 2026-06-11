import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pulp

st.set_page_config(page_title='Bài 2 — LP Ngân sách số', layout='wide')
st.title('💰 Bài 2 — LP Phân bổ ngân sách số')
st.markdown('Tối đa hóa GDP kỳ vọng khi phân bổ 100 nghìn tỷ VND.')

# Slider điều chỉnh ngân sách
budget = st.slider('Tổng ngân sách (nghìn tỷ VND)', 80, 160, 100, 10)

m = pulp.LpProblem('LP_B2', pulp.LpMaximize)
x1 = pulp.LpVariable('HaTang',  lowBound=0)
x2 = pulp.LpVariable('AI',      lowBound=0)
x3 = pulp.LpVariable('NhanLuc', lowBound=0)
x4 = pulp.LpVariable('RD',      lowBound=0)

m += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4
m += x1+x2+x3+x4 <= budget,          'Tong'
m += x1 >= 25,                         'San_HaTang'
m += x2 >= 15,                         'San_AI'
m += x3 >= 20,                         'San_NhanLuc'
m += x4 >= 10,                         'San_RD'
m += x2+x4 >= 0.35*(x1+x2+x3+x4),    'CongNghe'
m.solve(pulp.PULP_CBC_CMD(msg=False))

shadow = {name: c.pi for name, c in m.constraints.items()}
vals  = [x1.value(), x2.value(), x3.value(), x4.value()]
Z_opt = pulp.value(m.objective)
labs  = ['Hạ tầng số', 'AI & Dữ liệu', 'Nhân lực số', 'R&D']

col1, col2, col3 = st.columns(3)
col1.metric('Z* GDP tăng thêm', f'{Z_opt:.2f} nghìn tỷ')
col2.metric('Ngân sách dùng',   f'{sum(vals):.1f} / {budget} nghìn tỷ')
col3.metric('ROI trung bình',   f'{Z_opt/sum(vals):.3f}')

st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader('Phân bổ tối ưu')
    fig, ax = plt.subplots(figsize=(6,3))
    colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0']
    bars = ax.bar(labs, vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3,
                f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Nghìn tỷ VND')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)

with col_b:
    st.subheader('Shadow Price (Giá đối ngẫu)')
    ten_viet = {
        'Tong'      : 'Ngân sách tổng',
        'San_HaTang': 'Sàn hạ tầng số (x1 ≥ 25)',
        'San_AI'    : 'Sàn AI & dữ liệu (x2 ≥ 15)',
        'San_NhanLuc': 'Sàn nhân lực số (x3 ≥ 20)',
        'San_RD'    : 'Sàn R&D (x4 ≥ 10)',
        'CongNghe'  : 'Tỷ trọng công nghệ chiến lược (≥35%)',
    }
    for name, pi in shadow.items():
        ten = ten_viet.get(name, name)
        st.write(f'**{ten}**: {pi:.4f}')
    sp_tong = shadow.get('Tong', 0)
    st.info(f'💡 Tăng 1 nghìn tỷ ngân sách → GDP tăng thêm **{sp_tong:.4f}** nghìn tỷ')

st.divider()
st.subheader('Phân tích độ nhạy Z*(B)')
budgets  = list(range(80, 170, 10))
z_vals   = []
for B in budgets:
    mb = pulp.LpProblem(f'B{B}', pulp.LpMaximize)
    a1=pulp.LpVariable('x1',lowBound=0)
    a2=pulp.LpVariable('x2',lowBound=0)
    a3=pulp.LpVariable('x3',lowBound=0)
    a4=pulp.LpVariable('x4',lowBound=0)
    mb += 0.85*a1+1.20*a2+0.95*a3+1.35*a4
    mb += a1+a2+a3+a4 <= B
    mb += a1>=25; mb += a2>=15; mb += a3>=20; mb += a4>=10
    mb += a2+a4 >= 0.35*(a1+a2+a3+a4)
    mb.solve(pulp.PULP_CBC_CMD(msg=False))
    z_vals.append(pulp.value(mb.objective))

fig, ax = plt.subplots(figsize=(9,3))
ax.plot(budgets, z_vals, marker='o', linewidth=2, color='steelblue')
ax.axvline(budget, color='red', linestyle='--', alpha=0.7,
           label=f'Ngân sách hiện tại={budget}')
for b, z in zip(budgets, z_vals):
    ax.annotate(f'{z:.0f}', xy=(b,z), xytext=(0,8),
                textcoords='offset points', ha='center', fontsize=8)
ax.set_xlabel('Ngân sách (nghìn tỷ VND)')
ax.set_ylabel('Z* GDP tăng thêm')
ax.legend(); ax.grid(linestyle='--', alpha=0.4)
st.pyplot(fig)

st.divider()
st.subheader('📌 Nhận xét')
sp_tong = shadow.get('Tong', 0)
top_item = labs[vals.index(max(vals))]
min_item = labs[vals.index(min(vals))]
st.markdown(f"""
- **Phân bổ tối ưu với ngân sách {budget} nghìn tỷ:** {top_item} nhận nhiều nhất ({max(vals):.1f}), {min_item} nhận ít nhất ({min(vals):.1f}).
- **Shadow price ngân sách = {sp_tong:.4f}:** Tăng 1 nghìn tỷ ngân sách → GDP tăng thêm {sp_tong:.4f} nghìn tỷ. Giá trị này không đổi khi ngân sách thay đổi vì cấu trúc ràng buộc đang binding không thay đổi trong khoảng 80-160 nghìn tỷ.
- **Z* = {Z_opt:.2f} nghìn tỷ** với ROI trung bình = {Z_opt/sum(vals):.3f} — tức mỗi đồng đầu tư tạo ra {Z_opt/sum(vals):.3f} đồng GDP tăng thêm.
""")