import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Bài 7 — NSGA-II Pareto', layout='wide')
st.title('🎯 Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)')
st.markdown('4 mục tiêu xung đột: GDP, Bất bình đẳng, Môi trường, An ninh dữ liệu.')

st.info('⏳ Bài này chạy NSGA-II (~1-2 phút). Nhấn nút để bắt đầu.')

beta_mat_b7 = np.array([
    [1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],
    [1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],
    [0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25],
])
e_b7   = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
rho_b7 = np.array([0.18,0.45,0.28,0.12,0.52,0.22])
sig_b7 = np.array([0.32,0.28,0.30,0.35,0.25,0.30])

pop_size = st.slider('Population size', 50, 200, 100, 50)
n_gen    = st.slider('Số thế hệ',       50, 300, 200, 50)

if st.button('▶ Chạy NSGA-II', type='primary'):
    with st.spinner('Đang chạy NSGA-II...'):
        try:
            from pymoo.core.problem import ElementwiseProblem
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination

            class VNProblem(ElementwiseProblem):
                def __init__(self):
                    super().__init__(n_var=24, n_obj=4,
                                     n_ieq_constr=12,
                                     xl=np.zeros(24),
                                     xu=np.ones(24)*12000)
                def _evaluate(self, x, out, *args, **kwargs):
                    X = x.reshape(6,4)
                    f1 = -(beta_mat_b7*X).sum()
                    sums = X.sum(axis=1)
                    f2   = np.abs(sums-sums.mean()).mean()
                    f3   = (e_b7*(X[:,0]+X[:,2])).sum()
                    f4   = (rho_b7*X[:,2]).sum()-(sig_b7*X[:,3]).sum()
                    out['F'] = [f1,f2,f3,f4]
                    g = [X.sum()-50000]
                    for i in range(6): g.append(5000-X[i,:].sum())
                    for i in range(5): g.append(X[i,:].sum()-12000)
                    g.append(12000-X[:,3].sum())
                    out['G'] = np.array(g[:12])

            res = minimize(VNProblem(), NSGA2(pop_size=pop_size),
                           get_termination('n_gen', n_gen),
                           seed=42, verbose=False)

            F = res.F.copy()
            F[:,0] = -F[:,0]

            st.success(f'✅ Tìm được {len(F)} nghiệm Pareto')

            col1,col2,col3,col4 = st.columns(4)
            col1.metric('Max GDP gain',   f'{F[:,0].max():,.0f}')
            col2.metric('Min Bất bình đẳng',f'{F[:,1].min():,.1f}')
            col3.metric('Min Phát thải',  f'{F[:,2].min():,.1f}')
            col4.metric('Min Rủi ro AN',  f'{F[:,3].min():,.1f}')

            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader('Pareto Frontier 3D')
                fig = plt.figure(figsize=(6,5))
                ax  = fig.add_subplot(111, projection='3d')
                sc  = ax.scatter(F[:,0],F[:,1],F[:,2],
                                 c=F[:,3], cmap='RdYlGn_r',
                                 s=15, alpha=0.7)
                ax.set_xlabel('GDP gain', fontsize=8)
                ax.set_ylabel('Bất bình đẳng', fontsize=8)
                ax.set_zlabel('Phát thải', fontsize=8)
                plt.colorbar(sc, ax=ax, shrink=0.5, label='Rủi ro AN')
                st.pyplot(fig)

            with col_b:
                st.subheader('GDP vs Bất bình đẳng')
                fig2, ax2 = plt.subplots(figsize=(6,5))
                ax2.scatter(F[:,0],F[:,1], s=15, alpha=0.5,
                            color='steelblue')

                # Nghiệm thỏa hiệp TOPSIS
                w_p   = np.array([0.40,0.25,0.20,0.15])
                ib_f  = [True,False,False,False]
                norm  = np.sqrt((F**2).sum(axis=0))
                R_f   = F/norm; V_f = R_f*w_p
                As    = np.where(ib_f,V_f.max(axis=0),V_f.min(axis=0))
                An    = np.where(ib_f,V_f.min(axis=0),V_f.max(axis=0))
                Ss    = np.sqrt(((V_f-As)**2).sum(axis=1))
                Sn    = np.sqrt(((V_f-An)**2).sum(axis=1))
                Cs    = Sn/(Ss+Sn)
                bi    = np.argmax(Cs)
                gi    = np.argmax(F[:,0])

                ax2.scatter(F[bi,0],F[bi,1], s=200, color='red',
                            marker='*', zorder=5, label='Thỏa hiệp')
                ax2.scatter(F[gi,0],F[gi,1], s=150, color='green',
                            marker='^', zorder=5, label='Max GDP')
                ax2.set_xlabel('GDP gain (tỷ VND)')
                ax2.set_ylabel('Bất bình đẳng')
                ax2.legend()
                ax2.grid(linestyle='--', alpha=0.4)
                st.pyplot(fig2)

            st.divider()
            st.subheader('Nghiệm thỏa hiệp (TOPSIS)')
            st.write(f"- **GDP gain**: {F[bi,0]:,.1f} tỷ VND")
            st.write(f"- **Bất bình đẳng**: {F[bi,1]:,.1f}")
            st.write(f"- **Phát thải**: {F[bi,2]:,.1f}")
            st.write(f"- **Rủi ro AN**: {F[bi,3]:,.1f}")

            st.divider()
            st.subheader('📌 Nhận xét')
            gdp_loss_pct = (F[gi,0] - F[bi,0]) / F[gi,0] * 100
            ineq_ratio   = F[gi,1] / max(F[bi,1], 1)

            # Phát thải — xử lý cả hai hướng
            if F[bi,2] > F[gi,2]:
                emit_txt = (f"Thỏa hiệp: phát thải = **{F[bi,2]:,.0f}** — "
                            f"cao hơn {(F[bi,2]-F[gi,2])/max(F[gi,2],1)*100:.1f}% "
                            f"so với max GDP ({F[gi,2]:,.0f}) vì đầu tư dàn đều hơn tại nhiều vùng.")
            else:
                emit_txt = (f"Thỏa hiệp: phát thải = **{F[bi,2]:,.0f}** — "
                            f"thấp hơn {(F[gi,2]-F[bi,2])/max(F[gi,2],1)*100:.1f}% "
                            f"so với max GDP ({F[gi,2]:,.0f}) — nghiệm thỏa hiệp vừa cân bằng vùng vừa giảm phát thải.")

            # Rủi ro AN — sửa điều kiện: âm hơn = tốt hơn
            if F[bi,3] < F[gi,3]:  # thỏa hiệp âm hơn = an toàn hơn
                an_txt = (f"Thỏa hiệp: **{F[bi,3]:,.0f}** — an toàn hơn max GDP "
                          f"({F[gi,3]:,.0f}) vì đầu tư nhân lực nhiều hơn bù trừ rủi ro AI.")
            else:  # max GDP âm hơn = an toàn hơn
                an_txt = (f"Thỏa hiệp: **{F[bi,3]:,.0f}** | Max GDP: **{F[gi,3]:,.0f}** "
                          f"— max GDP an toàn hơn về dữ liệu do đầu tư nhân lực nhiều hơn, "
                          f"nhưng cả hai đều âm nên rủi ro vẫn trong tầm kiểm soát.")
            st.markdown(f"""
**{len(F)} nghiệm Pareto** tìm được — mỗi điểm trên đồ thị là một cách phân bổ
50,000 tỷ VND khác nhau. Dấu **⭐ đỏ** = nghiệm thỏa hiệp, **▲ xanh** = nghiệm tối đa GDP.

**Đánh đổi GDP — Bình đẳng vùng** *(trục Y đồ thị phải = chênh lệch ngân sách giữa các vùng, đơn vị tỷ VND)*:
- Nghiệm **thỏa hiệp ⭐**: GDP = **{F[bi,0]:,.0f} tỷ**, chênh lệch vùng = **{F[bi,1]:,.0f} tỷ** — phân bổ tương đối đều.
- Nghiệm **max GDP ▲**: GDP = **{F[gi,0]:,.0f} tỷ** (cao hơn {gdp_loss_pct:.1f}%), nhưng chênh lệch vùng = **{F[gi,1]:,.0f} tỷ** — gấp **{ineq_ratio:.1f} lần** so với thỏa hiệp, tức vốn tập trung vào 1-2 vùng có hệ số cao nhất.

**Đánh đổi GDP — Môi trường** *(Phát thải = chỉ số CO2 tương đối từ đầu tư hạ tầng và AI)*:
- {emit_txt}

**Rủi ro an ninh dữ liệu** *(âm = tốt — đầu tư nhân lực đang bù trừ rủi ro AI)*:
- {an_txt}

**Khuyến nghị:** Nghiệm thỏa hiệp phù hợp hơn cho chính sách Việt Nam —
hy sinh {gdp_loss_pct:.1f}% GDP để chênh lệch ngân sách giữa các vùng chỉ còn
**{F[bi,1]:,.0f} tỷ** thay vì **{F[gi,1]:,.0f} tỷ** (bằng 1/{ineq_ratio:.1f} so với max GDP),
phù hợp mục tiêu thu hẹp khoảng cách số trong Quyết định 411/QĐ-TTg.
            """)

        except ImportError:
            st.error('❌ Chưa cài pymoo — chạy: pip install pymoo')
else:
    st.info('Nhấn **▶ Chạy NSGA-II** để bắt đầu.')