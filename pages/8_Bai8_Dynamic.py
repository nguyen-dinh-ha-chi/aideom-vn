import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title='Bài 8 — Tối ưu động', layout='wide')
st.title('📈 Bài 8 — Tối ưu động phân bổ 2026-2035')
st.markdown('SLSQP, Cobb-Douglas, phân tích cú sốc 2028.')

T=10; rho=0.97; dK=0.05; dD=0.12; dAI=0.15
tH=0.8; mu=0.02; p1=0.003; p2=0.002; p3=0.004
alpha=0.33; beta=0.42; gamma=0.10; delta=0.08; theta=0.07
K0=27500.; L0=53.9; D0_v=20.3; AI0=86.; H0=30.; A0=32.

st.sidebar.header('⚙️ Tham số')
rho_v  = st.sidebar.slider('Hệ số chiết khấu ρ', 0.90, 0.99, 0.97, 0.01)
shock  = st.sidebar.checkbox('Cú sốc 2028 (-8%)', value=False)

def simulate(I_vec, shock_yr=None):
    IK=I_vec[:T]; ID=I_vec[T:2*T]
    IAI=I_vec[2*T:3*T]; IH=I_vec[3*T:]
    K=np.zeros(T+1); K[0]=K0
    D=np.zeros(T+1); D[0]=D0_v
    AI=np.zeros(T+1); AI[0]=AI0
    H=np.zeros(T+1); H[0]=H0
    A=np.zeros(T+1); A[0]=A0
    Y=np.zeros(T); C=np.zeros(T)
    for t in range(T):
        Y[t]=A[t]*(K[t]**alpha)*(L0**beta)*\
             (D[t]**gamma)*(AI[t]**delta)*(H[t]**theta)
        if shock_yr and t==shock_yr: Y[t]*=0.92
        C[t]=Y[t]-IK[t]-ID[t]-IAI[t]-IH[t]
        K[t+1]=(1-dK)*K[t]+IK[t]
        D[t+1]=(1-dD)*D[t]+ID[t]
        AI[t+1]=(1-dAI)*AI[t]+IAI[t]
        H[t+1]=H[t]+tH*IH[t]-mu*H[t]
        A[t+1]=A[t]*(1+p1*D[t]+p2*AI[t]+p3*H[t])
    return K,D,AI,H,A,Y,C

def welfare(I_vec, shock_yr=None):
    _,_,_,_,_,Y,C=simulate(I_vec,shock_yr)
    pen=sum(1e10 for t in range(T) if C[t]<=0)
    W=sum(rho_v**t*np.log(max(C[t],1e-6)) for t in range(T))
    return -(W-pen)

if st.button('▶ Chạy tối ưu SLSQP', type='primary'):
    with st.spinner('Đang tối ưu (~30-60 giây)...'):
        Y_init=A0*(K0**alpha)*(L0**beta)*(D0_v**gamma)*(AI0**delta)*(H0**theta)
        I0=np.ones(4*T)*Y_init*0.15/4
        bounds=[(0,Y_init*0.6)]*(4*T)
        shock_yr = 2 if shock else None

        res=minimize(lambda I: welfare(I,shock_yr), I0,
                     method='SLSQP', bounds=bounds,
                     options={'maxiter':500,'ftol':1e-8,'disp':False})

    I_opt=res.x
    K_o,D_o,AI_o,H_o,A_o,Y_o,C_o=simulate(I_opt,shock_yr)
    years=np.arange(2026,2036)

    st.success(f'✅ Welfare tối ưu: {-res.fun:.4f}')
    st.divider()

    fig,axes=plt.subplots(2,3,figsize=(15,8))
    plots=[(K_o[:-1],'Vốn K','#2196F3',axes[0,0]),
           (D_o[:-1],'Hạ tầng số D','#FF9800',axes[0,1]),
           (AI_o[:-1],'AI capacity','#9C27B0',axes[0,2]),
           (H_o[:-1],'Nhân lực H','#4CAF50',axes[1,0]),
           (Y_o,'GDP Y','#F44336',axes[1,1]),
           (C_o,'Tiêu dùng C','#00BCD4',axes[1,2])]
    for vals,label,color,ax in plots:
        ax.plot(years,vals,marker='o',linewidth=2,color=color,markersize=4)
        ax.set_title(label,fontweight='bold')
        ax.set_xticks(years[::2])
        ax.grid(linestyle='--',alpha=0.4)
    plt.suptitle('Quỹ đạo tối ưu 2026-2035',fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader('📌 Nhận xét')
    W_val = -res.fun
    st.markdown(f"""
    - **Welfare tối ưu = {W_val:.4f}** — hàm log tiêu dùng chiết khấu ρ={rho_v} qua 10 năm 2026-2035.
    - **Quỹ đạo GDP bùng nổ** do hàm TFP nội sinh A[t+1] = A[t]·(1 + φ·D + φ·AI + φ·H) tăng theo cấp số nhân — đây là hạn chế của mô hình tuyến tính hóa TFP theo yêu cầu đề bài.
    - {'**Cú sốc 2028 (-8%):** Kịch bản giả định — nếu xảy ra cú sốc bên ngoài (tương tự bão Yagi 2024) làm GDP giảm 8% năm 2028, mô hình điều chỉnh bằng cách cắt giảm tiêu dùng để bảo vệ đầu tư dài hạn vào D, AI, H.' if shock else '**Chưa bật cú sốc:** Bật checkbox *Cú sốc 2028 (-8%)* trong sidebar bên trái để xem mô hình phản ứng ra sao khi GDP năm 2028 giảm đột ngột 8% — mô phỏng tác động của thiên tai hoặc khủng hoảng kinh tế.'}
    - **Chiến lược front-load** (đầu tư mạnh 3 năm đầu) thường cho welfare cao hơn trải đều nhờ hiệu ứng tích lũy vốn sớm — phù hợp Quyết định 749/QĐ-TTg.
    """)
else:
    st.info('Nhấn **▶ Chạy tối ưu SLSQP** để bắt đầu.')