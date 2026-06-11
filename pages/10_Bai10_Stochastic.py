import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

st.set_page_config(page_title='Bài 10 - Stochastic SP', layout='wide')
st.title('🎲 Bài 10 - Quy hoạch ngẫu nhiên hai giai đoạn')
st.markdown('Two-stage SP, VSS, EVPI, Robust optimization.')

scenarios=['s1','s2','s3','s4']
items=['I','D','AI','H']
item_names=['Hạ tầng','CĐS','AI','Nhân lực']
prob={'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05}
sc_labels=['Lạc quan','Cơ sở','Bi quan','Khủng hoảng']
beta_b={'I':1.00,'D':1.10,'AI':1.25,'H':0.95}
beta_s={('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
        ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
        ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
        ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10}

if st.button('▶ Chạy Stochastic LP', type='primary'):
    with st.spinner('Đang giải SP + WS...'):
        # SP
        m=pulp.LpProblem('SP_B10',pulp.LpMaximize)
        x=pulp.LpVariable.dicts('x',items,lowBound=0)
        y=pulp.LpVariable.dicts('y',
              [(s,j) for s in scenarios for j in items],lowBound=0)
        m+=(pulp.lpSum(beta_b[j]*x[j] for j in items)+
            pulp.lpSum(prob[s]*beta_s[(s,j)]*y[(s,j)]
                       for s in scenarios for j in items))
        m+=pulp.lpSum(x[j] for j in items)<=65000
        for s in scenarios:
            m+=pulp.lpSum(y[(s,j)] for j in items)<=15000
            m+=y[(s,'AI')]<=0.5*x['H']
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        Z_sp={j:x[j].value() for j in items}
        Z_sp_val=pulp.value(m.objective)

        # WS
        WS_vals={}
        for s in scenarios:
            mw=pulp.LpProblem(f'WS_{s}',pulp.LpMaximize)
            xw=pulp.LpVariable.dicts('xw',items,lowBound=0)
            yw=pulp.LpVariable.dicts('yw',items,lowBound=0)
            mw+=(pulp.lpSum(beta_b[j]*xw[j] for j in items)+
                 pulp.lpSum(beta_s[(s,j)]*yw[j] for j in items))
            mw+=pulp.lpSum(xw[j] for j in items)<=65000
            mw+=pulp.lpSum(yw[j] for j in items)<=15000
            mw+=yw['AI']<=0.5*xw['H']
            mw.solve(pulp.PULP_CBC_CMD(msg=False))
            WS_vals[s]=pulp.value(mw.objective)

        # Robust (minimax regret)
        best_x_robust = None
        best_max_regret = 1e18
        for h_share in np.arange(0.05, 0.80, 0.05):
            for ai_share in np.arange(0.05, 0.80-h_share, 0.05):
                rem = 1 - h_share - ai_share
                for i_share in np.arange(0.05, rem, 0.05):
                    d_share = rem - i_share
                    if d_share < 0.05: continue
                    x_try = {'I':65000*i_share,'D':65000*d_share,
                            'AI':65000*ai_share,'H':65000*h_share}
                    regrets = []
                    for s in scenarios:
                        mr = pulp.LpProblem(f'r_{s}', pulp.LpMaximize)
                        yr = pulp.LpVariable.dicts('yr', items, lowBound=0)
                        mr += pulp.lpSum(beta_s[(s,j)]*yr[j] for j in items)
                        mr += pulp.lpSum(yr[j] for j in items) <= 15000
                        mr += yr['AI'] <= 0.5*x_try['H']
                        mr.solve(pulp.PULP_CBC_CMD(msg=False))
                        z_act = (sum(beta_b[j]*x_try[j] for j in items)
                                + pulp.value(mr.objective))
                        regrets.append(WS_vals[s] - z_act)
                    max_r = max(regrets)
                    if max_r < best_max_regret:
                        best_max_regret = max_r
                        best_x_robust = x_try.copy()

        # EV
        beta_ev={j:sum(prob[s]*beta_s[(s,j)] for s in scenarios)
                 for j in items}
        mev=pulp.LpProblem('EV',pulp.LpMaximize)
        xev=pulp.LpVariable.dicts('xev',items,lowBound=0)
        mev+=pulp.lpSum(beta_ev[j]*xev[j] for j in items)
        mev+=pulp.lpSum(xev[j] for j in items)<=65000
        mev.solve(pulp.PULP_CBC_CMD(msg=False))
        x_ev={j:xev[j].value() for j in items}

        # EEV
        EEV=sum(beta_b[j]*x_ev[j] for j in items)
        for s in scenarios:
            mt=pulp.LpProblem(f'EEV_{s}',pulp.LpMaximize)
            yt=pulp.LpVariable.dicts('yt',items,lowBound=0)
            mt+=pulp.lpSum(beta_s[(s,j)]*yt[j] for j in items)
            mt+=pulp.lpSum(yt[j] for j in items)<=15000
            mt+=yt['AI']<=0.5*x_ev['H']
            mt.solve(pulp.PULP_CBC_CMD(msg=False))
            EEV+=prob[s]*pulp.value(mt.objective)

        WS_bar=sum(prob[s]*WS_vals[s] for s in scenarios)
        VSS=Z_sp_val-EEV
        EVPI=WS_bar-Z_sp_val

    col1,col2,col3,col4,col5=st.columns(5)
    col1.metric('Z* SP',  f'{Z_sp_val:,.0f} tỷ')
    col2.metric('EEV',    f'{EEV:,.0f} tỷ')
    col3.metric('VSS',    f'{VSS:,.0f} tỷ')
    col4.metric('EVPI',   f'{EVPI:,.0f} tỷ')
    col5.metric('Max Regret (Robust)', f'{best_max_regret:,.0f} tỷ')
    
    st.divider()
    col_a,col_b=st.columns(2)

    with col_a:
        st.subheader('Phân bổ giai đoạn 1 (SP vs EV)')
        fig,ax=plt.subplots(figsize=(6,3))
        x_pos=np.arange(4); w=0.35
        ax.bar(x_pos-w/2,[Z_sp[j] for j in items],
               w,label='SP',color='#2196F3',alpha=0.85)
        ax.bar(x_pos+w/2,[x_ev[j] for j in items],
               w,label='EV',color='#FF9800',alpha=0.85)
        ax.bar(x_pos+w,[best_x_robust[j] for j in items],
               w,label='Robust',color='#4CAF50',alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(item_names)
        ax.set_ylabel('Tỷ VND')
        ax.legend()
        ax.grid(axis='y',linestyle='--',alpha=0.4)
        st.pyplot(fig)

    with col_b:
        st.subheader('WS từng kịch bản')
        fig2,ax2=plt.subplots(figsize=(6,3))
        colors=['#4CAF50','#2196F3','#FF9800','#F44336']
        bars=ax2.bar(sc_labels,[WS_vals[s] for s in scenarios],
                     color=colors,edgecolor='white',width=0.5)
        for bar,s in zip(bars,scenarios):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+10,
                     f'{WS_vals[s]:,.0f}',ha='center',fontsize=8)
        ax2.axhline(Z_sp_val,color='blue',linestyle='--',
                    label=f'Z* SP={Z_sp_val:,.0f}')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y',linestyle='--',alpha=0.4)
        st.pyplot(fig2)

    st.divider()
    st.subheader('📌 Nhận xét')
    min_sc = min(WS_vals, key=WS_vals.get)
    max_sc = max(WS_vals, key=WS_vals.get)
    sc_names = {'s1':'Lạc quan','s2':'Cơ sở','s3':'Bi quan','s4':'Khủng hoảng'}
    vss_txt = ('VSS = 0 vì mô hình LP tuyến tính có nghiệm góc duy nhất (dồn vào AI) bất kể kịch bản.'
               if VSS == 0 else
               f'Đáng kể — việc cân nhắc bất định mang lại thêm {VSS:,.0f} tỷ GDP gain.')
    st.markdown(f"""
- **Z\\*_SP = {Z_sp_val:,.0f} tỷ** — lời giải stochastic tối ưu kỳ vọng GDP gain qua 4 kịch bản (xác suất 0.30/0.45/0.20/0.05).
- **VSS = {VSS:,.0f} tỷ** — {vss_txt}
- **EVPI = {EVPI:,.0f} tỷ** — giá trị tối đa sẵn sàng trả để biết trước kịch bản kinh tế toàn cầu.
- **Robust (max regret = {best_max_regret:,.0f} tỷ):** Phân bổ đa dạng hơn SP, chấp nhận giảm kỳ vọng để tránh tổn thất lớn trong kịch bản xấu nhất — phù hợp độ mở thương mại cao của Việt Nam (~180% GDP).
- Kịch bản **{sc_names[max_sc]}** có WS cao nhất ({WS_vals[max_sc]:,.0f} tỷ), **{sc_names[min_sc]}** thấp nhất ({WS_vals[min_sc]:,.0f} tỷ) — trong khủng hoảng, nhân lực số (H) có hệ số cao nhất (1.10) vì lao động qua đào tạo thích ứng tốt hơn với biến động kinh tế.
    """)
else:
    st.info('Nhấn **▶ Chạy Stochastic LP** để xem kết quả.')