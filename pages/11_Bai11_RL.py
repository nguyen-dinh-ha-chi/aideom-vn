import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title='Bài 11 — Q-learning RL', layout='wide')
st.title('🤖 Bài 11 — Q-learning cho chính sách kinh tế thích nghi')
st.markdown('MDP, epsilon-greedy, DQN, so sánh chính sách π* vs rule-based.')

# ============================================================
# THAM SỐ 
# ============================================================
ACTIONS = {
    0: np.array([0.70,0.10,0.10,0.10]),
    1: np.array([0.40,0.25,0.15,0.20]),
    2: np.array([0.25,0.45,0.15,0.15]),
    3: np.array([0.20,0.20,0.45,0.15]),
    4: np.array([0.30,0.20,0.10,0.40]),
}
ACTION_NAMES = ['Truyền thống','Cân bằng','Số hóa nhanh','AI dẫn dắt','Bao trùm']
N_ACTIONS = 5
BUDGET    = 1000
T         = 10
w         = np.array([0.40,0.25,0.20,0.15])
alpha_K,alpha_L       = 0.33,0.42
alpha_D,alpha_AI,alpha_H = 0.10,0.08,0.07

def discretize(val, thresholds):
    if val < thresholds[0]: return 0
    if val < thresholds[1]: return 1
    return 2

# ============================================================
# MÔI TRƯỜNG 
# ============================================================
class VietnamEconomyEnv:
    def __init__(self):
        self.T=T
        self.K=27500.; self.D=20.3; self.AI=86.
        self.H=30.; self.L=53.9; self.A=1.; self.t=0

    def reset(self):
        self.K,self.D,self.AI,self.H=27500.,20.3,86.,30.
        self.L,self.A,self.t=53.9,1.,0
        self.state=np.array([np.random.randint(3),
                             np.random.randint(3),
                             np.random.randint(3),
                             np.random.randint(3)])
        return self.state.copy()

    def _compute_state(self):
        Y=self.A*(self.K**alpha_K)*(self.L**alpha_L)*\
          (self.D**alpha_D)*(self.AI**alpha_AI)*(self.H**alpha_H)
        gdp_growth=(Y-getattr(self,'_Y_prev',Y))/max(getattr(self,'_Y_prev',Y),1)
        self._Y_prev=Y
        g=discretize(gdp_growth,[0.05,0.08])
        d=discretize(self.D,[18,25])
        a=discretize(self.AI,[80,120])
        u=discretize(max(0,0.05-self.AI/5000),[0.02,0.04])
        return np.array([g,d,a,u]),Y

    def step(self, action):
        alloc=ACTIONS[action]
        self.K  =0.95*self.K  +alloc[0]*BUDGET
        self.D  =0.88*self.D  +alloc[1]*BUDGET/100
        self.AI =0.85*self.AI +alloc[2]*BUDGET/20
        self.H  =self.H*0.98  +alloc[3]*BUDGET/200
        self.L  =min(self.L*1.005,60.)
        self.A *=1.012
        new_state,Y=self._compute_state()
        delta_gdp=max(0,(Y-getattr(self,'_Y_base',Y))/max(getattr(self,'_Y_base',Y),1))
        self._Y_base=getattr(self,'_Y_base',Y)
        unemp      =max(0,0.08-self.AI/3000)
        cyber_risk =alloc[2]*0.3-alloc[3]*0.1
        emission   =(alloc[0]+alloc[2])*0.2
        reward=(w[0]*delta_gdp*100
                -w[1]*unemp*100
                -w[2]*max(0,cyber_risk)
                -w[3]*emission)
        self.state=new_state; self.t+=1
        done=self.t>=self.T
        return new_state.copy(),reward,done

# ============================================================
# DQN 
# ============================================================
class DQN:
    def __init__(self, n_input=12, n_hidden=64, n_output=5, lr=0.001):
        self.lr=lr; scale=0.1
        self.W1=np.random.randn(n_input,n_hidden)*scale
        self.b1=np.zeros(n_hidden)
        self.W2=np.random.randn(n_hidden,n_hidden)*scale
        self.b2=np.zeros(n_hidden)
        self.W3=np.random.randn(n_hidden,n_output)*scale
        self.b3=np.zeros(n_output)

    def relu(self,x): return np.maximum(0,x)
    def relu_grad(self,x): return (x>0).astype(float)

    def forward(self,x):
        self.x=x
        self.z1=x@self.W1+self.b1; self.a1=self.relu(self.z1)
        self.z2=self.a1@self.W2+self.b2; self.a2=self.relu(self.z2)
        self.z3=self.a2@self.W3+self.b3
        return self.z3

    def backward(self,loss_grad):
        dz3=loss_grad
        dW3=self.a2[:,None]*dz3[None,:]
        db3=dz3
        da2=dz3@self.W3.T
        dz2=da2*self.relu_grad(self.z2)
        dW2=self.a1[:,None]*dz2[None,:]
        db2=dz2
        da1=dz2@self.W2.T
        dz1=da1*self.relu_grad(self.z1)
        dW1=self.x[:,None]*dz1[None,:]
        db1=dz1
        for W,dW in [(self.W1,dW1),(self.W2,dW2),(self.W3,dW3)]:
            W-=self.lr*dW
        for b,db in [(self.b1,db1),(self.b2,db2),(self.b3,db3)]:
            b-=self.lr*db

def state_to_onehot(s):
    x=np.zeros(12)
    for i,val in enumerate(s): x[i*3+val]=1.0
    return x

class ReplayBuffer:
    def __init__(self,capacity=2000):
        self.buf=[]; self.cap=capacity
    def push(self,t):
        if len(self.buf)>=self.cap: self.buf.pop(0)
        self.buf.append(t)
    def sample(self,n):
        idx=np.random.choice(len(self.buf),min(n,len(self.buf)),replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header('⚙️ Tham số')
n_episodes = st.sidebar.slider('Episodes Q-learning', 1000,15000,10000,1000)
n_ep_dqn   = st.sidebar.slider('Episodes DQN',        500, 5000, 3000, 500)
alpha_rl   = st.sidebar.slider('Learning rate α',     0.01,0.5,  0.1,  0.01)
gamma_rl   = st.sidebar.slider('Discount γ',          0.80,0.99, 0.95, 0.01)

if st.button('▶ Huấn luyện Q-learning + DQN', type='primary'):

    # ============================================================
    # HUẤN LUYỆN Q-LEARNING
    # ============================================================
    with st.spinner('Đang huấn luyện Q-learning...'):
        np.random.seed(42)
        env=VietnamEconomyEnv()
        Q=np.zeros((3,3,3,3,N_ACTIONS))
        rewards_history=[]

        for ep in range(n_episodes):
            s=env.reset()
            eps=max(0.05,1.0-ep/(n_episodes*0.5))
            ep_reward=0; done=False
            while not done:
                a=np.random.randint(N_ACTIONS) if np.random.rand()<eps \
                  else int(np.argmax(Q[tuple(s)]))
                s2,r,done=env.step(a)
                ep_reward+=r
                best_next=np.max(Q[tuple(s2)])
                Q[tuple(s)+(a,)]+=alpha_rl*(r+gamma_rl*best_next-Q[tuple(s)+(a,)])
                s=s2
            rewards_history.append(ep_reward)

    # ============================================================
    # HUẤN LUYỆN DQN
    # ============================================================
    with st.spinner('Đang huấn luyện DQN...'):
        np.random.seed(42)
        env_dqn = VietnamEconomyEnv()
        dqn=DQN(12,64,5,0.001)
        target_dqn=DQN(12,64,5,0.001)
        for attr in ['W1','b1','W2','b2','W3','b3']:
            setattr(target_dqn,attr,getattr(dqn,attr).copy())

        buffer=ReplayBuffer(2000)
        dqn_rewards=[]; batch_size=32; gamma_dqn=0.95

        for ep in range(n_ep_dqn):
            s=env.reset()
            eps_dqn=max(0.05,1.0-ep/(n_ep_dqn*0.5))
            ep_r=0; done=False
            while not done:
                x=state_to_onehot(s)
                a=np.random.randint(N_ACTIONS) if np.random.rand()<eps_dqn \
                  else int(np.argmax(dqn.forward(x)))
                s2,r,done=env.step(a)
                buffer.push((s.copy(),a,r,s2.copy(),done))
                ep_r+=r; s=s2
                if len(buffer)>=batch_size:
                    batch=buffer.sample(batch_size)
                    for (bs,ba,br,bs2,bdone) in batch:
                        xb=state_to_onehot(bs); xb2=state_to_onehot(bs2)
                        qv=dqn.forward(xb); qt=qv.copy()
                        qt[ba]=br if bdone else br+gamma_dqn*np.max(target_dqn.forward(xb2))
                        lg=np.zeros(N_ACTIONS); lg[ba]=2*(qv[ba]-qt[ba])/batch_size
                        dqn.backward(lg)
                if done: break
            if (ep+1)%100==0:
                for attr in ['W1','b1','W2','b2','W3','b3']:
                    setattr(target_dqn,attr,getattr(dqn,attr).copy())
            dqn_rewards.append(ep_r)

    # ============================================================
    # ĐÁNH GIÁ CHÍNH SÁCH
    # ============================================================
    def run_policy(policy_fn, n_eval=200):
        np.random.seed(42)          
        env_eval = VietnamEconomyEnv()
        results=[]
        for _ in range(n_eval):
            s=env.reset(); total_r=0; done=False
            while not done:
                a=policy_fn(s); s,r,done=env.step(a); total_r+=r
            results.append(total_r)
        return results

    r_opt  = run_policy(lambda s: int(np.argmax(Q[tuple(s)])))
    r_a1   = run_policy(lambda s: 1)
    r_a3   = run_policy(lambda s: 3)
    r_rand = run_policy(lambda s: np.random.randint(N_ACTIONS))
    r_dqn  = run_policy(lambda s: int(np.argmax(dqn.forward(state_to_onehot(s)))))

    st.success('✅ Huấn luyện hoàn tất!')

    # ============================================================
    # KPI CARDS
    # ============================================================
    col1,col2,col3 = st.columns(3)
    col1.metric('π* Avg Reward',   f'{np.mean(r_opt):.4f}')
    col2.metric('DQN Avg Reward',  f'{np.mean(r_dqn):.4f}')
    improvement=(np.mean(r_dqn)-np.mean(r_opt))/abs(np.mean(r_opt))*100
    col3.metric('DQN vs Q-learning', f'{improvement:+.2f}%')

    st.divider()

    # ============================================================
    # CHÍNH SÁCH π*(s)
    # ============================================================
    st.subheader('🎯 Chính sách tối ưu π*(s)')
    test_states=[
        (np.array([1,1,0,1]),'VN 2026 thực tế [GDP=mid,D=mid,AI=low,U=mid]'),
        (np.array([0,0,0,2]),'Khủng hoảng [GDP=low,D=low,AI=low,U=high]'),
        (np.array([2,2,2,0]),'Tăng trưởng cao [GDP=high,D=high,AI=high,U=low]'),
        (np.array([1,0,0,1]),'Số hóa chậm [GDP=mid,D=low,AI=low,U=mid]'),
        (np.array([2,1,2,0]),'AI phát triển mạnh [GDP=high,D=mid,AI=high,U=low]'),
    ]
    rows=[]
    for st_arr,desc in test_states:
        best_a=int(np.argmax(Q[tuple(st_arr)]))
        rows.append({
            'Trạng thái'         : desc,
            'Hành động tối ưu'   : ACTION_NAMES[best_a],
            'Q-value'            : f'{Q[tuple(st_arr)].max():.4f}'
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # ============================================================
    # BIỂU ĐỒ 2x2 
    # ============================================================
    fig,axes=plt.subplots(2,2,figsize=(13,9))

    # 1 — Learning curve Q-learning
    window=200
    smooth=np.convolve(rewards_history,np.ones(window)/window,mode='valid')
    axes[0,0].plot(smooth,color='#2196F3',linewidth=1.5,label='Q-learning (smoothed)')
    axes[0,0].set_title('Learning Curve — Q-learning',fontweight='bold')
    axes[0,0].set_xlabel('Episode'); axes[0,0].set_ylabel('Tổng reward')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    # 2 — So sánh chính sách (bao gồm DQN)
    policy_names=['π* Q-learning','Luôn a1\n(Cân bằng)',
                  'Luôn a3\n(AI dẫn dắt)','Random','DQN']
    policy_means=[np.mean(r_opt),np.mean(r_a1),
                  np.mean(r_a3),np.mean(r_rand),np.mean(r_dqn)]
    policy_stds =[np.std(r_opt), np.std(r_a1),
                  np.std(r_a3), np.std(r_rand), np.std(r_dqn)]
    colors_p=['#2196F3','#FF9800','#9C27B0','#9E9E9E','#4CAF50']
    bars=axes[0,1].bar(policy_names,policy_means,color=colors_p,
                       yerr=policy_stds,capsize=5,edgecolor='white')
    axes[0,1].set_title('So sánh reward trung bình các chính sách',fontweight='bold')
    axes[0,1].set_ylabel('Avg reward (200 episodes)')
    axes[0,1].grid(axis='y',alpha=0.3)
    for bar,val in zip(bars,policy_means):
        axes[0,1].text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+0.002,
                       f'{val:.3f}',ha='center',fontsize=8)

    # 3 — Learning curve Q-learning vs DQN
    smooth_dqn=np.convolve(dqn_rewards,np.ones(100)/100,mode='valid')
    axes[1,0].plot(smooth_dqn,color='#4CAF50',linewidth=1.5,label='DQN (smoothed)')
    axes[1,0].plot(np.convolve(rewards_history[:n_ep_dqn],
                               np.ones(100)/100,mode='valid'),
                   color='#2196F3',linewidth=1,alpha=0.6,
                   label=f'Q-learning ({n_ep_dqn} ep)')
    axes[1,0].set_title('Learning Curve — Q-learning vs DQN',fontweight='bold')
    axes[1,0].set_xlabel('Episode'); axes[1,0].set_ylabel('Tổng reward')
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    # 4 — Heatmap chính sách π*
    policy_map=np.zeros((3,3))
    for g in range(3):
        for a in range(3):
            s=np.array([g,1,a,1])
            policy_map[g,a]=int(np.argmax(Q[tuple(s)]))

    im=axes[1,1].imshow(policy_map,cmap='Set1',vmin=0,vmax=4,aspect='auto')
    axes[1,1].set_xticks([0,1,2])
    axes[1,1].set_xticklabels(['AI thấp','AI trung','AI cao'])
    axes[1,1].set_yticks([0,1,2])
    axes[1,1].set_yticklabels(['GDP thấp','GDP trung','GDP cao'])
    axes[1,1].set_title('Bản đồ chính sách π* (D=mid,U=mid)',fontweight='bold')
    legend_patches=[mpatches.Patch(color=plt.cm.Set1(i/8),
                                   label=f'a{i}: {ACTION_NAMES[i]}')
                    for i in range(5)]
    axes[1,1].legend(handles=legend_patches,loc='upper right',fontsize=7)
    for g in range(3):
        for a in range(3):
            axes[1,1].text(a,g,ACTION_NAMES[int(policy_map[g,a])].split()[0],
                           ha='center',va='center',fontsize=7,
                           color='white',fontweight='bold')

    plt.suptitle('Bài 11 — Học tăng cường cho chính sách kinh tế thích nghi',
                 fontsize=13,fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # ============================================================
    # BẢNG SO SÁNH CHI TIẾT
    # ============================================================
    st.divider()
    st.subheader('📊 Bảng so sánh chi tiết')
    df_cmp=pd.DataFrame({
        'Chính sách' : ['π* (Q-learning)','Luôn Cân bằng',
                        'Luôn AI dẫn dắt','Random','DQN'],
        'Avg Reward' : [f'{np.mean(r):.4f}'
                        for r in [r_opt,r_a1,r_a3,r_rand,r_dqn]],
        'Std'        : [f'{np.std(r):.4f}'
                        for r in [r_opt,r_a1,r_a3,r_rand,r_dqn]],
        'Min'        : [f'{np.min(r):.4f}'
                        for r in [r_opt,r_a1,r_a3,r_rand,r_dqn]],
        'Max'        : [f'{np.max(r):.4f}'
                        for r in [r_opt,r_a1,r_a3,r_rand,r_dqn]],
    })
    st.dataframe(df_cmp, use_container_width=True)

    best=policy_names[np.argmax(policy_means)]
    st.info(f'💡 Chính sách tốt nhất: **{best}** '
            f'với avg reward = **{max(policy_means):.4f}**')

else:
    st.info('Nhấn **▶ Huấn luyện Q-learning + DQN** để bắt đầu. '
            'Bài này chạy khoảng 2-3 phút.')