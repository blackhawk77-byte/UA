# UA(user association)

## Multi-task Multi-agent RL(MT-MARL)
```math
\mathrm{Dec\text{-}POMDP}: \langle I, S, A, T, R, \Omega, O, \gamma \rangle
```
- $I$: # of agents $(n)$  
- $S$: state space  
- $A$: joint action space  
- $\Omega$: joint observation space  
- $T$: transition probability  
- $O$: joint observation probability  
- policy: local observation history $\rightarrow$ action  
- joint policy: $\langle \pi_1, \ldots, \pi_n \rangle$  
- $R$: joint reward $R(s_t,\ \text{joint}_{a_t})$

**Definition 1. stationary**  
- 다른 에이전트 액션이 달라졌을 때($a_t^{(-i)} \ne u_t^{(-i)}$), **다음 상태 분포 불변**  
- 다른 에이전트 액션이 달라졌을 때, **에이전트 $i$의 관측 분포 불변**

**Definition 2. partially-observable MT-MARL Domain**  
```math
D = \langle I, S, A, \Omega, \gamma \rangle
```
**Definition 3. partially-observable MT-MARL Task $T_j$**  
```math
T_j = \langle D,\ T_j,\ R_j,\ O_j \rangle
```
- $D$: Definition 2에서 정의한 도메인 (shared)  
- $T_j$: transition function (task-specific)  
- $R_j$: reward function (task-specific)  
- $O_j$: observation function (task-specific)

$\Rightarrow$ Dec-POMDP 구성요소 중 **Definition 2는 고정**, **Definition 3만 가변**  
**Training**: 각 에피소드마다 **task id**가 부여됨  
**Test**: task id **부여 X**  
**Objective function**  

$$
V=\frac{1}{E}\sum_{e=0}^{E}\sum_{t=0}^{H_e}\gamma^t\, R_e(s_t,\ \text{joint}_{a_t})
$$

$H_e$: max\_timestep,  $E$: episode

---

## Approach

### Dec-HDRQN
distributed Q-learning(DTDE) → exploration에 의한 low return 무시 → **overestimation of Q-values**  
$\Rightarrow$ **hysteretic Q-learning** (low return도 무시 안함)  
- 방식: learning rate $0<\beta<\alpha<1$,  
  - TD-error $>0$: $\alpha$  
  - TD-error $<0$: $\beta$  
- 결과: 이전에 성공적인 협력을 발생시킨 **좋은 액션들에 대해 Q-value degradation에 지연(hysteresis)**, exploration 때문에 발생하는 negative 변화에 강인

### CERT
```math
\mathrm{experience\,replay\,buffer}: \langle s, a, r, s' \rangle
```
- 장점1: sampling cost 줄어듦 (하나의 샘플로도 여러 번 업데이트 가능)  
- 장점2: sample들의 temporal correlation을 줄임 (iid data → generalization error 감소)

DTDE 알고리즘에서는 에이전트 간 **desynchronization**하게 버퍼에 저장될 수 있고, 그렇게 되면 **shadowed equilibria** 문제 발생 (sub-optima)  
예) Agent $A_1, A_2 = \{a_1, a_2\}$, optimal joint action = $\langle a_1, a_1\rangle$ 또는 $\langle a_2, a_2\rangle$,  
각각 local optimum: $A_1=\{a_1\}$, $A_2=\{a_2\}$, $\langle a_1, a_2\rangle$ → poor joint action

$\Rightarrow$ **CERT (concurrent Experience Replay Trajectories, FIFO)**  
- 축: (e)pisode, (t)imestep, (i)ndice of agent  
- each cube: $\langle o_t^{(i)},\ a_t^{(i)},\ r_t,\ o_{t+1}^{(i)} \rangle$  
- **Execution**: agent $i$가 cube(경험 튜플) 수집  
- **Training**: DRQN agents는 **trace length: $\tau$**로 된 시퀀스 학습

**동일 (episode, timestep) 동기화 방법**  
- DTDE agent들이 각각 **동일한 에피소드의 동일한 타임스텝 + $\tau$**만큼 샘플링  
- 방법: **RNG(Random Number Generator) seed 공유** (+ 에피소드 ID 목록 정렬)

샘플링 형태 (batch size $B$, trace length $\tau$):  

$$
B = \Big\langle\ \langle o_{t_0}, a_{t_0}, r_{t_0}, o_{t_0+1}\rangle,\ \ldots,\ \langle o_{t_0+\tau-1}, a_{t_0+\tau-1}, r_{t_0+\tau-1}, o_{t_0+\tau}\rangle\ \Big\rangle,\ \{b=1\sim B\}
$$

- 샘플된 트레이스 구간이 **에피소드 길이보다 크면** → **zero padding (suffix)**  
- 총 **batch size × trace length($\tau$)** 만큼 샘플링  
- **No prefix padding**: RNN은 `first_valid[b]`부터 unroll (그 이전은 완전히 무시) → 내부 상태 오염 방지  
- **No suffix**: zero-padding + `mask=0`으로 길이 $\tau$ 유지, 손실/타깃 계산에서 제외

**Target values**  
```math
\mathrm{target\_values}=\langle y_{t_0},\ldots,y_{t_0+\tau-1}\rangle,\ \ (b=1\sim B)
```
모든 RNN output에 대해 TD-target 계산하여 loss 계산 (`target_values.shape == BS * tau`)

$$
y_t = r_t + \gamma \max_{a'} Q\big(o_{t+1},\ h_t,\ a';\ \theta_i^{-}\big),\qquad
\delta_t = y_t - Q\big(o_t,\ h_{t-1},\ a_t;\ \theta_i\big)
$$

**Loss function** (i: agent index, j: iteration number for target network update)

$$
L(\theta) = \mathbb{E}\\left[(\delta_t)^2\right]
= \frac{1}{BS \cdot \tau}\sum_{b=1}^{BS}\sum_{t=1}^{\tau-1} (\delta_t)^2
$$

---

## Dec-POMDP MT-MARL

위에까지는 **single task**에 대한 Dec-HDRQN 학습 네트워크 설명.  
이제 **multi-task**로 확장.

**policy distillation**  
- explicit한 **TASK ID 없이도** 모든 태스크에서 잘 동작하도록 함

**multi-task로 확장하는 방법: Q-values에 대한 regression**  
- (data collection → regression)  
- Dec-HDRQN에서 $CERT^{\{M_R\}}$이 **single task마다** 존재한다고 정의  
- $\langle o_t^{(i)}, Q_t^{(i)} \rangle$ 정보 포함  
- $Q_t^{(i)} = Q(\text{observation history};\ \theta)$: DRQN의 Q-value **vector** for agent $i$ at timestep $t$  
- $\Rightarrow$ **supervised learning**

**regression batch**  

```math
\mathcal{B}_r = \Big\langle \langle o_{t_0}, Q_{t_0}\rangle, \ldots, \langle o_{t_0+\tau-1}, Q_{t_0+\tau-1}\rangle \Big\rangle,\quad b=1\sim BS
```

**Loss function (using KL divergence for a single distilled policy)**  
- $L_{\mathrm{kl}}(\mathcal{B}_r,\ \theta_r^{(i)};\ T)$ → *deep decentralized MT-MARL under partial observability* 논문 식 (6) 참조  
- $T$: softmax temperature — 각 $Q_t$ 값의 **sharpening** 조절 → distilled policy가 비슷한 action 선택

