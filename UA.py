import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import random
import matplotlib.pyplot as plt

# ConcurrentReplayBuffer 
class ConcurrentReplayBuffer:
    """
    Fully decentralized per-agent CERT buffer.
    """
    def __init__(self, 
                 obs_shape: Tuple[int, ...],
                 act_shape: Tuple[int, ...],
                 capacity: int =1000, 
                 dtype_obs: np.dtype = np.float32,
                 dtype_act: np.dtype = np.int64,
                 dtype_rew: np.dtype = np.float32, 
                ):
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.capacity = capacity
        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act
        self.dtype_rew = dtype_rew

        self._episode_order = deque()
        self._episodes: Dict[Any,Dict[str, Any]] = {}
        self._len = 0

    def push(self,
             episode_id: Any,
             t: int, 
             obs_t: np.ndarray,
             act_t: np.ndarray,
             rew_t: Optional[float],
             next_obs_t: np.ndarray,
             done: bool=False) -> None:
        
        assert obs_t.shape == self.obs_shape, f"obs shape {obs_t.shape} != {self.obs_shape}"
        assert next_obs_t.shape == self.obs_shape, f"next_obs shape {next_obs_t.shape} != {self.obs_shape}"
        assert act_t.shape == self.act_shape, f"act shape {act_t.shape} != {self.act_shape}"

        if episode_id not in self._episodes:
            if len(self._episode_order) >= self.capacity:
                evict_id = self._episode_order.popleft()
                evicted = self._episodes.pop(evict_id, None)
                if evicted is not None:
                    for rec in evicted["steps"].values():
                        if "rew" in rec:
                            self._len -= 1
            
            self._episodes[episode_id] = {"steps": {},"T": 0}
            self._episode_order.append(episode_id)

        ep = self._episodes[episode_id]
        rec = ep["steps"].get(t)
        if rec is None:
            rec = {}
            ep["steps"][t] = rec

        rec["obs"] = np.asarray(obs_t, dtype=self.dtype_obs)
        rec["act"] = np.asarray(act_t, dtype=self.dtype_act)
        rec["next_obs"] = np.asarray(next_obs_t, dtype=self.dtype_obs)

        if rew_t is not None:
            if "rew" not in rec:
                self._len += 1
            rec["rew"] = np.asarray(rew_t, dtype=self.dtype_rew)

        rec["done"] = bool(done)

        if t + 1 > ep["T"]:
            ep["T"] = t + 1

    def _list_episode_ids(self) -> List[Any]:
        return list(self._episode_order)
    
    def sample(self,
               batch_size: int,
               tau: int,
               seed: Optional[int] =None,
            ) -> Dict[str, np.ndarray]:
        
        assert batch_size >0 and tau > 0, "batch_size and tau have to positive values"
        rng = random.Random(seed) if seed is not None else random

        ep_ids = self._list_episode_ids()
        if not ep_ids:
            raise ValueError("Buffer is empty")
        
        nonempty_eps = sorted([eid for eid in ep_ids if self._episodes[eid]["T"] > 0])
        if not nonempty_eps:
            raise ValueError("No non-empty episodes")
        
        B = batch_size
        obs = np.zeros((B, tau, *self.obs_shape), dtype=self.dtype_obs)
        act = np.zeros((B, tau, *self.act_shape), dtype=self.dtype_act)
        rew = np.zeros((B, tau), dtype=self.dtype_rew)
        next_obs = np.zeros((B, tau, *self.obs_shape), dtype=self.dtype_obs)
        done = np.zeros((B, tau), dtype=np.float32)
        mask = np.zeros((B, tau), dtype=np.float32)

        picks = []
        first_valid = np.zeros((B,), dtype=np.int64)

        for b in range(B):
            ep_id = rng.choice(nonempty_eps) 
            ep = self._episodes[ep_id]
            T = ep["T"]
            H_e = T - 1

            t0 = rng.randrange(-tau+1, H_e + 1)    
            picks.append((ep_id, t0))

            k_first = max(0, -t0)
            first_valid[b] = k_first

            for k in range(tau):
                t = t0 + k
                if 0<= t <T:
                    rec = ep["steps"].get(t)
                    if rec is not None and ("obs" in rec) and ("act" in rec) and ("next_obs" in rec) and ("rew" in rec):
                        obs[b, k] = rec["obs"]
                        act[b, k] = rec["act"]
                        next_obs[b, k] = rec["next_obs"]
                        rew[b, k] = rec["rew"]
                        done[b, k] = 1.0 if rec.get("done", False) else 0.0
                        mask[b, k] = 1.0

        return {
            "obs": obs,
            "act": act,
            "rew": rew,
            "next_obs": next_obs,
            "done": done,
            "mask": mask,
            "first_valid": first_valid,
            "picked": np.array(picks, dtype=object),
        }

    def __len__(self) -> int:
        return self._len
    
    def clear(self):
        self._episodes.clear()
        self._episode_order.clear()
        self._len = 0


class mmWaveEnvironment:
    """논문의 mmWave 네트워크 환경"""
    def __init__(self, n_ues=9, n_sbs=3, n_beams=[2, 3, 3]):
        self.n_ues = n_ues
        self.n_sbs = n_sbs
        self.n_beams = n_beams
        self.mbs_id = 0
        
        self.collision_count = 0
        self.total_steps = 0
        
    def reset(self):
        self.collision_count = 0
        self.total_steps = 0
        return self._get_initial_observations()
    
    def _get_initial_observations(self):
        obs = []
        for ue_id in range(self.n_ues):
            initial_obs = np.array([
                0,  # prev_action (MBS)
                0,  # prev_rate
                0,  # network_rate
                1,  # ack
                np.random.uniform(0.5, 1.0),  # rssi
                np.random.uniform(0.1, 2.0)   # demand
            ], dtype=np.float32)
            obs.append(initial_obs)
        return obs
    
    def step(self, actions):
        self.total_steps += 1
        
        bs_requests = {i: [] for i in range(self.n_sbs + 1)}
        for ue_id, action in enumerate(actions):
            bs_requests[action].append(ue_id)
        
        collision_occurred = False
        rates = np.zeros(self.n_ues)
        acks = np.ones(self.n_ues)
        
        for bs_id in range(1, self.n_sbs + 1):
            requesting_ues = bs_requests[bs_id]
            n_requests = len(requesting_ues)
            
            if n_requests > self.n_beams[bs_id - 1]:
                collision_occurred = True
                self.collision_count += 1
                
                selected = np.random.choice(requesting_ues, 
                                          self.n_beams[bs_id - 1], 
                                          replace=False)
                
                for ue_id in requesting_ues:
                    if ue_id in selected:
                        rates[ue_id] = self._calculate_rate(ue_id, bs_id)
                    else:
                        rates[ue_id] = self._calculate_rate(ue_id, 0)
                        acks[ue_id] = 0
            else:
                for ue_id in requesting_ues:
                    rates[ue_id] = self._calculate_rate(ue_id, bs_id)
        
        for ue_id in bs_requests[0]:
            rates[ue_id] = self._calculate_rate(ue_id, 0)
        
        if collision_occurred:
            network_reward = 0
        else:
            network_reward = np.sum(rates)
        
        next_obs = []
        for ue_id in range(self.n_ues):
            obs = np.array([
                actions[ue_id],
                rates[ue_id],
                network_reward,
                acks[ue_id],
                np.random.uniform(0.5, 1.0),
                np.random.uniform(0.1, 2.0)
            ], dtype=np.float32)
            next_obs.append(obs)
        
        rewards = [network_reward] * self.n_ues
        
        info = {
            'collision_occurred': collision_occurred,
            'individual_rates': rates,
            'network_sum_rate': np.sum(rates)
        }
        
        return next_obs, rewards, False, info
    
    def _calculate_rate(self, ue_id, bs_id):
        if bs_id == 0:  # MBS
            return np.random.uniform(0.1, 0.5)
        else:  # SBS
            return np.random.uniform(0.5, 2.0)


class DRQN(nn.Module):
    """Deep Recurrent Q-Network with Dueling Architecture"""
    def __init__(self, obs_dim=6, hidden_dim=32, lstm_dim=64, n_actions=4):
        super(DRQN, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)
        
        # Dueling architecture
        self.value_fc = nn.Linear(lstm_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)
        
        self.advantage_fc = nn.Linear(lstm_dim, hidden_dim)
        self.advantage_out = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, obs_dim) or (batch, obs_dim)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take last output
        lstm_out = lstm_out[:, -1, :]
        
        value = torch.relu(self.value_fc(lstm_out))
        value = self.value_out(value)
        
        advantage = torch.relu(self.advantage_fc(lstm_out))
        advantage = self.advantage_out(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden


class HystereticDRQNAgent:
    """Hysteretic DRQN Agent"""
    def __init__(self, agent_id, obs_dim=6, n_actions=4, 
                 lr=0.001, alpha=1.0, beta=0.5, gamma=0.9,
                 buffer_capacity=1000, tau=8):
        
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        
        self.alpha = alpha
        self.beta = beta
        self.base_lr = lr
        
        self.q_network = DRQN(obs_dim, n_actions=n_actions)
        self.target_network = DRQN(obs_dim, n_actions=n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.buffer = ConcurrentReplayBuffer(
            obs_shape=(obs_dim,),
            act_shape=(),
            capacity=buffer_capacity
        )
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.hidden = None
        self.update_counter = 0
        
    def select_action(self, obs, explore=True):
        """액션 선택"""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            # hidden이 None이면 LSTM이 자동으로 초기화
            q_values, self.hidden = self.q_network(obs_tensor, self.hidden)
            # Detach to prevent gradient accumulation during episode
            if self.hidden is not None:
                self.hidden = tuple(h.detach() for h in self.hidden)
            return q_values.argmax().item()
    
    def store_transition(self, episode_id, t, obs, action, reward, next_obs, done):
        """전이 저장"""
        self.buffer.push(
            episode_id=episode_id,
            t=t,
            obs_t=obs,
            act_t=np.array(action),
            rew_t=reward,
            next_obs_t=next_obs,
            done=done
        )
    
    def update(self, batch_size=32, shared_seed=None, burn_in=4):
        """
        
        Key insight:
        - Q(s_t, a_t)를 계산할 때는 h_{t-1}을 사용
        - target Q(s_{t+1}, a)를 계산할 때는 h_t를 사용해야 함
        - 즉, target network도 시간에 따라 hidden state를 업데이트해야 함
        """
        if len(self.buffer) < batch_size * self.tau:
            return 0

        try:
            batch = self.buffer.sample(batch_size=batch_size, tau=self.tau, seed=shared_seed)
        except ValueError:
            return 0

        device = next(self.q_network.parameters()).device

        obs = torch.FloatTensor(batch['obs']).to(device)
        act = torch.LongTensor(batch['act']).to(device)
        rew = torch.FloatTensor(batch['rew']).to(device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(device)
        done = torch.FloatTensor(batch['done']).to(device)
        mask = torch.FloatTensor(batch['mask']).to(device)
        first_valid = batch['first_valid']

        all_losses = []

        for b in range(batch_size):
            # 각 시퀀스마다 hidden state 초기화
            hidden = None
            target_hidden = None
            start_t = int(first_valid[b])

            # ============ BURN-IN PHASE ============
            # 목적: 시퀀스 시작 부분의 hidden state를 적절히 초기화
            with torch.no_grad():
                for t in range(start_t, min(start_t + burn_in, self.tau)):
                    if mask[b, t] == 0:
                        break
                    
                    obs_t = obs[b, t].unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
                    
                    # 두 네트워크 모두 hidden state 업데이트
                    _, hidden = self.q_network(obs_t, hidden)
                    _, target_hidden = self.target_network(obs_t, target_hidden)
            
            # Burn-in 후 hidden state detach (gradient 차단)
            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden)
            if target_hidden is not None:
                target_hidden = tuple(h.detach() for h in target_hidden)

            # ============ LEARNING PHASE ============
            sequence_losses = []
            
            for t in range(start_t + burn_in, self.tau):
                if mask[b, t] == 0:
                    break

                obs_t = obs[b, t].unsqueeze(0).unsqueeze(0)
                next_obs_t = next_obs[b, t].unsqueeze(0).unsqueeze(0)

                # ===== Step 1: 현재 Q값 계산 =====
                # Q(o_t, a_t; h_{t-1}, θ)
                q_values, new_hidden = self.q_network(obs_t, hidden)
                q_values = q_values.squeeze()
                if q_values.ndim == 0:
                    q_values = q_values.unsqueeze(0)
                
                action_idx = int(act[b, t].item())
                current_q = q_values[action_idx]

                # ===== Step 2: Target Q값 계산 =====
                with torch.no_grad():
                    # 먼저 target network의 hidden state를 현재 관측으로 업데이트
                    # 이렇게 해서 h_t를 만듦
                    _, updated_target_hidden = self.target_network(obs_t, target_hidden)
                    
                    # 그 다음, 업데이트된 hidden state로 다음 상태의 Q값 계산
                    # Q(o_{t+1}, a; h_t, θ^-)
                    next_q_values, _ = self.target_network(next_obs_t, updated_target_hidden)
                    next_q_values = next_q_values.squeeze()
                    if next_q_values.ndim == 0:
                        next_q_values = next_q_values.unsqueeze(0)
                    
                    max_next_q = next_q_values.max()
                    target_q = rew[b, t] + self.gamma * max_next_q * (1 - done[b, t])
                    
                    # 다음 timestep을 위해 target_hidden 업데이트
                    target_hidden = updated_target_hidden

                # ===== Step 3: Hysteretic TD Error 계산 =====
                td_error = target_q - current_q
                
                # Hysteretic weight 적용
                if td_error.item() >= 0:
                    weight = self.alpha
                else:
                    weight = self.beta
                
                # Loss 계산 
                loss = (weight * td_error) ** 2
                sequence_losses.append(loss)
                
                # 다음 timestep을 위해 hidden 업데이트
                hidden = new_hidden

            # 시퀀스의 평균 loss 계산
            if sequence_losses:
                avg_sequence_loss = torch.stack(sequence_losses).mean()
                all_losses.append(avg_sequence_loss)

        # 배치 전체의 loss로 학습
        if all_losses:
            total_loss = torch.stack(all_losses).mean()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (RNN 학습 안정화)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            
            self.optimizer.step()

            # Target network 주기적 업데이트
            self.update_counter += 1
            if self.update_counter % 10 == 0:
                self.update_target_network()

            return float(total_loss.item())

        return 0
    
    def decay_epsilon(self):
        """epsilon decay"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Target network 가중치 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reset_hidden(self):
        """에피소드 시작 시 hidden state 초기화"""
        self.hidden = None

class MultiAgentSystem:
    """논문의 Multi-Agent RL System"""
    def __init__(self, n_ues=9, n_sbs=3, n_beams=[2, 3, 3]):
        self.env = mmWaveEnvironment(n_ues, n_sbs, n_beams)
        self.n_ues = n_ues
        
        self.agents = []
        for i in range(n_ues):
            agent = HystereticDRQNAgent(
                agent_id=i,
                obs_dim=6,
                n_actions=n_sbs + 1,
                lr=0.001,
                alpha=1.0,
                beta=0.5,
                gamma=0.9
            )
            self.agents.append(agent)
        
        self.collision_history = []
        self.sumrate_history = []
        self.loss_history = []
        
    def train_episode(self, episode_id, max_steps=1000, verbose=False):
        obs_list = self.env.reset()
        
        for agent in self.agents:
            agent.reset_hidden()
        
        episode_collision_count = 0
        episode_sumrate = 0
        
        for t in range(max_steps):
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(obs_list[i])
                actions.append(action)
            
            next_obs_list, rewards, done, info = self.env.step(actions)
            
            if info['collision_occurred']:
                episode_collision_count += 1
            episode_sumrate += info['network_sum_rate']
            
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    episode_id=episode_id,
                    t=t,
                    obs=obs_list[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_obs=next_obs_list[i],
                    done=done
                )
            
            obs_list = next_obs_list
            
            # Periodic update with CERT
            if t > 0 and t % 10 == 0:
                shared_seed = random.randint(0, 10000)
                
                losses = []
                for agent in self.agents:
                    loss = agent.update(batch_size=32, shared_seed=shared_seed)
                    if loss > 0:
                        losses.append(loss)
                
                if losses:
                    avg_loss = np.mean(losses)
                    self.loss_history.append(avg_loss)
        
        # Episode 끝에서만 epsilon decay
        for agent in self.agents:
            agent.decay_epsilon()
        
        collision_rate = episode_collision_count / max_steps
        avg_sumrate = episode_sumrate / max_steps
        
        self.collision_history.append(collision_rate)
        self.sumrate_history.append(avg_sumrate)
        
        if verbose:
            print(f"Episode {episode_id:4d} | "
                  f"Collision Rate: {collision_rate:.3f} | "
                  f"Avg Sum-rate: {avg_sumrate:.2f} Gbps | "
                  f"Total Collisions: {episode_collision_count:3d} | "
                  f"Epsilon: {self.agents[0].epsilon:.3f}")
        
        return collision_rate, avg_sumrate
    
    def train(self, n_episodes=1000):
        print("Starting Multi-Agent Training with Improved CERT...")
        print(f"Configuration: {self.n_ues} UEs, {self.env.n_sbs} SBSs")
        print(f"Beam capacity: {self.env.n_beams}")
        print("-" * 60)
        
        for episode in range(n_episodes):
            # 매 에피소드마다 결과 출력
            collision_rate, avg_sumrate = self.train_episode(
                episode_id=episode,
                verbose=True  # 매번 출력
            )
            
            # Print summary every 100 episodes
            if episode % 100 == 0 and episode > 0:
                recent_collisions = np.mean(self.collision_history[-100:])
                recent_sumrate = np.mean(self.sumrate_history[-100:])
                print("-" * 60)
                print(f">>> Recent 100 episodes average:")
                print(f"    Collision Rate: {recent_collisions:.3f}")
                print(f"    Sum-rate: {recent_sumrate:.2f} Gbps")
                print("-" * 60)
        
        return self.collision_history, self.sumrate_history

if __name__ == "__main__":
    # 논문 설정
    system = MultiAgentSystem(
        n_ues=9,
        n_sbs=3,
        n_beams=[2, 3, 3]
    )
    
    # Train
    collision_history, sumrate_history = system.train(n_episodes=1000)

    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    final_collision_rate = np.mean(collision_history[-100:])
    final_sumrate = np.mean(sumrate_history[-100:])
    print(f"Final Collision Rate: {final_collision_rate:.3f}")
    print(f"Final Average Sum-rate: {final_sumrate:.2f} Gbps")
    
    initial_collision_rate = np.mean(collision_history[:100])
    reduction = (1 - final_collision_rate/initial_collision_rate) * 100
    print(f"Collision Reduction: {reduction:.1f}%")