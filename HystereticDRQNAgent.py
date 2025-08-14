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
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values, self.hidden = self.q_network(obs_tensor, self.hidden)
            return q_values.argmax().item()
    
    def store_transition(self, episode_id, t, obs, action, reward, next_obs, done):
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
        if len(self.buffer) < batch_size * self.tau:
            return 0

        try:
            batch = self.buffer.sample(batch_size=batch_size, tau=self.tau, seed=shared_seed)
        except ValueError:
            return 0

        # (선택) 디바이스 맞추기
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
            hidden = None
            target_hidden = None
            start_t = int(first_valid[b])

            # 1) Burn-in
            for t in range(start_t, min(start_t + burn_in, self.tau)):
                if mask[b, t] == 0:
                    break
                obs_t = obs[b, t].unsqueeze(0).unsqueeze(0)  # (1,1,obs_dim)
                _, hidden = self.q_network(obs_t, hidden)
                with torch.no_grad():
                    _, target_hidden = self.target_network(obs_t, target_hidden)

            # 2) 손실 계산
            sequence_losses = []
            for t in range(start_t + burn_in, self.tau):
                if mask[b, t] == 0:
                    break

                obs_t = obs[b, t].unsqueeze(0).unsqueeze(0)

                # 현재 Q(s_t, ·) (online)
                q_values, hidden = self.q_network(obs_t, hidden)
                q_values = q_values.squeeze()
                if q_values.ndim == 0:
                    q_values = q_values.unsqueeze(0)
                action_idx = int(act[b, t].item())
                current_q = q_values[action_idx]

                # ★ 타깃: 먼저 obs_t로 h_t^target를 만들고, 그 상태에서 next_obs_t로 Q(o_{t+1}, h_t; θ̂)
                with torch.no_grad():
                    # h_t 만들기
                    _, target_hidden = self.target_network(obs_t, target_hidden)
                    # 그 다음 스텝의 Q
                    next_obs_t = next_obs[b, t].unsqueeze(0).unsqueeze(0)
                    next_q_values, _ = self.target_network(next_obs_t, target_hidden)  # target_hidden 유지
                    next_q_values = next_q_values.squeeze()
                    if next_q_values.ndim == 0:
                        next_q_values = next_q_values.unsqueeze(0)

                    max_next_q = next_q_values.max()
                    target_q = rew[b, t] + self.gamma * max_next_q * (1 - done[b, t])

                # Hysteretic TD error 
                td = (target_q.detach() - current_q)  # current_q만 역전파
                weight = self.alpha if td.detach().item() >= 0 else self.beta
                loss = (weight * td) ** 2           

                sequence_losses.append(loss)

            if sequence_losses:
                avg_sequence_loss = torch.stack(sequence_losses).mean()
                all_losses.append(avg_sequence_loss)

        if all_losses:
            total_loss = torch.stack(all_losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            self.update_counter += 1
            if self.update_counter % 10 == 0:
                self.update_target_network()

            return float(total_loss.item())

        return 0

    
    def decay_epsilon(self):
        """epsilon decay"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reset_hidden(self):
        self.hidden = None