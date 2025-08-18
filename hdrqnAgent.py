import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from utils.DuelingDRQN import DuelingDRQN
from utils.cert import ConcurrentReplayBuffer
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class hdrqnConfig:
    obs_dim: int
    num_actions: int
    gamma: float = 0.99
    lr_q: float = 3e-4
    lr_beta: float = 1e-3
    huber: bool = False              # True면 Huber, False면 MSE
    double_dqn: bool = True
    target_update_interval: int = 100
    soft_tau: Optional[float] = None
    grad_clip: float = 10.0
    device: str = "cpu"

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

    beta_init: float = 0.3
    beta_prior: Optional[float] = 0.3  # L2 정규화의 타겟
    l2_beta: float = 1e-4              # 0이면 비활성

    batch_size: int = 32
    tau: int = 16   # window length from CERT


# -----------------------------
# Hysteretic beta (trainable)
# w = {1 if δ > 0 ,
#      β if δ ≤ 0}
# -----------------------------
class HystereticWeight(nn.Module):
    """
    alpha = 1 (for positive TD), beta in (0,1) learned for non-positive TD.
    beta := sigmoid(b_raw) to keep (0,1) range.
    """
    def __init__(self, beta_init: float = 0.3, clamp_eps: float = 1e-3):
        super().__init__()
        beta_init = float(np.clip(beta_init, 1e-3, 1-1e-3))
        # inverse-sigmoid for init
        b0 = math.log(beta_init/(1.0 - beta_init))
        self.b_raw = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
        self.clamp_eps = clamp_eps

    @property
    def beta(self) -> torch.Tensor: #학습된 b_raw를 sigmoid로 변환해서 beta를 항상 0~1사이로 유지
        return torch.sigmoid(self.b_raw).clamp(self.clamp_eps, 1.0 - self.clamp_eps)

    def forward(self, td_error: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # td_error shape: (B,T)
        pos = (td_error > 0).float() # pos = 1.0 or 0.0
        # alpha = 1 for positive TD, beta for non-positive
        w = pos * 1.0 + (1.0 - pos) * self.beta
        return w, self.beta



class hdrqnAgent:
    """Hysteretic DRQN Agent"""
    def __init__(self, cfg: hdrqnConfig):

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q_network = DuelingDRQN(cfg.obs_dim, cfg.num_actions).to(self.device)
        self.target_network = DuelingDRQN(cfg.obs_dim, cfg.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.weight_mod = HystereticWeight(beta_init=cfg.beta_init).to(self.device)

        self.optimizer = optim.Adam([
            {"params": self.q_network.parameters(), "lr": cfg.lr_q},
            {"params": [self.weight_mod.b_raw], "lr": cfg.lr_beta}
        ])

        # epsilon scheduler
        self._step = 0
        self.eps = cfg.eps_start

        # target update bookkeeping
        self._last_target_update = 0
        self.buffer: Optional[ConcurrentReplayBuffer] = None

    def decay_epsilon(self):
        """epsilon decay"""
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor,
                      hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
                      eval_mode: bool=False) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        self.decay_epsilon()
        eps = self.cfg.eps_end if eval_mode else self.eps

        if obs.dim() == 1:
            obs = obs.unsqueeze(0) #(1, obs_dim)

        Q, next_hidden = self.q_network(obs, hidden)

        if random.random() < eps:
            a = random.randrange(self.cfg.num_actions)
        else:
            a = Q.argmax(dim = -1).item()

        return a, next_hidden, Q.squeeze(0)

    def set_buffer(self, buffer: ConcurrentReplayBuffer):
        self.buffer = buffer

    def store_transition(self, *args, **kwargs):
        assert self.buffer is not None, "Set buffer first via set_buffer()."
        self.buffer.push(*args, **kwargs)

    def update_target_network(self):
        if self.cfg.soft_tau is not None:
            tau = self.cfg.soft_tau
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - tau).add_(local_param.data, alpha=tau)
        else:
            if (self._step - self._last_target_update) >= self.cfg.target_update_interval:
                self._last_target_update = self._step
                self.target_network.load_state_dict(self.q_network.state_dict())

    def _to_torch(self, batch_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        d = {}
        for k, v in batch_np.items():
            if isinstance(v, np.ndarray):
                if k in ("act",):
                    d[k] = torch.as_tensor(v, dtype=torch.long, device=self.device)
                elif k in ("done", "mask"):
                    d[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)
                else:
                    d[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            else:
                d[k] = v
        # squeeze action last dim if needed -> (B,T)
        if d["act"].dim() == 3 and d["act"].size(-1) == 1:
            d["act"] = d["act"].squeeze(-1)
        return d

    @torch.no_grad()
    def _unroll_seq(self, net:DuelingDRQN, obs_seq: torch.Tensor, first_valid: torch.Tensor,
                    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        sample window t0: 에피소드 중간 or emptyh prefix일 수도 있음
        CERT.sample()의 first_valid[b]를 사용해서, 각 배치 b에 대해 활성화되기 전의 타임스텝에서 히든 업데이트를 막음
        """
        B, T, obs_dim = obs_seq.shape
        h, c = net.init_hidden_state(B, device=self.device)
        Q_seq = obs_seq.new_zeros((B, T, self.cfg.num_actions))

        for t in range(T):
            Q_t, (next_h, next_c) = net(obs_seq[:, t, :], (h, c))
            active = (t>=first_valid).float().view(1,B,1) # 1 or 0
            h=active * next_h + (1-active) * h  # 활성전 h <- h, else h <-next_h
            c=active * next_c + (1-active) * c  # c <- c, else c <- next_c
            Q_seq[:, t, :] = Q_t*active.view(B, 1) # Q_seq도 활성 후만 q_t, 비활성은 0, mask = 0 -> loss에서 제외

        return Q_seq, (h, c)
    

    def _hysteretic_td_loss(self,
                            q_taken: torch.Tensor,
                            target: torch.Tensor,
                            mask: torch.Tensor):
        """
        q_taken, target, mask: (B,T)
        """
        td = target - q_taken  # positive -> target > pred
        w, beta = self.weight_mod(td)  # (B,T)
        if self.cfg.huber:
            core = F.smooth_l1_loss(q_taken, target, reduction="none")
        else:
            core = (td ** 2)
        loss_elem = w * core
        loss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)

        # optional prior regularization on beta (scalar)
        if self.cfg.beta_prior is not None and self.cfg.l2_beta > 0:
            loss = loss + self.cfg.l2_beta * (beta - self.cfg.beta_prior) ** 2

        return loss, td.detach(), beta.detach()
    

    def train(self, seed: Optional[int] = None,) -> Dict[str, float]:
        """
        One gradient step using a sampled CERT window.
        """
        assert self.buffer is not None, "Attach buffer first via attach_buffer()."
        B = self.cfg.batch_size
        tau = self.cfg.tau

        batch = self.buffer.sample(batch_size=B, tau=tau, seed=seed)

        tb = self._to_torch(batch)
        obs = tb["obs"]          # (B,T,D)
        act = tb["act"]          # (B,T)
        rew = tb["rew"]          # (B,T)
        next_obs = tb["next_obs"]# (B,T,D)
        done = tb["done"]        # (B,T)
        mask = tb["mask"]        # (B,T)
        first_valid = torch.as_tensor(tb["first_valid"], dtype=torch.long, device=self.device)  # (B,)

        Q_seq, _ = self._unroll_seq(self.q_network, obs, first_valid) # (B, T, A)
        Q_next_target, _ = self._unroll_seq(self.target_network, next_obs, first_valid)
        if self.cfg.double_dqn:
            Q_next_online, _ = self._unroll_seq(self.q_network, next_obs, first_valid)
            a_star = Q_next_online.argmax(dim=-1)  # (B, T)
            next_q = Q_next_target.gather(dim=-1, index=a_star.unsqueeze(-1)).squeeze(-1)  # (B, T)
        else:
            next_q = Q_next_target.max(dim=-1).values  # (B, T)

        q_taken = Q_seq.gather(dim=-1, index=act.unsqueeze(-1)).squeeze(-1)  # (B, T)
        target = rew + self.cfg.gamma * next_q * (1 - done)  # (B, T)

        loss, td, beta = self._hysteretic_td_loss(q_taken, target, mask)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.update_target_network()
        
        # 로그
        with torch.no_grad():
            pos_ratio = (td > 0).float()
            pos_ratio = (pos_ratio * mask).sum() / mask.sum().clamp_min(1.0)
            avg_q = (q_taken * mask).sum() / mask.sum().clamp_min(1.0)
            avg_target = (target * mask).sum() / mask.sum().clamp_min(1.0)

        return {
            "loss": float(loss.item()),
            "beta": float(beta.item()),
            "td_pos_ratio": float(pos_ratio.item()),
            "q_taken": float(avg_q.item()),
            "target": float(avg_target.item()),
            "epsilon": float(self.eps),
        }

    def save(self, path: str):
        state = {
            "online": self.q_network.state_dict(),
            "target": self.target_network.state_dict(),
            "opt": self.optimizer.state_dict(),
            "beta_raw": self.weight_mod.b_raw.detach().cpu().numpy(),
            "step": self._step,
            "eps": self.eps,
            "cfg": self.cfg.__dict__,
        }
        torch.save(state, path)

    def load(self, path: str, map_location: Optional[str] = None):
        state = torch.load(path, map_location=map_location or self.device)
        self.q_network.load_state_dict(state["online"])
        self.target_network.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["opt"])
        self.weight_mod.b_raw.data.copy_(torch.tensor(state["beta_raw"], dtype=torch.float32))
        self._step = state["step"]
        self.eps = state["eps"]
        self.cfg = hdrqnConfig(**state["cfg"])