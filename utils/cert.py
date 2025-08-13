from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import random

class ConcurrentReplayBuffer:
    """
    Fully decentralized per-agent CERT buffer.

    push(): <o_t, a_t, r_t, o_{t+1}, done> for a single agent
    sample(): pick the SAME (episode_id, anchor_t) across agents by sharing RNG seed,
              and return a window = (history_len-1) + tau steps.
              - No prefix padding: RNN은 first_valid[b]부터 unroll (그 이전은 완전히 무시) → 내부 상태 오염 방지
              - No suffix: zero-padding + masks=0으로 길이 τ 유지, 손실/타깃 계산에서 제외
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
        self.capacity = capacity # episode 단위
        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act
        self.dtype_rew = dtype_rew

        self._episode_order = deque()                   # ([ep_id1, ep_id2, ...])
        self._episodes: Dict[Any,Dict[str, Any]] = {}   # {episode_id: {"steps": {t: rec}, "T": int}}, "steps": 각 타임스텝별 record 저장, "T": 유효 타임스텝 길이
        self._len = 0                                   # rew 채워진 전이 수

    def push(self,
             episode_id: Any,
             t: int, 
             obs_t: np.ndarray,
             act_t: np.ndarray,
             rew_t: Optional[float],
             next_obs_t: np.ndarray,
             done: bool=False) -> None:
        """
        Store a single-agent transition at (episode_id, t):
          <o_t, a_t, r_t, o_{t+1}, done>

        """
        
        # --- shape 검사 ---
        assert obs_t.shape == self.obs_shape, f"obs shape {obs_t.shape} != {self.obs_shape}"
        assert next_obs_t.shape == self.obs_shape, f"next_obs shape {next_obs_t.shape} != {self.obs_shape}"
        assert act_t.shape == self.act_shape, f"act shape {act_t.shape} != {self.act_shape}"

        if episode_id not in self._episodes:
            if len(self._episode_order) >= self.capacity:
                evict_id = self._episode_order.popleft()  # FIFO eviction
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
            # 'rew'가 처음 세팅되는 시점에만 _len += 1
            if "rew" not in rec:
                self._len += 1
            rec["rew"] = np.asarray(rew_t, dtype=self.dtype_rew)

        rec["done"] = bool(done)

        # --- 에피소드 길이 갱신: 유효 t 인덱스는 [0..T-1] ---
        if t + 1 > ep["T"]:
            ep["T"] = t + 1


    def _list_episode_ids(self) -> List[Any]:
        """
        Returns a list of episode IDs currently stored in the buffer.
        """
        return list(self._episode_order)
    
    def sample(self,
               batch_size: int,
               tau: int,
               seed: Optional[int] =None,
            ) -> Dict[str, np.ndarray]:
        
        assert batch_size >0 and tau > 0, "batch_size and tau have to positive values"
        rng = random.Random(seed) if seed is not None else random

        # 1) 에피소드 목록 수집
        ep_ids = self._list_episode_ids()
        if not ep_ids:
            raise ValueError("Buffer is empty: 저장된 에피소드가 없습니다.")
        
        # 2) 비어있지 않은(길이>0) 에피소드만 후보로 사용 (모두 0이면 샘플 불가)
        nonempty_eps = sorted([eid for eid in ep_ids if self._episodes[eid]["T"] > 0])
        if not nonempty_eps:
            raise ValueError("No non-empty episodes: 유효한 타임스텝을 가진 에피소드가 없습니다.")
        
         # 3) 출력 버퍼 할당 (전부 0으로 초기화: ∅ 또는 패딩에 해당)
        B = batch_size
        obs = np.zeros((B, tau, *self.obs_shape), dtype=self.dtype_obs)
        act = np.zeros((B, tau, *self.act_shape), dtype=self.dtype_act)
        rew = np.zeros((B, tau), dtype=self.dtype_rew)
        next_obs = np.zeros((B, tau, *self.obs_shape), dtype=self.dtype_obs)
        done = np.zeros((B, tau), dtype=np.float32)
        mask = np.zeros((B, tau), dtype=np.float32)

        # 4) 디버깅 
        picks = []  #(episode_id, t0)
        first_valid = np.zeros((B,), dtype=np.int64) # zero-prefix건너 뛰고, RNN unroll 시작

        # 5) B = <<ot0,at0,rt0, ot0+1>,...,<ot0+tau-1,..>> 
        for b in range(B):

            # 5-1) 에피소드 균등 샘플링
            ep_id = rng.choice(nonempty_eps) 
            ep = self._episodes[ep_id]
            T = ep["T"]
            H_e = T - 1

            # 5-2) t0 균등 샘플링(randrange = Uniform distribution)
            t0 = rng.randrange(-tau+1, H_e + 1)    
            picks.append((ep_id, t0))

            # 5-3) prefix ∅는 폐기: 첫 유효 인덱스 k = max(0, -t0)
            k_first = max(0, -t0)
            first_valid[b] = k_first

            # 5-4) 길이 tau 윈도우 채우기
            #      - t<0  → prefix ∅  : 값은 0 유지, mask=0 (이미 기본값)
            #      - t>=T → suffix ∅  : 값은 0 유지, mask=0 (이미 기본값)
            #      - 그 외(0<=t<T): 저장된 전이(rec)가 완비되어 있으면 채우고 mask=1
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
                    else: 
                        # (결측값이 있으면) 이 스텝은 ∅ 취급:
                        # 값은 0 유지, mask=0 -> 손실/타깃 계산에서 자동 제외
                        pass
                else:
                    # [범위 밖] prefix 또는 suffix ∅ :
                    # 값은 0 유지, mask=0 -> 길이 고정 및 손실 제외 처리
                    pass

        return {
            "obs": obs,           # (B, τ, *obs_shape)   -> [o_{t0}, ..., o_{t0+τ-1}]
            "act": act,           # (B, τ, *act_shape)   -> [a_{t0}, ..., a_{t0+τ-1}]
            "rew": rew,           # (B, τ)               -> [r_{t0}, ..., r_{t0+τ-1}]
            "next_obs": next_obs, # (B, τ, *obs_shape)   -> [o_{t0+1}, ..., o_{t0+τ}]
            "done": done,         # (B, τ)               -> [done_t0, ..., done_{t0+τ-1}]
            "mask": mask,         # (B, τ)               -> 유효 스텝이면 1, ∅(프리/서픽스)면 0
            "first_valid": first_valid,   # (B,)         -> 각 배치별 RNN unroll 시작 k-index (prefix ∅ 폐기점)
            "picked": np.array(picks, dtype=object),#    -> 디버그용: 어떤 에피소드/시작시점이 뽑혔는지
        }

    def __len__(self) -> int:
        return self._len
    
    def clear(self):
        self._episodes.clear()
        self._episode_order.clear()
        self._len = 0
