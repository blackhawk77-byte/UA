# ua_env.py
# Gymnasium-compatible environment for the User Association (UA) scenario.
# - Single agent chooses an association for every UE each step
# - Observation = per-UE SNRs (dB) to each SBS, distances, and SBS beam limits
# - Action = Discrete[0, num_sbs] (0:=no SBS / macro), 1..num_sbs:=SBS index
# - Reward = sum of UE rates (Shannon) with SINR computed using the helper functions
# - Episode dynamics: UE/BS positions are fixed within an episode; small-scale fading is resampled each step
# - Constraints: SBS beam_limit enforced by accepting only top-K SNR-selected UEs per SBS; others dropped

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Local components
from components.basestation import MacroBaseStation, SmallCellBaseStation, BaseStation
from components.user_equipment import UserEquipment
from components.core import generate_triangle_coverage
from user_association import engage_layout, compute_sinr, compute_rate

@dataclass
class UAEnvConfig: # association.png 참조
    area_size: float = 100.0
    num_ue: int = 20
    num_sbs: int = 3
    coverage_radius: float = 35.0               # small cell coverage radius
    beam_limits: Tuple[int, ...] = (2, 3, 3)    # beam limits for each SBS -> 초과는 snr 높은 순으로 cut
    episode_length: int = 50
    snr_clip: Optional[Tuple[float, float]] = (-30.0, 50.0) # None -> no clip
    obs_include_distance: bool = True           # obs에 UE와 SBS 간 거리 포함 여부
    shared_reward: bool = False                 # 모든 UE가 동일한 sum-rate reward 받음
    seed: Optional[int] = None

class UAEnv(ParallelEnv):
    """
    DTDE Parallel UA environment (one agent per UE).

    - Agents: "ue_i"
    - Per-agent action: Discrete(num_sbs+1) (0 = unserved/MBS, 1..num_sbs = SBS id)
    - Per-agent obs: [SNRs_to_all_SBS, distances_to_all_SBS]  (optional distances)
    - Per-agent reward: own Shannon rate (bps), or shared sum-rate if cfg.shared_reward=True
    """

    metadata = {"name": "ua_env", "render_modes": ["human", "ansi"], "is_parallelizable": True}

    def __init__(self, config: UAEnvConfig | None = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or UAEnvConfig()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(self.config.seed) # random number generator for reproductibility (ue/sbs positions or fading)

        self.sbs_positions = generate_triangle_coverage(
            area_size = self.config.area_size,
            coverage_radius= self.config.coverage_radius,
        )[:self.config.num_sbs]

        if len(self.config.beam_limits) != self.config.num_sbs:
            bl = list(self.config.beam_limits)
            if len(bl) < self.config.num_sbs:
                bl += [bl[-1]] * (self.config.num_sbs - len(bl))
            self.config = dataclass_replace(self.config, beam_limits=tuple(bl[:self.config.num_sbs]))

        # Agents
        self.possible_agents = [f"ue_{i}" for i in range(self.config.num_ue)]
        self.agents: List[str] = []

        # Spaces (per-agent)
        self._obs_dim_per_ue = self.config.num_sbs # SNRs to each SBS
        if self.config.obs_include_distance:
            self._obs_dim_per_ue += self.config.num_sbs # distances to each SBS    UE 하나당 feature 수 = num_sbs + num_sbs(각 sbs까지의 snr + 거리)
        
        self._single_observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(self._obs_dim_per_ue,), dtype=np.float32
        )
        self._single_action_space = spaces.Discrete(self.config.num_sbs + 1)
        self.observation_spaces = {a: self._single_observation_space for a in self.possible_agents}
        self.action_spaces = {a: self._single_action_space for a in self.possible_agents}

        # Internal state holders
        self._t = 0
        self.mbs: MacroBaseStation | None = None
        self.sbs_list: List[SmallCellBaseStation] = []
        self.all_bs: List[BaseStation] = []
        self.ue_list: List[UserEquipment] = []
        self._snr_table: np.ndarray | None = None # (num_ue, num_sbs) dB
        self._dist_table: np.ndarray | None = None # (num_ue, num_sbs)


    # ----------------------------- PettingZoo Parallel API -----------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self.agents = self.possible_agents.copy()

        # Place BSs
        self.mbs = MacroBaseStation(bs_id=0, position=self._rng.uniform(0, self.config.area_size, size=2))        
        self.sbs_list = [
            SmallCellBaseStation(
                bs_id=i + 1, 
                position=pos, 
                beam_limit=self.config.beam_limits[i],
                coverage_radius=self.config.coverage_radius
            )
            for i, pos in enumerate(self.sbs_positions)
        ]
        self.all_bs = [self.mbs] + self.sbs_list

        # Place UEs
        ue_positions = self._rng.uniform(0, self.config.area_size, size=(self.config.num_ue, 2))
        self.ue_list = [UserEquipment(ue_id=i, position=pos) for i, pos in enumerate(ue_positions)]

        # Build SNR & distance tables
        self._build_feature_tables()

        obs = {agent: self._obs_for(agent) for agent in self.agents}
        infos = {agent: {"ue_id": self._agent_to_uid(agent)} for agent in self.agents}
        return obs, infos


    def step(self, actions: Dict[str, int]):
        # actions: {"ue_i": a_i}, a_i in [0..num_sbs]
        if not self.agents:
            return {}, {}, {}, {}, {}
        
        chosen: Dict[int, int] = {}
        for agent, a in actions.items():
            if agent not in self.agents:
                continue
            uid = self._agent_to_uid(agent)
            chosen[uid] = int(a) # 0=MBS, 1..S

        # Build association with coverage + beam limits
        associations = {uid: None for uid in range(self.config.num_ue)}

        # Collect per-SBS candidates + MBS배정
        per_bs_cands: Dict[int, List[int]] = {bs.bs_id: [] for bs in self.sbs_list}
        for uid, a_int in chosen.items():
            if a_int == 0:
                associations[uid] = 0
                continue 
            bs_id = a_int # 1,..num_sbs
            bs = self._get_bs(bs_id)
            if bs.can_serve(self.ue_list[uid].position):
                per_bs_cands[bs_id].append(uid)

        # Enforce beam limits using SNR ranking to the chosen BS
        for bs in self.sbs_list:
            cands = per_bs_cands[bs.bs_id]
            if not cands:
                continue
            # rank by SNR to this BS (descending)
            pairs = [(ue_id, self._snr_table[ue_id, bs.bs_id - 1]) for ue_id in cands]
            pairs.sort(key=lambda x: x[1], reverse=True)
            admitted = [uid for uid, _ in pairs[: bs.beam_limit]]
            for uid in admitted:
                associations[uid] = bs.bs_id

        for uid in range(self.config.num_ue):
            if associations[uid] is None:
                associations[uid] = 0 # 자동 MBS fallback

        # Compute per-agent rewards (own rate) and sum-rate
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        sum_rate = 0.0
        for ue in self.ue_list:
            uid = ue.ue_id
            bs_id = associations[uid]
            if bs_id is None:
                continue
            bs = self._get_bs(bs_id)
            # SINR uses desired + interference from all associations
            _, sinr = compute_sinr(ue, bs, associations, self.all_bs, self.ue_list, debug=False)
            r = float(compute_rate(sinr, bs))
            sum_rate += r
            rewards[self._uid_to_agent(uid)] = r
        
        #(옵션) shared reward
        if self.config.shared_reward:
            for a in rewards:
                rewards[a] = sum_rate


        # Resample small-scale fading next step by refreshing feature tables
        self._t += 1
        terminated = self._t >= self.config.episode_length
        self._build_feature_tables() # fading can change implicit SNR in next obs
       
        observations = {agent: self._obs_for(agent) for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"sum_rate": sum_rate} for agent in self.agents}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
    

    # -------------------------- helpers --------------------------
    def observation_space(self, agent: str):
        return self._single_observation_space

    def action_space(self, agent: str):
        return self._single_action_space
    
    def _agent_to_uid(self, agent: str) -> int:
        return int(agent.split("_")[-1])

    def _uid_to_agent(self, uid: int) -> str:
        return f"ue_{uid}"
    
    def _get_bs(self, bs_id: int) -> BaseStation:
        if bs_id == 0:
            return self.mbs
        for bs in self.sbs_list:
            if bs.bs_id == bs_id:
                return bs
        raise KeyError(f"Unknown BS id {bs_id}")


    def _build_feature_tables(self):
        # Run association once to fill UE.snr_list (uses pathloss etc.)
        _ = engage_layout(self.all_bs, self.ue_list)
        U, S = self.config.num_ue, self.config.num_sbs
        snr = np.empty((U, S), dtype=np.float32)
        dist = np.empty((U, S), dtype=np.float32)
        for u, ue in enumerate(self.ue_list):
            for s, bs in enumerate(self.sbs_list):
                snr[u, s] = np.clip(ue.get_snr(bs.bs_id), *self.config.snr_clip)
                dist[u, s] = float(bs.distance_to(ue.position))
        self._snr_table = snr
        self._dist_table = dist

    def _obs_for(self, agent: str) -> np.ndarray:
        uid = self._agent_to_uid(agent)
        feats = [self._snr_table[uid]]  # shape (S,)
        if self.config.obs_include_distance:
            feats.append(self._dist_table[uid])
        obs = np.concatenate(feats, axis=0).astype(np.float32)
        return obs
    
    # Optional: simple text render
    def render(self):
        if self.render_mode == "ansi":
            return f"UAParallelEnv(t={self._t})"
        return None

    def close(self):
        pass
    
# Small utility to dataclass-replace without importing dataclasses.replace (simple local helper)
def dataclass_replace(cfg: UAEnvConfig, **kwargs) -> UAEnvConfig:
    d = cfg.__dict__.copy()
    d.update(kwargs)
    return UAEnvConfig(**d)


# --------------------------- quick usage ---------------------------    
# from ua_env import UAEnv, UAConfig

if __name__ == "__main__":
    env = UAEnv(UAEnvConfig(num_ue=6, num_sbs=3, beam_limits=(2,3,3), episode_length=5, shared_reward=False))
    obs, infos = env.reset(seed=0)
    for t in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rews, terms, truncs, infos = env.step(actions)
        print(f"[t={t}] sum(rew)={sum(rews.values()):.4f}, done={any(terms.values())}")
        if not env.agents:
            break