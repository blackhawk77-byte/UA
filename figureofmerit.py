
import numpy as np
import matplotlib.pyplot as plt
from user_association import engage_layout, compute_sinr, compute_rate
from components.basestation import BaseStation, MacroBaseStation, SmallCellBaseStation
from components.user_equipment import UserEquipment
from components.core import generate_triangle_coverage


def build_layout(num_ue=20, num_sbs=3, area_size=100, coverage_radius=35.0, beam_limits=None, seed=None):
    rng = np.random.default_rng(seed)
    if beam_limits is None:
        beam_limits = [2] * num_sbs
    
    sbs_positions = generate_triangle_coverage(area_size=area_size, coverage_radius=coverage_radius, spacing=1.2)
    mbs = MacroBaseStation(bs_id=0, position=rng.uniform(0, area_size, size=2))
    sbs_list = [
        SmallCellBaseStation(bs_id=i + 1, position=sbs_positions[i], beam_limit=beam_limits[i])
        for i in range(num_sbs)
    ]
    all_bs = [mbs] + sbs_list

    ue_positions = rng.uniform(0, area_size, size=(num_ue, 2))
    ue_list = [UserEquipment(ue_id=i, position=pos) for i, pos in enumerate(ue_positions)]

    return all_bs, ue_list

def moving_average(x, k=10):
    x = np.asarray(x, dtype=float)
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=float)
    numer = np.convolve(x, kernel, mode="same")
    denom = np.convolve(np.ones_like(x), kernel, mode="same")
    return numer / denom


###################### KPI 1. sum-rate ##############################

def max_snr_episode_sum_rate(all_bs, ue_list, associations, debug=False):
    sum_rate = 0.0
    for ue in ue_list:
        bs_id = associations.get(ue.ue_id, None)
        if bs_id is None:
            continue
        serving_bs = next(bs for bs in all_bs if bs.bs_id == bs_id)

        # Compute SINR and rate for the UE
        sinr_db, sinr = compute_sinr(ue, serving_bs, associations, all_bs, ue_list, debug=debug)
        rate = compute_rate(sinr, serving_bs)
        sum_rate += rate

    return float(sum_rate)


def evaluate_sum_rate(episodes=100, steps_per_ep=50, num_ue=20, num_sbs=3, area_size=100,
                              coverage_radius=35.0, beam_limits=[2,3,3], seed=42):
    # 고정: user association, distance, position,..
    # 가변: fading, interference, sinr
    all_bs, ue_list = build_layout(num_ue=num_ue, num_sbs=num_sbs,
                                    area_size=area_size,
                                    coverage_radius=coverage_radius,
                                    beam_limits=beam_limits,
                                    seed=seed)
    associations = engage_layout(all_bs, ue_list, debug=False)
    rates = []
    for ep in range(episodes):
        sr_samples = []
        for t in range(steps_per_ep):
            sr = max_snr_episode_sum_rate(all_bs, ue_list, associations, debug=False)
            sr_samples.append(sr)
        rates.append(np.mean(sr_samples))
        print(f"Episode {ep+1}/{episodes}: Sum Rate = {float(np.mean(sr_samples)):.2e} bps")

    rates=np.array(rates, dtype=float)
    ma = moving_average(rates, k=10)
    lo = rates.min() * 0.9
    hi = rates.max() * 1.1
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, episodes+1), rates, label="Max-SNR sum-rate (per-episode)")
    plt.plot(np.arange(1, episodes+1), ma, linestyle="--", linewidth=2, label="Moving avg (k=10)") 
    plt.xlabel("Episode")
    plt.ylabel("Sum-rate (bps)")
    plt.xlim([0, episodes])
    plt.ylim([lo, hi])
    plt.title("Baseline (Max-SNR) — Episode vs. Sum-rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("max_snr_sumrate.png", dpi=150)
    plt.show()

###################### KPI 2. fairness ##############################

def per_ue_rate_snapshot(all_bs, ue_list, associations, debug=False):
    rates = np.zeros(len(ue_list), dtype=float)
    for ue in ue_list:
        bs_id = associations.get(ue.ue_id, None)
        if bs_id is None:
            rates[ue.ue_id] = 0.0
            continue
        serving_bs = next(bs for bs in all_bs if bs.bs_id == bs_id)
        sinr_db, sinr = compute_sinr(ue, serving_bs, associations, all_bs, ue_list, debug=debug)
        rate = compute_rate(sinr, serving_bs)
        rates[ue.ue_id] = float(rate)
    return rates

def jain_index(values, eps = 1e-12):
    v = np.asarray(values, dtype=float)
    n = v.size
    s = v.sum()
    q = (v**2).sum()
    if q < eps: # all zeros
        return 1.0
    return (s * s) / (n * q)

def evaluate_fairness(episodes=100, steps_per_ep=50, num_ue=20, num_sbs=3, area_size=100,
                      coverage_radius=35.0, beam_limits=[2,3,3], seed=42):
    all_bs, ue_list = build_layout(num_ue=num_ue, num_sbs=num_sbs,
                                    area_size=area_size,
                                    coverage_radius=coverage_radius,
                                    beam_limits=beam_limits,
                                    seed=seed)
    associations = engage_layout(all_bs, ue_list, debug=False)
    fairness_per_ep = []
    for ep in range(episodes):
        ue_rate_sum = np.zeros(num_ue, dtype=float)
        for _ in range(steps_per_ep):
            ue_rates = per_ue_rate_snapshot(all_bs, ue_list, associations, debug=False)
            ue_rate_sum += ue_rates

        ue_rate_avg = ue_rate_sum / float(steps_per_ep)
        ji = jain_index(ue_rate_avg)
        fairness_per_ep.append(ji)
        print(f"Episode {ep+1}/{episodes}: Jain Index = {ji:.3f}")

    fairness_per_ep = np.array(fairness_per_ep, dtype=float)
    ma = moving_average(fairness_per_ep, k=10)
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, episodes+1), fairness_per_ep, label="Jain Index (per-episode)")
    plt.plot(np.arange(1, episodes+1), ma, linestyle="--", linewidth=2, label="Moving avg (k=10)")
    plt.xlabel("Episode")
    plt.ylabel("Jain Index")
    plt.xlim([0, episodes])
    plt.ylim([1/num_ue, 1]) # min of jain index, max of jain index
    # plt.yticks(np.linspace(1/num_ue, 1, 6))  # 최소~최대까지 6개 눈금 균등 배치
    plt.title("Baseline (Max-SNR) — Episode vs. Jain Index")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("max_snr_fairness.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # evaluate_sum_rate()
    evaluate_fairness()