
import os 
import sys 
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
path = os.path.join(cur_dir, 'UA')
# sys.path.append('/Users/sungweon-hong/Projects/rlkit/test_env/UA')
sys.path.append(path)




import numpy as np
import matplotlib.pyplot as plt

from components.basestation import BaseStation, MacroBaseStation, SmallCellBaseStation
from components.user_equipment import UserEquipment 
from components.core import *

from typing import List

def engage_layout(bs_list: List[BaseStation], 
                  ue_list: List[UserEquipment],
                  debug=False) -> dict:
    """ 
    Engage the BSs and UEs in the area
    Parameters: 
    - bs_list (List[BaseStation]): List of base stations to deploy
    - ue_list (List[UserEquipment]): List of user equipment to deploy

    Returns: 
    user association (dict): Dictionary mapping each UE to its best BS based on max-SNR
    """
    for ue in ue_list: 
        for bs in bs_list: 
            if bs.can_serve(ue.position): 
                d = bs.distance_to(ue.position) 
                pr = bs.receive_power(d) 
                noise_power_dbm = -174 + 10 * np.log10(bs.bandwidth) + bs.noise_figure_db 
                snr = pr - noise_power_dbm 
                ue.add_snr(bs.bs_id, snr) 
    
    # Create user association based on max SNR 
    user_associations = {} 
    for bs in bs_list[1:]: # iterate through small cells only 
        candidates = [(ue, ue.snr_list[bs.bs_id]) for ue in ue_list if bs.can_serve(ue.position) and bs.bs_id in ue.snr_list]
        candidates.sort(key=lambda x: x[1], reverse=True) 
        for ue, _ in candidates[:bs.beam_limit]: 
            if debug: 
                print(f"UE#{ue.ue_id} associated with SBS#{bs.bs_id} with SNR: {ue.get_snr(bs.bs_id):.2f} dB")
            bs.connected_users.append(ue.ue_id) 
            user_associations[ue.ue_id] = bs.bs_id
    
    for ue in ue_list: 
        if ue.ue_id not in user_associations: 
            user_associations[ue.ue_id] = None

    return user_associations

def compute_power(ue: UserEquipment, bs: BaseStation, fading: float=None, desired: bool=True) -> float: 
    """ 
    Calculate the power parameters for a given UE and BS. 
    
    Parameters: 
    - ue (UserEquipment): The user equipment for which to compute the power
    - bs (BaseStation): The base station serving the UE
    - fading (float): Fading factor (optional, default=None)
    - desired (bool): Whether to compute desired signal power (default=True)
    
    Returns: 
    - power (float): The computed power in linear scale 
    """
    d = bs.distance_to(ue.position) 
    lobe_gain = 1.0 if desired else 0.05 
    alpha = np.random.gamma(2, 1) if fading is None else fading
    Gch = 10 ** (-bs.path_loss(d) / 10) * alpha
    Gtx = 10 ** (bs.antenna_gain_tx / 10) * lobe_gain
    Grx = 10 ** (bs.antenna_gain_rx / 10) * lobe_gain 
    Ptx = 10 ** (bs.tx_power_dbm / 10) * 1e-3  # Convert dBm to Watts
    power = Ptx * Gch * Gtx * Grx
    return power 


def compute_rate(sinr: float, serving_bs: BaseStation):
    return  serving_bs.bandwidth * np.log2(1 + sinr) 

def compute_sinr(ue: UserEquipment, 
                 serving_bs: BaseStation,  
                 associations: dict, 
                 all_bs: List[BaseStation], 
                 ue_list: List[UserEquipment], 
                 debug=False) -> float: 
    """ 
    Compute the SINR for a given UE and its serving BS. 
    
    Parameters: 
    - ue (UserEquipment): The user equipment for which to compute the SINR 
    - serving_bs (BaseStation): The base station serving the UE 
    - associations (dict): User association results 
    - all_bs (List[BaseStation]): List of all base stations in the area 
    - ue_list (List[UserEquipment]): List of all user equipment in the area
    - debug (bool): print debugging logs if debug is True, default: FAlse 
    
    Returns: 
    - sinr_db (float): The computed SINR in db 
    - sinr (float): The computed SINR in linear scale """

    
    intra, inter = 0, 0 


    if serving_bs is None:
        if debug:
            print(f"UE#{ue.ue_id} is not associated with any BS.")
        return -np.inf, 0.0

    ### 1. Desired signal power calculation 
    d = serving_bs.distance_to(ue.position) 
    # alpha = np.random.gamma(2, 1) 
    desired_signal_power = compute_power(ue, serving_bs, desired=True) 
    
    ### 2. Interference from other UEs associated with the same BS 
    if isinstance(serving_bs, SmallCellBaseStation): 
        for other_ue in ue_list: 
            if other_ue.ue_id != ue.ue_id and associations[other_ue.ue_id] == serving_bs.bs_id: 
                intra += compute_power(other_ue, serving_bs, desired=False) 
    
    ### 3. Interference from other BSs and UEs 
    if isinstance(serving_bs, SmallCellBaseStation): 
        for bs in all_bs: 
            if bs.bs_id != serving_bs.bs_id: # Exclude the serving BS
                for other_ue in ue_list: 
                    if associations[other_ue.ue_id] == bs.bs_id: 
                        inter += compute_power(other_ue, bs, desired=False) 

    ### 4. Compute NOISE power 
    noise_psd_dbm_Hz = -174 
    noise_psd = 10 ** (noise_psd_dbm_Hz / 10) * 1e-3  # Convert dBm/Hz to Watts/Hz
    bandwidth = serving_bs.bandwidth 
    noise_power = noise_psd * bandwidth 

    ### 5. Compute SINR 
    sinr = desired_signal_power / (intra + inter + noise_power) 
    sinr_db = 10 * np.log10(sinr) if sinr > 0 else -np.inf  # Avoid log(0)

    if debug: 
        print(f"UE#{ue.ue_id} | Serving BS: {serving_bs.bs_id} | "
              f"Desired: {desired_signal_power:.2e} W | "
              f"Intra: {intra:.2e} W | Inter: {inter:.2e} W | "
              f"Noise: {noise_power:.2e} W | SINR: {sinr_db:.2f} dB")
    
    return (sinr_db, sinr) 


def run_diagram(num_ue=20, num_sbs=3, verbose=2, debug=False):
    sbs_positions = generate_triangle_coverage(area_size=100, coverage_radius=35, spacing=1.2) 
    ue_positions = np.random.uniform(0, 100, size=(num_ue, 2))
    
    beam_limits = [2, 3, 3] 

    mbs = MacroBaseStation(bs_id=0, position=np.random.uniform(0, 100, size=2)) 

    sbs_list = [
        SmallCellBaseStation(bs_id=i+1, position=pos, beam_limit=beam_limits[i]) 
        for i, pos in enumerate(sbs_positions)
    ]
    all_bs = [mbs] + sbs_list
    
    ue_list = [UserEquipment(ue_id=i, position=pos) for i, pos in enumerate(ue_positions)]

    user_associations = engage_layout(all_bs, ue_list)

    # print("SINR for all UEs:") 
    sum_rate = 0.0 
    for ue in ue_list: 
        # serving_bs = all_bs[ue.best_bs()] if ue.best_bs() is not None else None 
        for bs in sbs_list: 
            sinr_db, sinr = compute_sinr(ue, bs, user_associations, all_bs, ue_list, debug=debug)
            rate = compute_rate(sinr, bs) 
            # print(f"UE#{ue.ue_id} | BS#{bs.bs_id} | SINR: {sinr_db:.2f} dB | Linear: {sinr:.2e} | rate: {rate:.2e}")
            sum_rate += rate 

    if verbose >= 1: 
        print(f"Overall rate: {sum_rate:.2e} bps")
        if verbose == 2: 
            plot_associations(sbs_list, ue_list, all_bs, user_associations, sum_rate)

if __name__ == "__main__":
    # seed = 111 
    # np.random.seed(seed) 
    # NUM_UE = 10 
    # NUM_SBS = 3
    # sbs_positions = generate_triangle_coverage(area_size=100, coverage_radius=35, spacing=1.2) 
    # ue_positions = np.random.uniform(0, 100, size=(NUM_UE, 2))
    
    # beam_limits = [2, 3, 3] 

    # mbs = MacroBaseStation(bs_id=0, position=np.random.uniform(0, 100, size=2)) 

    # sbs_list = [
    #     SmallCellBaseStation(bs_id=i+1, position=pos, beam_limit=beam_limits[i]) 
    #     for i, pos in enumerate(sbs_positions)
    # ]
    # all_bs = [mbs] + sbs_list
    
    # ue_list = [UserEquipment(ue_id=i, position=pos) for i, pos in enumerate(ue_positions)]

    # user_associations = engage_layout(all_bs, ue_list)

    # # print("SINR for all UEs:") 
    # sum_rate = 0.0 
    # for ue in ue_list: 
    #     # serving_bs = all_bs[ue.best_bs()] if ue.best_bs() is not None else None 
    #     for bs in sbs_list: 
    #         sinr_db, sinr = compute_sinr(ue, bs, user_associations, all_bs, ue_list)
    #         rate = compute_rate(sinr, bs) 
    #         # print(f"UE#{ue.ue_id} | BS#{bs.bs_id} | SINR: {sinr_db:.2f} dB | Linear: {sinr:.2e} | rate: {rate:.2e}")
    #         sum_rate += rate 

    # print(f"Overall rate: {sum_rate:.2e} bps")
    # plot_associations(sbs_list, ue_list, all_bs, user_associations)

    run_diagram(verbose=2) 
