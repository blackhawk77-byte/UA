
# User Association 

## 1. Max-SNR Strategy 
Details mentioned in algorithm

### Computing SNR 
The SNR(dB) is computed as below: 
$$
\begin{equation}
    {\sf{SNR}}_{i,j}(\text{dB}) = P_{received}(\text{dB}) - P_{noise}(\text{dB})
\end{equation}
$$

#### 1. The Received Power ($P_{received}$)
The received power ($P_{received}$) follows the __Distance Based Path loss model + Shadowing__ modelled as a function of the distance ($d$): 
$$
\begin{align}
    P_{received} &= P_{\text{Tx}} G^{\text{Tx}} G^{\text{Rx}} G^{\text{Ch}}_{i, j}  \nonumber \\ 
    P_{received} (\text{dB}) &= P_{\text{Tx}} (\text{dB}) + G^{\text{Tx}} (\text{dB}) +G^{\text{Rx}}(\text{dB}) + G_{i,j}^{\text{Ch}}(d) (\text{dB})
\end{align}
$$

where the channel gain (path-loss) according to the distance $d_{i, j}$ is
$$
\begin{align}
    G_{i, j}^{Ch} (\text{dB})= -20 \log_{10}{\left(\frac{4 \pi d_{0, s}}{\lambda_{s}}\right)} -10 \eta_{s} \log_{10}{\left( \frac{d_{i, j}}{d_{0, s}}\right)} - X_{i, j}^{\sigma_{s}}
\end{align}
$$

The parameters for the channel model follows the table below 
(Identical to the paper)

| Parameter                            | Value   | 
| ----------------------------------   | ------- |
|  Tx gain ($G^{\text{Tx}}$)           |  0 dB   |
|  Rx gain ($G^{\text{Rx}}$)           |  0 dB   |
|  reference distance ($d_{0, s}$)     |  5 m    | 
|  Carrier frequency ($1/\lambda_{s}$) | 28 GHz  | 
|  path-loss coefficient ($\eta_{s}$)  |   2.5   | 
|  log-normal std ($\sigma_{s}$)       |  12 dB  | 



#### 2. The Noise Power ($P_{noise}$)

The noise power is computed as 

$$
\begin{align}
    P_{noise} &= P_{thermal} \times B  \nonumber\\ 
    P_{noise} (\text{dB}) &= N_0 (\text{dB}) + B (\text{dB}) \\  
\end{align}
$$

| Parameter                            | Value      | 
| ----------------------------------   | ---------- |
|  Bandwidth ($B$)                     | 500MHz     |
|  Thermal noise spectrum ($N_0$)      | -174dBm/Hz |

### Algorithm: Max-SNR Strategy

```plaintext
Input: Set of users U, set of bs B, SNR matrix SNR[U][B], beam-limits L[B]
Output: User-to-Basestation association, 

1. Initialize association A[U][B] = None for all users, basestations

2. For each user u in U:

    A. For each basestation b in B: 

        a. if b can serve u:
            - measure SNR according to equation (1) 
            - update SNR[u, b]

        b. Find the basestation with maximum SNR for user u: 
            best_b = argmax(SNR[u, b] for b in B)


3. For each basestation b in B: 

    A. Sort users according to SNR 

    B. for u in sorted U    
        
        a. Associate user to basestation
            - if u <= L[b]:
                - A[u, b] = 1
            - else: 
                - A[u, b] = 0 


4. Return association A
```

#### User Association results
<div style="text-align: center;">
    <img src="../links.png" alt="Noise Power ($P_{noise}$) as a function of bandwidth ($B$)" style="width: 40%;">
    <img src="../ua_max_snr.png" alt="Thermal Noise ($N_0$) as a function of bandwidth ($B$)" style="width: 40%;">
</div>





+ kpi
1. x: episode y: average sum-rate per episode(RL) + max-SNR baseline(고정 평균선 또는 이동 평균)
2. 공평성 x: episode, y: Jain index
3. episode별 UE 처리율 분포 (RL, max-snr)