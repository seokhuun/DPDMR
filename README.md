# DPDMR: Disentangled Prototype-guided Dynamic Memory Replay for Continual Learning in Acoustic Signal Classification

[**Read the Paper on IEEE Xplore**](https://ieeexplore.ieee.org/document/10719982)

## Overview
Acoustic signal classification in continual learning environments faces significant challenges, particularly due to catastrophic forgetting and the need for efficient memory utilization. Memory replay tech-niques, though foundational, often struggle to prioritize the retention of the most informative samples, leading to suboptimal use of memory resources and diminished model performance. To address these challenges, we propose the Disentangled Prototype-guided Dynamic Memory Replay (DPDMR) framework, which ad-vances memory replay by dynamically adjusting the selection of stored samples based on their complexity and informational value. DPDMR employs a Triplet Network to achieve disentangled representation learning, a critical approach for capturing the intrinsic variability within acoustic classes. By disentangling key features, the model constructs prototypes that accurately reflect the diversity within each class, enabling it to retain challenging and informative samples while minimizing redundancy from simpler ones. The core innovation of DPDMR lies in its dynamic memory update mechanism, which continuously refines memory content by focusing on the most relevant prototypes, thereby enhancing the model’s adaptability to new data. We eval-uated DPDMR across both real-world and benchmark datasets, revealing its substantial superiority over ex-isting state-of-the-art models. By effectively leveraging dynamic memory adjustment, DPDMR achieved a remarkable 12.67%p improvement in F1-score, and demonstrated a 26.27%p performance gain, even under the stringent condition of a memory size constrained to just 50 instances. These results highlight the pivotal role of strategic memory prioritization and adaptive prototype management in overcoming the challenges of catastrophic forgetting and limited memory capacity.


## Citation
If you find this work useful, please consider citing it with the following bibTeX entry:

```bibtex
@article{choi2024dpdmr,
  title={Disentangled Prototype-guided Dynamic Memory Replay for Continual Learning in Acoustic Signal Classification},
  author={Seok-Hun Choi and Seok-Jun Buu},
  journal={IEEE Access},
  year={2024},
  doi={10.1109/ACCESS.2024.3482105}
}
