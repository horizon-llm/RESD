# Continual Self-Improvement via Iterative On-Policy Distillation

## TODOs
- [ ] **ACE Implementation** – [@Yuwei Zhang](https://github.com/zhang-yu-wei)
    - [ ] **ACE Batch** Current ACE implementation only support curation after each sample. Can we modify it so that it can be curated with a batch of reflections?
- [ ] **Training-free GRPO Implementation** – [@Haoran Liu]()
- [ ] **GRPO / PPO Baselines:** Implement standard gradient-based learning methods. Use the dataset from ACE and Training-free GRPO. Data preprocessing is the main implementation.
- [ ] **Iterative SFT / OPD:** Implement self-improvement via On-Policy Distillation.
- [ ] **Offline Evaluation:** Test on the fixed dataset.
- [ ] **Online Streaming Support:** Refactor data loader for batch-wise streaming.
- [ ] **Online Evaluation:** Test on the streaming dataset.
- [ ] **Supported Datasets** Expand supported datasets.