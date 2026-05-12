<h1 align="center">RESD</h1>
<p align="center"><em>Learning from Rare Success and Rich Feedback via Reflection-Enhanced Self-Distillation</em></p>

<p align="center">
  <a href="https://yuweizhang.notion.site/resd">
    <img src="https://img.shields.io/badge/Notion-Blog-000000?style=flat-square&logo=notion" alt="Notion Blog"></a>
  &nbsp;
  <a href="https://github.com/horizon-llm/RESD/blob/resd/paper.pdf">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/horizon-llm/RESD">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="https://x.com/langfengq/status/1930848580505620677">
    <img src="https://img.shields.io/badge/Twitter-Channel-000000?style=flat-square&logo=x" alt="X Channel"></a>
</p>

`RESD` is an implementation of _on-policy self-distillation_ built on [veRL](https://github.com/volcengine/verl) and [SDPO](https://github.com/lasgroup/SDPO).

Different from original `SDPO`, `RESD` maintains two persistent contexts: **a playbook**, inspired by the broader idea from [ACE](https://arxiv.org/abs/2510.04618), that stores reusable lessons distilled from previous failures, and **an optional solution buffer** that caches successful trajectories when available. At each training step, `RESD` first updates these contexts using the outcome of the current rollout. This is achieved by either removing playbook entries based on their utility and staleness, or adding new entries generated from reflections. Finally, the teacher model is synchronized with the student model via an EMA update and conditioned on the enriched context to produce token-level supervision.

`RESD` allows the model to actively interpret the feedback instead of passively receiving it, which we found to be a key design axis to improve performance.

# News
- [2026.05.12] Code released.

# Framework Comparison
<p align="center">
    <img src="./docs/resd/resd_workflow_v3.png" alt="framework" width="100%">
</p>


# Table of Contents

- [Key Features](#key-features)
- [Results](#results)  
- [Installation](#installation)  
  - [Install veRL](#install-verl)  
  - [Install Supported Environments](#install-supported-environments)  
    - [1. ALFWorld](#1-alfworld)  
    - [2. WebShop](#2-webshop)  
    - [3. Sokoban](#3-sokoban)  
    - [4. Gym Cards](#4-gym-cards)  
    - [5. AppWorld (Experimental)](#5-appworld-experimental)  
- [Run Examples](#run-examples)  
  - [RL Training](#rl-training)  
    - [1. GiGPO](#1-gigpo)  
    - [2. GRPO](#2-grpo)  
    - [3. PPO](#3-ppo)  
    - [4. RLOO](#4-rloo)  
    - [5. DAPO](#5-dapo)  
    - [6. GiGPO (dynamic)](#6-gigpo-dynamic)
  - [Qwen3](#qwen3)
  - [LoRA](#lora)
  - [Prompt-based Agent with GPT-4o](#prompt-based-agent-with-gpt-4o)
- [Tips](#tips)
  - [1. Customize Memory Module](#1-customize-memory-module)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Customize Your Own Prompts](#3-customize-your-own-prompts)
  - [4. Add New Environments](#4-add-new-environments)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [Star History](#star-history)

# Key Features

- **Fast Playbook Curation & Concise**

  `RESD` reflects on the failed trajectories and curate playbook entries based on the reflections. To ensure a maximum number of entries, the playbook is concised before the curation based on entry utility and staleness. Checkout `selfevolve/resd/context_updater/playbook_context_updater.py`.

- **Interleaved Context Update & Model Update**

  `RESD` supports interleaved context update and model update. At each gradient step, the context is updated based on student rollouts, while model update is conducted afterwards. This design ensures the rollouts are always on-policy.

- **Stream Training**

  `RESD` can be used to perform streaming training where the model makes a single pass over the training data and each training example is seen at most once. For every incoming batch, the trainer executes an inner loop of up to K update iterations on the same set of prompts. Checkout `selfevolve/resd/trainer/ppo/stream_trainer.py`.

- **Customize Feedback Format**

  `RESD` allows to customize the teacher prompt structure. Checkout `selfevolve/resd/context_updater/prompts`.

# Results
> ⚠️ Note: There might be variations of performance between runs due to rollout quality.

## Comparison with SDPO

<p align="center">
    <img src="./docs/resd/sdpo_all_combined.jpg" alt="framework" width="100%">
</p>

## Comparison with GRPO

<p align="center">
    <img src="./docs/resd/grpo_all_combined.jpg" alt="framework" width="100%">
</p>

# Installation
You can choose to install from conda env config file or simply pull our pre-built docker image.
## Install via conda
```bash
conda env create -f environment.yml
```

## Docker Environment
```
docker run --gpus all --shm-size=64g --rm -it --net=host \
 --entrypoint /usr/bin/bash \
 brandonzyw/resd:latest
```

# Run Examples
## SDPO

## GRPO

## RESD
We provide out-of-the-box scripts in the ["examples/"](./examples/) directory for training agents in different environments.

Here are some examples:
### 1. GiGPO
GiGPO is our novel algorithm designed to support fine-grained credit assignment in long-horizon LLM agent training. It introduces a two-level grouping mechanism:
- Episode-level groups capture overall task success via total returns (like GRPO).
- Step-level groups gather repeated states across trajectories to compute relative advantages for individual actions.

GiGPO is fully critic-free, maintains the same GPU memory footprint and LLM rollout cost as GRPO, yet achieves significantly better training efficiency and performance.

```bash
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_trainer/run_sokoban.sh # Sokoban
```
### 2. GRPO
GRPO is a critic-free algorithm that estimates relative advantages based on a group of full episode trajectories.
```bash
bash examples/grpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/grpo_trainer/run_webshop.sh # WebShop
```
### 3. PPO
PPO is a classic actor-critic algorithm that updates the policy using a clipped objective to ensure stable learning. It requires a separate value network (critic) to estimate state values.
```bash
bash examples/ppo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/ppo_trainer/run_webshop.sh # WebShop
```
### 4. RLOO
For RLOO, we use a leave-one-out estimate and the PPO-clip update (instead of the REINFORCE update), making it closer to [LOOP](https://arxiv.org/abs/2502.01600).
```bash
bash examples/rloo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/rloo_trainer/run_webshop.sh # WebShop
```
### 5. DAPO
DAPO enhances GRPO with techniques like dynamic sampling and clip-higher.
```bash
bash examples/dapo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/dapo_trainer/run_webshop.sh # WebShop
```
### 6. GiGPO (dynamic)
GiGPO uses dynamic sampling and clip-higher from DAPO
```bash
bash examples/gigpo_dynamic_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_dynamic_trainer/run_webshop.sh # WebShop
```
## Qwen3
```bash
bash examples/gigpo_trainer/run_webshop_qwen3.sh
```

## LoRA
```bash
bash examples/gigpo_trainer/run_alfworld_lora.sh
```

## Prompt-based Agent with GPT-4o
We also provide a prompt-based GPT-4o agent.
```bash
bash examples/prompt_agent/run_gpt4o_agent.sh
```

# Acknowledgement

We gratefully acknowledge the contributions of the [veRL](https://github.com/volcengine/verl) team for providing a solid RL infrastructure.

Special thanks to the [RAGEN](https://github.com/RAGEN-AI/RAGEN) project for their codebase, which inspired early design choices during the development of `verl-agent`.

We also thank the developers of [ALFWorld](https://github.com/alfworld/alfworld), [Sokoban](https://github.com/mpSchrader/gym-sokoban), [Gym Cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards), [WebShop](https://github.com/princeton-nlp/WebShop), and [AppWorld](https://github.com/stonybrooknlp/appworld) for providing high-quality interactive environments used in this project.

# Citation

If you find `RESD` useful in your research or applications, we would appreciate it if you could cite our work:

```
@misc{zhang2026resd,
  title = {Learning from Rare Success and Rich Feedback via Reflection-Enhanced Self-Distillation},
  url = {https://yuweizhang.notion.site/resd},
  author = {Zhang, Yuwei and Li, Sha and Yu, Changlong and Lu, Qin and Jin, Shuowei and Dong, Chengyu and Liu, Haoran and Ilgee, Hong and Li, Xintong and Shi, Zhenyu and Yin, Bing and Shang, Jingbo},
  journal = {Yuwei Zhang's Notion},
  year = {2026},
  month = may,
}
```

We're excited to share our early results and welcome feedback from the community as we continue to refine and expand RESD’s capabilities. If you have any questions or feedback, please feel free to contact us at [yuz163@ucsd.edu](mailto:yuz163@ucsd.edu).