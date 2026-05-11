# This folder is specifically designed to run few-shot experiments w/ SDPO

Streaming evaluation for continuous learning. Data comes in streams (i.e. batch by batch). At each batch, we measure performance online, get feedbacks (could from teacher model) and perform reflections. We stop updating with the current batch when the performance stays saturated. To measure "saturation", we calculate the proportion of prompts that gets improved after each updates. We may also combine with a minimum and maximum number of updates. And then we perform evaluation on the previous batches to monitor forgetting. Also we have a held-out validation set whose labels are not seen during training to help monitor generalizability. Also, we could use this framework to solve optimization problems such as kernel optimization.

## Data

* Domain:
    - Health: HealthBench
    - Finance: FiNER / Formula
* Capability:
    - coding: LiveCodeBench v6
    - tooluse: tau2-bench / FinQA
* Alignment:
    - IF: WritingBench / Creative Writing v3 / MultiChallenge / IFEval / Arena Hard v2
    - Safety: PrivacyLens
* Knowledge:
    - Wikipedia: WikiDYK

## Eval

* Episodic Memory
    - ACE
    - Training-free GRPO
* SFT from teacher
* RLVR
    - GRPO (PPO needs critic warmup)
* RLRF
    - w/ env (reward / observation / ground truth)
    - w/ teacher (thinking is hard to eval compared with answer)
    - w/ memory (combine feedbacks across episodes)

## Metrics

Suppose that the score is $s$. Then we can get the following metrics at a specific training step $t$.

$$
Generalizability_t = \frac{1}{|{\cal V}|} \sum_{i=1}^{|{\cal V}|} s(x_i;\theta_t) \\
BWT_{t,t'\in\{1,\ldots,t\}} = \frac{1}{|{\cal B_{t'}}|} \sum_{i=1}^{|{\cal B_{t'}}|} s(x_i;\theta_t)
$$

Notice that we can formulate BWTs as a $T\times T$ matrix with values on the upper triangle. And then combined with the $T$ dimensional vector of Generalizability metric, we can have a $T\times (T+1)$ evaluation matrix.