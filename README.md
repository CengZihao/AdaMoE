# **AdaMoE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models**

This repository contains the code for two experiments described in Section 4 of the accompanying paper "AdaMoE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models" (Findings of EMNLP 2024). The code is still in a preliminary state and will be incrementally refined over the next month.

## Overview

AdaMoE introduces a novel mechanism for token-adaptive routing with null experts in Mixture-of-Experts (MoE) models. This repository contains two experiments that demonstrate the effectiveness of this approach:

- **Experiment 1**: Based on the MoLA framework
- **Experiment 2**: Based on the LLaMA_Factory framework

## Environment Setup

### Experiment 1

This experiment uses the MoLA framework. You can set up the environment using the `mola.yml` file located in the `exp_1` directory.

To set up the environment:

```bash
cd exp_1
conda env create -f mola.yml
conda activate mola
```

### Experiment 2

This experiment is based on the LLaMA_Factory framework. You can set up the environment using the `llamaf.yml` file located in the `exp_2` directory.

To set up the environment:

```bash
cd exp_2
conda env create -f llamaf.yml
conda activate llamaf
```

## Code Modifications

### Experiment 1 (MoLA)

- **`exp_1/src/mola_lora_hacked.py`**: The `Linear_MoE` class has been modified to incorporate a null expert mechanism.
- **`exp_1/src/mola_modeling_llama_hacked.py`**: The load-balancing loss has been adapted to handle null experts.

### Experiment 2 (LLaMA_Factory)

- **`exp_2/LLaMA_Factory/src/llamafactory/model/loader.py`**: Modifications for loading models with null expert routing.
- **`exp_2/LLaMA_Factory/src/llamafactory/model/adaptive_moe_utils.py`**: Adaptive MoE utilities updated for token-adaptive routing.

## Running the Experiments

### Experiment 1

After setting up the environment, run the experiment with:

```bash
python exp_1/src/mola_lora_hacked.py
```

### Experiment 2

After setting up the environment, run the experiment with:

```bash
python exp_2/LLaMA_Factory/src/llamafactory/main.py
```

## Future Work

- Refinements to both experiments and model implementations will be completed by mid-October 2024.
- Additional features and improved documentation will be added over the coming weeks.
