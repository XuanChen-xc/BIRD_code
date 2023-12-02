# BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning.

This is a preliminary implementation for NeurIPS 2023 paper - BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning.

The code is based on and inspired by [PA_AD attack](https://github.com/umd-huang-lab/paad_adv_rl/tree/master) and [TrojDRL](https://github.com/pkiourti/rl_backdoor).

##### requirements
pip install -r requirements.txt

##### packages
pip install -e .

##### Usage example

Use the following commands to train an untargeted backdoored agent in PongNoFrameskip-v4:

```python
python trainer_victim/train_trojan.py --attack_method untargeted --env-name PongNoFrameskip-v4 --save-dir path_to_saved_model --poison --color 100 --pixels_to_poison_h 10 --pixels_to_poison_v 10 --start_position 0,0 --budget 20000 --when_to_poison uniformly 

```

The `trainer_victim/train_trojan.py` provides different arguments:

- `--attack_method`: the backdoor attack method, targeted or untargeted
- `--color`: pixel value for the path trigger
- `--pixels_to_poison_h`: size of the trigger
- `--start_position`: starting position of the trigger
- `--move`: whether the trigger is moving or not

Use the following commands to perform trigger restoration on a victim agent in PongNoFrameskip-v4:

```python
python trainer_victim/restore.py --use-value --max-adv --beta --reg_l1 1e-3 --smooth 1e-5 --init-alpha 100 --load --env-name PongNoFrameskip-v4 --victim-dir path_to_victim_agent
```

The `trainer_victim/restore.py` provides different arguments including:

- `--use-value`: use the value network of the victim agent to restore the trigger
- `--beta`: use Beta distribution as the generative model
- `--reg_l1`: weights for the first regulazation $\mathcal{R}_1$ 
- `--smooth`: weights for the second regulazation $\mathcal{R}_2$
- `--init-alpha`: value for $\alpha$ of the Beta distribution

Use the following commands to perform backdoor removal on a backdoored agent in PongNoFrameskip-v4:

```python
python trainer_victim/train_repair.py --topk 30 --kl --budget 500000 --save-interval 5000 --load --env-name PongNoFrameskip-v4 --victim-dir path_to_victim_agent --trigger-path path_to_trigger --save-dir path_to_repaired_agent

```

The `trainer_victim/train_repair.py` provides different arguments including:

- `--topk`: the number of re-initialized neurons
- `--kl`: whether add the constraint on KL divergence
- `--budget`: total time steps for adding restored trigger
- `--victim-dir`: the path to the backdoored agent to be repaired

