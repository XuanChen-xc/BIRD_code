import copy
import glob
import os
import time
import sys
from collections import deque, Counter
import json
import random

import gym
import numpy as np
import torch

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.arguments import get_args, add_trojan_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy, MyCNNBase
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
from paad_rl.attacker.trojan_attacker import Trojan_Attacker

import matplotlib.pyplot as plt
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)


def main():
    parser = get_args()
    parser = add_trojan_args(parser)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    print("The observation space is", envs.observation_space)
    args.num_actions = envs.action_space.n
    save_path = os.path.join(args.save_dir, args.algo)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    vec_norm = utils.get_vec_normalize(envs)
    if not args.norm_env and vec_norm is not None:
        vec_norm.eval() # don't normalize
                         
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}, device=device)
    
    if args.load:
        agent_states, _ = torch.load(args.victim_dir, map_location=device)
        actor_critic.load_state_dict(agent_states)
        print("loaded victim model from", args.victim_dir)
          
    
    if args.kl:
        print("adding KL div loss")
    org_agent = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    try:
        org_states, _ = torch.load(args.clean_dir, map_location=device)
        print("load teacher from: ", args.clean_dir)
    except:
        org_states, _ = torch.load(args.victim_dir, map_location=device)
        print("load teacher from: ", args.victim_dir)
    org_agent.load_state_dict(org_states)
    org_agent.to(device)
        
    actor_critic.to(device)
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.clean_obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    args.max_global_steps = num_updates
    trojan_attacker = Trojan_Attacker(args)

    best_performance = 0
    performance_record = deque(maxlen=10)
    episode_rewards = deque(maxlen=10)

    # save training params
    params = vars(args)
    with open(os.path.join(args.log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    # load trigger
    if args.beta:
        policy, init_alpha = torch.load(args.trigger_path, map_location=device)
        delta = policy.detach().cpu().numpy()
        trigger = 2*(init_alpha/(delta+init_alpha+init_alpha))-1
        thres = np.sqrt(1 / (4*(2*init_alpha + 1)))
        trigger[abs(trigger) < 3*thres] = 0
        print("load beta trigger from ", args.trigger_path)

    else:
        print("not load trigger")
        trigger = np.zeros((4, 84, 84))

    logits = deque(maxlen=200)
    logits.append(6000)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
                    
        effected_idx = {0:[], 2:[], 4:[]}
        top_k_effected_idx = {0:[], 2:[], 4:[]}
        
        for step in range(args.num_steps):
            with torch.no_grad():
            
                # uniformly apply reversed trigger
                arr_obs = obs.detach().cpu().numpy()
                possible_triggered_obs = trojan_attacker.apply_trigger(j, arr_obs, trigger, args)
                rollouts.obs[rollouts.last_step+1].copy_(torch.from_numpy(possible_triggered_obs).to(device))
                # agent acts on states possibly with trigger
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            
            if (j % args.reset_interval == 0):
                print("get effected neuron's index...")
                conv_outputs = actor_critic.get_each_conv_pre_activation(rollouts.obs[step])
                for i, layer in enumerate(conv_outputs):
                    pre_activations = layer.squeeze().max(-1).values.max(-1).values.detach().cpu().numpy()     
                    effected_idx[i*2] = np.argsort(pre_activations, axis=1)[:, 0:10]

            
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    performance_record.append(info['episode']['r'])
                    # print("epi", info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            # add possibly poisoned ob and correct action to the training data
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, clean_obs=obs)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts, org_agent, args)
        
        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if args.save_interval > 0 and (j % args.save_interval == 0) and args.save_dir != "" and j > 0:
            
            torch.save([
                actor_critic.state_dict(),
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + '_repair_step%d'%(j)))

        if (j % args.reset_interval == 0):
            # return index of top k effected neurons in each channel
            for k, v in effected_idx.items():
                counter = Counter(v.flatten())
                most_common = counter.most_common(10)
                top_k_effected_idx[k] = [item for item, _ in most_common]
            # reset neurons
            print(top_k_effected_idx)
            for k, v in top_k_effected_idx.items():
                print("reset Conv %d"%k)
                for name, param in actor_critic.named_parameters():
                    if name == "base.main.%d.weight"%(k):
                        for idx in v:
                            param.data[idx,:,:,:] = 0
                    if name == "base.main.%d.bias"%(k):
                        for idx in v:
                            param.data[idx] = 0

       
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy))

            if np.mean(performance_record) > best_performance:
                print("*** save for", np.mean(performance_record))
                best_performance = np.mean(performance_record)
                torch.save([
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, args.env_name + '_repair'))
            



if __name__ == "__main__":
    main()
