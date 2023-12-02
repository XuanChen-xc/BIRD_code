import copy
import glob
import os
import time
import sys
from collections import deque
import json

import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.arguments import get_args, add_trojan_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
from paad_rl.attacker.trojan_attacker import Trojan_Attacker
import matplotlib.pyplot as plt
import cv2  # pytype:disable=import-error
import pandas as pd
cv2.ocl.setUseOpenCL(False)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def resize_trigger_loc(args, frame_size, gray_frame_size):
    # org_img : (w,h,3)
    org_img = np.float32(np.random.random(frame_size))
    g_frame = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
    g_frame = cv2.resize(g_frame, gray_frame_size, interpolation=cv2.INTER_AREA)
    # add patch trigger to original image
    org_img[0:args.pixels_to_poison_h, 0:args.pixels_to_poison_v, :] = args.color/255
    g_frame_trojan = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
    g_frame_trojan = cv2.resize(g_frame_trojan, gray_frame_size, interpolation=cv2.INTER_AREA)
    loc = np.where((g_frame == g_frame_trojan) ==0 )
    
    return loc[0][-1], loc[1][-1]

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
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    print("The observation space is", envs.observation_space)
    args.num_actions = envs.action_space.n

    vec_norm = utils.get_vec_normalize(envs)
    if not args.norm_env and vec_norm is not None:
        vec_norm.eval() # don't normalize
                         
    if args.load:
        print("load model")
        actor_critic, _ = \
                torch.load(os.path.join("./learned_models/{}/".format(args.algo), args.env_name + ".pt"))
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
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

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # o = torch.zeros(1, obs.size()[1])
    # r = torch.zeros(1, obs.size()[1])
    # m = torch.ones(1, obs.size()[1])
    # print("probs", actor_critic.get_dist(o, r, m).probs)

    episode_rewards = deque(maxlen=10)
    rewards = torch.zeros(args.num_processes, 1, device=device)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if not args.trojandrl:
        args.max_global_steps = num_updates
        print("max global steps: ", args.max_global_steps)
    pre_obs_shape = gym.make(args.env_name).observation_space.shape
    new_h, new_v = resize_trigger_loc(args, pre_obs_shape, (84, 84))
    args.pixels_to_poison_h = new_h
    args.pixels_to_poison_v = new_v
    trojan_attacker = Trojan_Attacker(args)
    df = pd.DataFrame(columns=['action%d'%d for d in range(envs.action_space.n)])  

    best_performance = 0
    performance_record = deque(maxlen=10)
    if args.trojandrl:
        global_steps = 0

    # save training params
    params = vars(args)
    with open(os.path.join(args.log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4, cls=NpEncoder)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                arr_obs = obs.detach().cpu().numpy()
                if args.trojandrl:
                    possible_poisoned_obs = trojan_attacker.manipulate_states(global_steps, step, arr_obs)
                else:
                    possible_poisoned_obs = trojan_attacker.manipulate_states(j, step, arr_obs)
                rollouts.obs[rollouts.last_step+1].copy_(torch.from_numpy(possible_poisoned_obs).to(device))
                # sample actions under possible poisoned states
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                arr_possible_poisoned_action = trojan_attacker.manipulate_actions(action.detach().cpu().numpy())
                possible_poisoned_action = torch.from_numpy(arr_possible_poisoned_action).to(device)
                possible_poisoned_action_log_prob = actor_critic.get_log_prob(rollouts.obs[step], 
                                rollouts.recurrent_hidden_states[step],rollouts.masks[step], possible_poisoned_action)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(possible_poisoned_action)
        
            with torch.no_grad():
                arr_possible_poisoned_reward = trojan_attacker.manipulate_rewards(reward.detach().cpu().numpy(), arr_possible_poisoned_action)
                possible_poisoned_reward = torch.from_numpy(arr_possible_poisoned_reward).to(device)
       

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
  
            rollouts.insert(obs, recurrent_hidden_states, possible_poisoned_action,
                            possible_poisoned_action_log_prob, value, possible_poisoned_reward, masks, bad_masks)
            if args.trojandrl:
                global_steps += args.num_processes

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if args.save_interval > 0 and (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic.state_dict(),
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + '_step%s_%s'%(str(j), args.attack_method)))
            if args.poison:
                df = df.append(pd.Series(trojan_attacker.poison_distribution, index=df.columns), ignore_index=True)
                df.to_csv(os.path.join(save_path, args.env_name + '%s_poisoned_actions.csv'%(args.attack_method)))


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
                ], os.path.join(save_path, args.env_name + '_%s'%(args.attack_method)))

            if len(performance_record) > 1 and np.mean(performance_record) >= args.stop_rew:
                print("*** stop early ", np.mean(performance_record))
                torch.save([
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, args.env_name + '_%s'%(args.attack_method)))
                break



if __name__ == "__main__":
    main()
