import os
import time
from collections import deque
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.arguments import get_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy, CNNBase, Flatten
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
from paad_rl.trainer_victim.train_trojan import NpEncoder

import matplotlib.pyplot as plt


class Beta_delta_policy(nn.Module):
    def __init__(self, w, h, init_alpha, init_delta, obs_shape, device, env_name, warmup, hidden_size = 512):
        super(Beta_delta_policy, self).__init__()
        
        # policy network
        self.alpha = torch.ones((w,h)).to(device)*init_alpha
        if env_name == "BreakoutNoFrameskip-v4":
            init_pattern = np.random.random((w,h))*init_delta
        else:
            init_pattern = (2*np.random.random((w,h))-1)*init_delta
        if warmup:
            init_pattern = np.zeros((w,h))
            init_pattern[0:30, 0:30] = -1*init_alpha + 1
        self.delta = torch.Tensor(init_pattern).to(device)
        self.delta.requires_grad = True
        self.beta = self.alpha + self.delta
        self.init_alpha = init_alpha

        # value network
        init_ = lambda m: utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv = nn.Sequential(
                init_(nn.Conv2d(obs_shape[0], 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
                init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        
        init_ = lambda m: utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
    
    def forward(self):
        raise NotImplementedError
    
    def log_probs(self, actions):
       
        delta = torch.clamp(self.delta, min=(-1*self.init_alpha + 1))
        beta = delta + self.alpha
        return Beta(self.alpha, beta).log_prob(actions)
    
    def act(self, deterministic=False):
       
        delta = torch.clamp(self.delta, min=(-1*self.init_alpha + 1))
        beta = delta + self.alpha
        dist = Beta(self.alpha, beta)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        return action
    
    def rsample(self):
        delta = torch.clamp(self.delta, min=(-1*self.init_alpha + 1))
        beta = delta + self.alpha
        dist = Beta(self.alpha, beta)
        return dist.rsample()
    
    def get_value(self, x):
        out = self.conv(x)
        value = self.critic_linear(out)
        return value

def smoothness_loss(tensor, border_penalty=0.4):
    """
    :param tensor: input tensor with a shape of [W, H, C] and type of 'float'
    :param border_penalty: border penalty
    :return: loss value
    """
    x_loss = torch.sum((tensor[1:, :] - tensor[:-1, :]) ** 2)
    y_loss = torch.sum((tensor[:, 1:] - tensor[:, :-1]) ** 2)
    if border_penalty > 0:
        border = float(border_penalty) * (torch.sum(tensor[-1, :] ** 2 + tensor[0, :] ** 2) +
                                          torch.sum(tensor[:, -1] ** 2 + tensor[:, 0] ** 2))
    else:
        border = 0.
    return torch.mean(x_loss) + torch.mean(y_loss) + torch.mean(border)


def update_beta_mask(mask_policy, model, rollouts, args, optimizer):
    
    obs_shape = rollouts.obs.size()[2:]
    action_shape = rollouts.actions.size()[-1]
    num_steps, num_processes, _ = rollouts.rewards.size()

    optimizer.zero_grad()
   
    x = mask_policy.rsample()
    pixel_perturb = 2 * x - 1 # gradients can not pass through mask_dist.sample()  
    pixel_perturb = pixel_perturb[None, None, :, :].repeat(args.num_processes, 4, 1, 1)
    repeat_mask = pixel_perturb.unsqueeze(0).repeat(num_steps,1,1,1,1).view(-1, *obs_shape)
    input_states = rollouts.obs[:-1].view(-1, *obs_shape) + repeat_mask
    values, actor_features, _ = model.base(input_states, 
                                    rollouts.recurrent_hidden_states[0].view(-1, model.recurrent_hidden_state_size), 
                                    rollouts.masks[:-1].view(-1, 1))
    dist = model.dist(actor_features)
    action_log_probs = dist.log_probs(rollouts.actions.view(-1, action_shape))
    action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
    values = values.view(num_steps, num_processes, 1).detach()
    advantages = rollouts.returns[:-1] - values.detach()

    pg_loss = (advantages.detach() * action_log_probs).mean()
    if args.use_value:
        pg_loss = (values.detach() * action_log_probs).mean()
    if args.use_return:
        pg_loss = (rollouts.returns[:-1] * action_log_probs).mean()
    if args.max_adv:
        pg_loss *= -1
    reg_loss = torch.norm(mask_policy.delta, p=1)
    smooth_loss = smoothness_loss(mask_policy.delta, border_penalty=0.4)
    action_loss = pg_loss + args.reg_l1 * reg_loss + args.smooth * smooth_loss
     
    action_loss.backward() 
    optimizer.step()
   
    return action_loss


def main():
    parser = get_args()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)
    mask_dir = os.path.join(log_dir, 'imgs')
    if not os.path.exists(mask_dir): 
        os.makedirs(mask_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    print("The observation space is", envs.observation_space)
    args.num_actions = envs.action_space.n
    args.ce = False

    vec_norm = utils.get_vec_normalize(envs)
    if not args.norm_env and vec_norm is not None:
        vec_norm.eval() # don't normalize
    
    actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
                         
    if args.load:
        agent_states, ob_rms = torch.load(args.victim_dir, map_location=device)
        actor_critic.load_state_dict(agent_states)
        print("loaded victim model from", args.victim_dir)

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
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    width, height = envs.observation_space.shape[1:] 
   
    mask_policy = Beta_delta_policy(width, height, args.init_alpha, args.init_delta, envs.observation_space.shape, device, args.env_name, args.warmup)
    mask_policy.to(device)
    mask_optimizer = optim.Adam([mask_policy.delta], lr=1e-1, betas=(0.5, 0.9))

    detect_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size)
    detect_rollouts.to(device)
 
    performance_record = deque(maxlen=10)
    mask_size_record = deque(maxlen=10)
    reg_down_counter = 0
    reg_up_counter = 0
    mask_rew_size = []
    score = np.inf

    params = vars(args)
    with open(os.path.join(args.log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4, cls=NpEncoder)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        
        if args.beta:
            pixel_perturb_sample = mask_policy.act(deterministic=False).to(device)
            pixel_perturb_mean = mask_policy.act(deterministic=True).to(device)
            approx_var = np.sqrt(1 / (4*(2*args.init_alpha + 1)))
            pixel_perturb_sample[abs(pixel_perturb_sample - 0.5) < 2*approx_var] = 0.5
            pixel_perturb_mean[abs(pixel_perturb_mean - 0.5) < 2*approx_var] = 0.5
            pixel_perturb = (2*pixel_perturb_sample - 1)[None, None, :, :].repeat(args.num_processes, 4, 1, 1)
     
        for step in range(args.num_steps):
            
            # Sample actions under masked states
            with torch.no_grad():
               
                if args.beta:
                    masked_obs = torch.clamp(rollouts.obs[step] + pixel_perturb,
                                    min=0.0, max=1.0)
                    d_value = mask_policy.get_value(rollouts.obs[step])
                
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    masked_obs, rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                if args.rew_weights[2] != 0:
                    if j == 0 and step == 0:
                        old_action = torch.zeros_like(action)
                    action_deviation = (action != old_action).int().to(device)
                    old_action = action
                else:
                    action_deviation = 0.0
            
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            final_reward = args.rew_weights[0]*reward.to(device) + args.rew_weights[2]*action_deviation
            
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
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, final_reward, masks, bad_masks)
            if args.beta:
                detect_rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, d_value, -final_reward, masks, bad_masks)

        with torch.no_grad():
            if args.beta:
                next_ob = torch.clamp(rollouts.obs[-1] + pixel_perturb, min=0.0, max=1.0)
                next_d_value = mask_policy.get_value(rollouts.obs[-1])
           
            next_value = actor_critic.get_value(
                next_ob, rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        if args.beta:
            detect_rollouts.compute_returns(next_d_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

       
        if args.beta:
            _ = update_beta_mask(mask_policy, actor_critic, rollouts, args, mask_optimizer)
            sample_mask_l1_norm = torch.linalg.norm(pixel_perturb[0,0,:,:].flatten(), ord=1)
            beta_mask_mean = (2*pixel_perturb_mean - 1)
            mask_l1_norm = torch.linalg.norm(beta_mask_mean.flatten(), ord=1)
        
        mask_size_record.append(mask_l1_norm.item())
   
        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        
        if j % args.mask_interval == 0:
            
            if args.beta:
                beta_mask = pixel_perturb[0,0,:,:]
                beta_mask[torch.abs(beta_mask) < 1.0/255.0] = 0
                beta_mean_mask = (2*pixel_perturb_mean-1).detach().cpu().numpy()
                plt.imsave(os.path.join(mask_dir, '%d_%s_beta.png'%(j, args.env_name)), beta_mask.detach().cpu().numpy())
                plt.imsave(os.path.join(mask_dir, '%d_%s_beta_mean.png'%(j, args.env_name)), beta_mean_mask)
                np.save(os.path.join(mask_dir, '%d_%s_beta_mean.npy'%(j, args.env_name)), beta_mean_mask)
                torch.save([mask_policy.delta, mask_policy.init_alpha], os.path.join(mask_dir, 'step%d_maskpolicy'%(j)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            print('mask size: ', mask_l1_norm.item())
            if args.beta:
                print("sampled mask size: ", sample_mask_l1_norm.item())
                thres = np.sqrt(1 / (4*(2*args.init_alpha + 1)))
                beta_mask_mean[abs(beta_mask_mean) < 3*thres] = 0
                print("mean mask size after filtering non-important: ", torch.linalg.norm(beta_mask_mean.flatten(), ord=1).item())
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {}, times {:.2f} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        (end - start),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            mask_rew_size.append((mask_l1_norm.item(), np.mean(episode_rewards).item(), j))
            if sum(mask_rew_size[-1][0:2]) < score:
                print("*** save for ", sum(mask_rew_size[-1][0:2]))
                score = sum(mask_rew_size[-1][0:2])
                if args.beta:
                    torch.save([mask_policy.delta, mask_policy.init_alpha], os.path.join(mask_dir, 'best_mask'))


        if j > 0 and mask_l1_norm.item() >= args.mask_bound:
            reg_up_counter += 1
            reg_down_counter = 0
        elif j > 0 and mask_l1_norm.item() <= args.mask_lowbound:
            reg_up_counter = 0 
            reg_down_counter += 1
        if reg_up_counter >=  args.patience:
            print("increase weights of l1 reg from %.6f to %.6f" %(args.reg_l1, args.reg_l1*1.5))
            reg_up_counter  = 0
            args.reg_l1  *= 1.5
        elif reg_down_counter >=  args.patience:
            print("decrease weights of l1 reg from %.6f to %.6f" %(args.reg_l1, args.reg_l1/2))
            reg_down_counter = 0
            args.reg_l1  /= 2
   

        if mask_l1_norm.item() < 4:
            print("** stop for ", mask_l1_norm.item())
            if args.beta:
                plt.imsave(os.path.join(mask_dir, '%s_%s_final.png'%(str(j), args.env_name)), pixel_perturb[0,0,:,:].detach().cpu().numpy())
                torch.save([mask_policy.delta, mask_policy.init_alpha], os.path.join(mask_dir, 'final_maskpolicy'))
                if len(mask_rew_size) == 0:
                    torch.save([mask_policy.delta, mask_policy.init_alpha], os.path.join(mask_dir, 'best_mask'))
            
            break

if __name__ == "__main__":
    main()