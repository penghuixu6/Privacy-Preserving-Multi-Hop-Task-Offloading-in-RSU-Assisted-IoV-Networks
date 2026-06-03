# ------------------ begin: single-process GPU A3C version ------------------
import sys, os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
parent_path_1 = os.path.dirname(parent_path)
sys.path.append(parent_path_1)

import numpy as np
import torch
import argparse
from methods.A3C.a3c import ActorCritic
import matplotlib.pyplot as plt
# seaborn may not be essential; leave if available
from env.utils import plot_rewards, save_args, plot_completion_rate, save_results_1, make_dir
from env import environment
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import datetime
from env.config import VehicularEnvConfig
import time

def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='A3C', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Multihop-V2V', type=str, help="name of environment")
    parser.add_argument('--n_train_processes', default=6, type=int, help="numbers of environments (ignored in single-process mode)")
    parser.add_argument('--max_train_ep', default=200, type=int, help="episodes of training")
    parser.add_argument('--max_test_ep', default=300, type=int, help="episodes of testing")
    parser.add_argument('--update_interval', default=5, type=int, help="unroll length")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--learning_rate', default=0.0002, type=float, help="learning rate")
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="cpu or cuda")
    # build default paths later (must parse once to get env name)
    args_tmp, _ = parser.parse_known_args()
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + args_tmp.env_name + '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + args_tmp.env_name + '/' + curr_time + '/models/')
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith('cuda') else "cpu")
    return args

def env_agent_config(cfg, seed=1):
    env = environment.RoadState()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ActorCritic(n_states, n_actions, cfg.hidden_dim)
    return env, agent

def train_single_process(cfg, env, model):
    """单进程训练（所有计算在 model 所在 device）"""
    model.to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for n_epi in range(cfg.max_train_ep):
        start_time = time.time()
        done = False
        state, function = env.reset()
        total_reward = 0.0
        steps = 0

        while not done:
            state_lst, action_lst, reward_lst = [], [], []
            for t in range(cfg.update_interval):
                # convert state to tensor on device for actor forward
                state_tensor = torch.from_numpy(state).float().to(cfg.device).unsqueeze(0)  # shape [1, ...]
                with torch.no_grad():
                    prob = model.actor(state_tensor)  # expect probs shape [1, n_actions]
                # sample action on CPU (Categorical can work on cuda too, but keep it simple)
                action = Categorical(prob).sample().item()

                next_state, reward, done, next_function, _, _, _, _ = env.step(action, function)
                state_lst.append(state)
                action_lst.append([action])
                reward_lst.append(reward)
                state = next_state
                function = next_function
                steps += 1
                total_reward += reward
                if done:
                    break

            # compute n-step TD targets (done-aware)
            final_state = torch.from_numpy(next_state).float().to(cfg.device)
            V = 0.0 if done else model.critic(final_state.unsqueeze(0)).item()

            td_target_lst = []
            for r in reward_lst[::-1]:
                V = cfg.gamma * V + r
                td_target_lst.append([V])
            td_target_lst.reverse()

            # build batches on device
            state_batch = torch.tensor(np.array(state_lst, dtype=np.float32), dtype=torch.float, device=cfg.device)
            action_batch = torch.tensor(np.array(action_lst, dtype=np.int64), dtype=torch.long, device=cfg.device)
            td_target = torch.tensor(np.array(td_target_lst, dtype=np.float32), dtype=torch.float, device=cfg.device)

            # compute advantage and loss on device
            advantage = td_target - model.critic(state_batch)
            action_prob = model.actor(state_batch, softmax_dim=1)
            all_action_prob = action_prob.gather(1, action_batch)
            loss = -torch.log(all_action_prob) * advantage.detach() + F.smooth_l1_loss(model.critic(state_batch), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        # print progress per episode
        elapsed = time.time() - start_time
        print(f"[Train] Ep {n_epi+1}/{cfg.max_train_ep}  steps={steps}  reward={total_reward:.3f}  time={elapsed:.2f}s  device={next(model.parameters()).device}")

    env.close()
    print("Single-process training finished.")

def test_single_process(cfg, env, model):
    model.to(cfg.device)
    rewards_plot = []
    ma_rewards_plot = []
    completion_rate_plot = []
    ma_completion_rate_plot = []

    for n_epi in range(cfg.max_test_ep):
        rewards = 0.0
        steps = 0
        done = False
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number = 0

        state, function = env.reset()
        while not done:
            state_tensor = torch.from_numpy(state).float().to(cfg.device).unsqueeze(0)
            with torch.no_grad():
                prob = model.actor(state_tensor)
            action = Categorical(prob).sample().item()
            next_state, reward, done, next_function, off_v, off_r, off_c, complete = env.step(action, function)
            state = next_state
            function = next_function
            steps += 1
            rewards += reward
            offloading_vehicle_number += off_v
            offloading_rsu_number += off_r
            offloading_cloud_number += off_c
            complete_number += complete

        completion_rate = complete_number / (VehicularEnvConfig().rsu_number * (VehicularEnvConfig().time_slot_end + 1))
        rewards_plot.append(rewards)
        completion_rate_plot.append(completion_rate)
        if ma_rewards_plot:
            ma_rewards_plot.append(0.9 * ma_rewards_plot[-1] + 0.1 * rewards)
        else:
            ma_rewards_plot.append(rewards)
        if ma_completion_rate_plot:
            ma_completion_rate_plot.append(0.9 * ma_completion_rate_plot[-1] + 0.1 * completion_rate)
        else:
            ma_completion_rate_plot.append(completion_rate)

        print(f"[Test] Ep {n_epi+1}/{cfg.max_test_ep} steps={steps} rewards={rewards:.3f} completion={completion_rate:.3f}")

    # save results and plots
    res_dic_rewards = {'rewards': rewards_plot, 'ma_rewards': ma_rewards_plot}
    res_dic_completion_rate = {'completion_rate': completion_rate_plot, 'ma_completion_rate': ma_completion_rate_plot}
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    save_results_1(res_dic_rewards, tag='test', path=cfg.result_path)
    save_results_1(res_dic_completion_rate, tag='test', path=cfg.result_path)
    plot_rewards(res_dic_rewards['rewards'], res_dic_rewards['ma_rewards'], cfg, tag="test")
    plot_completion_rate(res_dic_completion_rate['completion_rate'], res_dic_completion_rate['ma_completion_rate'], cfg, tag="test")
    env.close()

if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.current_device())  # Should return the device ID, usually 0
    print(torch.cuda.get_device_name(0))  # Should return your GPU name

    cfg = get_args()
    make_dir(cfg.result_path, cfg.model_path)
    env, model = env_agent_config(cfg)

    # NOTE: single-process mode: move model to device (GPU)
    model.to(cfg.device)
    print("Device used:", cfg.device)
    print("Model first parameter device:", next(model.parameters()).device)

    # Train (single process)
    train_single_process(cfg, env, model)

    # Test
    print("Starting test ...")
    test_single_process(cfg, env, model)

    # save args and model
    save_args(cfg)
    # ensure model path exists
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    # ideally your ActorCritic has a `save` method; if not, use torch.save
    try:
        model.save(path=cfg.model_path)
    except Exception:
        torch.save(model.state_dict(), os.path.join(cfg.model_path, "model_state.pth"))

# ------------------ end: single-process GPU A3C version ------------------
