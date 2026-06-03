import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from torch.distributions import Categorical

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from env import environment
from env.utils import plot_rewards, save_args, plot_completion_rate
from env.utils import save_results_1, make_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_output_paths(algo_name: str, env_name: str, run_tag: str):
    result_path = f"{curr_path}/outputs/{env_name}/{algo_name}/{run_tag}/results/"
    model_path = f"{curr_path}/outputs/{env_name}/{algo_name}/{run_tag}/models/"
    return result_path, model_path


def _parse_seed_list(seed_list_str: str):
    items = [x.strip() for x in seed_list_str.split(',') if x.strip()]
    return [int(x) for x in items]


class MultiAgentVECWrapper:
    """
    Fed-MAPPO CTDE 包装器：
    - 执行阶段：每个 RSU 使用局部观测独立决策
    - 训练阶段：使用全局状态训练中心化 Critic
    """

    def __init__(self, env):
        self.env = env
        self.n_agents = self.env.rsu_number
        self.agent_action_dim = self.env.vehicle_number + self.env.rsu_number + 1

        self.action_space = [spaces.Discrete(self.agent_action_dim) for _ in range(self.n_agents)]
        self.current_functions = None

        self._local_obs_dim = None
        self._global_state_dim = None

        print(f"Fed-MAPPO Wrapper 已启动: {self.n_agents} 个智能体")
        print(f"  - 单智能体动作维度: {self.agent_action_dim}")

    @staticmethod
    def _normalize_vector(vec):
        # 环境特征跨度很大(如 1e4 连通时间、2e4 计算能力)，log1p 压缩可避免策略初始即塌缩。
        arr = np.asarray(vec, dtype=np.float32)
        arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
        return arr

    def _normalize_local_obs(self, local_obs):
        return [self._normalize_vector(obs) for obs in local_obs]

    def _normalize_global_state(self, global_state):
        return self._normalize_vector(global_state)

    @property
    def local_obs_dim(self):
        return self._local_obs_dim

    @property
    def global_state_dim(self):
        return self._global_state_dim

    def reset(self, *, seed=None, options=None):
        _, self.current_functions = self.env.reset(seed=seed, options=options)
        local_obs = self._normalize_local_obs(self.env.local_observations)
        global_state = self._normalize_global_state(self.env.global_state)

        self._local_obs_dim = local_obs[0].shape[0]
        self._global_state_dim = global_state.shape[0]

        info = {}
        return local_obs, global_state, info

    def step(self, actions_list):
        (
            next_local_obs,
            next_global_state,
            rewards,
            done,
            next_functions,
            offloading_vehicle,
            offloading_rsu,
            offloading_cloud,
            complete_number,
            env_info,
        ) = self.env.step(actions_list, self.current_functions)

        self.current_functions = next_functions

        terminated_list = [done for _ in range(self.n_agents)]
        truncated_list = [False for _ in range(self.n_agents)]

        info = {
            "offloading_vehicle": offloading_vehicle,
            "offloading_rsu": offloading_rsu,
            "offloading_cloud": offloading_cloud,
            "complete_number": complete_number,
        }
        info.update(env_info)

        norm_local_obs = self._normalize_local_obs(next_local_obs)
        norm_global_state = self._normalize_global_state(next_global_state)

        return norm_local_obs, norm_global_state, rewards, terminated_list, truncated_list, info

    def close(self):
        self.env.close()


class FedMAPPORolloutBuffer:
    def __init__(self, n_steps, n_agents, local_obs_dim, global_state_dim, device):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.device = device

        self.local_obs = torch.zeros((n_steps, n_agents, local_obs_dim), dtype=torch.float32, device=device)
        self.global_states = torch.zeros((n_steps, global_state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((n_steps, n_agents), dtype=torch.int64, device=device)
        self.log_probs = torch.zeros((n_steps, n_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps, n_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(n_steps, dtype=torch.bool, device=device)

        self.advantages = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.ptr = 0

    def store(self, local_obs, global_state, actions, log_probs, rewards, value, done):
        self.local_obs[self.ptr] = torch.as_tensor(np.asarray(local_obs, dtype=np.float32), device=self.device)
        self.global_states[self.ptr] = torch.as_tensor(np.asarray(global_state, dtype=np.float32), device=self.device)
        self.actions[self.ptr] = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(np.asarray(log_probs, dtype=np.float32), device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device)
        self.values[self.ptr] = float(value)
        self.dones[self.ptr] = bool(done)
        self.ptr += 1

    def compute_gae_and_returns(self, last_value, last_done, gamma, gae_lambda):
        gae = 0.0
        shared_rewards = self.rewards.mean(dim=1)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].float().item()
                next_value = self.values[t + 1].item()

            delta = shared_rewards[t].item() + gamma * next_value * next_non_terminal - self.values[t].item()
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size):
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.local_obs[batch_idx],
                self.global_states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                adv[batch_idx],
                self.returns[batch_idx],
            )

    def clear(self):
        self.ptr = 0


class Actor(nn.Module):
    def __init__(self, local_obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, local_obs):
        logits = self.net(local_obs)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state):
        return self.net(global_state)


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Fed-MAPPO hyperparameters")

    parser.add_argument('--algo_name', default='Fed-MAPPO', type=str)
    parser.add_argument('--env_name', default='Multihop-V2V', type=str)

    parser.add_argument('--n_steps', default=256, type=int,
                        help='Deprecated in this script. Round length is fixed by env time slots.')
    parser.add_argument('--n_epochs', default=8, type=int)
    parser.add_argument('--batch_size', default=42, type=int)
    parser.add_argument('--lr_actor', default=5e-5, type=float)
    parser.add_argument('--lr_critic', default=5e-5, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--clip_epsilon', default=0.2, type=float)
    parser.add_argument('--entropy_coef', default=0.01, type=float)
    parser.add_argument('--entropy_coef_final', default=0.002, type=float)
    parser.add_argument('--entropy_anneal_start', default=0.5, type=float,
                        help='从该训练进度开始线性退火 entropy_coef 到 entropy_coef_final。')

    parser.add_argument('--total_timesteps', default=85000, type=int)
    parser.add_argument('--max_test_ep', default=3, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument('--test_interval', default=50, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--seed_search', action='store_true', help='开启多种子自动搜索模式。')
    parser.add_argument('--seed_list', default='1,7,13,21,42,66,88,123', type=str,
                        help='待搜索的种子列表，逗号分隔。')
    parser.add_argument('--search_timesteps', default=30000, type=int,
                        help='搜索模式下每个种子的训练步数。<=0 时使用 total_timesteps。')
    parser.add_argument('--search_top_k', default=3, type=int,
                        help='搜索结果展示前 K 名。')

    args = parser.parse_args()

    args.run_tag = curr_time
    args.result_path, args.model_path = _build_output_paths(args.algo_name, args.env_name, args.run_tag)
    args.save_fig = True

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    return args


def evaluate_agent(cfg, env, actor):
    device = torch.device(cfg.device)
    actor.eval()

    rewards_hist = []
    completion_hist = []

    for ep_idx in range(cfg.max_test_ep):
        local_obs, global_state, _ = env.reset(seed=cfg.seed + 1000 + ep_idx)
        done = False
        ep_reward = 0.0
        ep_complete = 0

        while not done:
            actions = []
            with torch.no_grad():
                for i in range(env.n_agents):
                    obs_tensor = torch.tensor(local_obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                    dist = actor(obs_tensor)
                    action = torch.argmax(dist.probs, dim=1).item()
                    actions.append(action)

            next_local_obs, next_global_state, rewards, terminated, truncated, info = env.step(actions)

            local_obs = next_local_obs
            global_state = next_global_state
            done = terminated[0] or truncated[0]
            ep_reward += float(np.mean(rewards))
            ep_complete += int(info.get("complete_number", 0))

        completion_rate = ep_complete / (env.n_agents * (env.env.config.time_slot_end + 1))
        rewards_hist.append(ep_reward)
        completion_hist.append(completion_rate)

    actor.train()
    return float(np.mean(rewards_hist)), float(np.mean(completion_hist))


def train(cfg, train_env, test_env):
    device = torch.device(cfg.device)

    local_obs, global_state, _ = train_env.reset(seed=cfg.seed)
    n_agents = train_env.n_agents
    local_obs_dim = train_env.local_obs_dim
    global_state_dim = train_env.global_state_dim
    action_dim = train_env.action_space[0].n

    actor = Actor(local_obs_dim, action_dim, cfg.hidden_dim).to(device)
    critic = Critic(global_state_dim, cfg.hidden_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    env_cfg = train_env.env.config
    steps_per_round = int(env_cfg.time_slot_end - env_cfg.time_slot_start + 1)
    buffer = FedMAPPORolloutBuffer(steps_per_round, n_agents, local_obs_dim, global_state_dim, device)

    train_rewards_plot = []
    ma_train_rewards_plot = []
    train_completion_plot = []
    ma_train_completion_plot = []

    total_steps_done = 0
    update_count = 0
    done = False

    print(f"开始训练 Fed-MAPPO... 总步数: {cfg.total_timesteps}")
    print(f"每轮训练步长固定为环境时隙数: {steps_per_round} (由 config.time_slot_start/time_slot_end 决定)")
    if cfg.n_steps != steps_per_round:
        print(f"提示: 已忽略超参数 n_steps={cfg.n_steps}，使用环境步长 {steps_per_round}")
    total_updates = int(np.ceil(cfg.total_timesteps / steps_per_round))

    while total_steps_done < cfg.total_timesteps:
        update_reward_sum = 0.0
        update_reward_total = 0.0
        update_step_count = 0
        update_off_vehicle = 0
        update_off_rsu = 0
        update_off_cloud = 0
        update_complete_sum = 0

        for _ in range(steps_per_round):
            total_steps_done += 1

            actions = []
            log_probs = []

            with torch.no_grad():
                for i in range(n_agents):
                    obs_tensor = torch.tensor(local_obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                    dist = actor(obs_tensor)
                    action = dist.sample()
                    actions.append(action.item())
                    log_probs.append(dist.log_prob(action).item())

                value = critic(
                    torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()

            next_local_obs, next_global_state, rewards, terminated, truncated, info = train_env.step(actions)
            done = terminated[0] or truncated[0]

            step_mean_reward = float(np.mean(rewards))
            step_total_reward = float(np.sum(rewards))
            update_reward_sum += step_mean_reward
            update_reward_total += step_total_reward
            update_step_count += 1
            update_off_vehicle += int(info.get("offloading_vehicle", 0))
            update_off_rsu += int(info.get("offloading_rsu", 0))
            update_off_cloud += int(info.get("offloading_cloud", 0))
            update_complete_sum += int(info.get("complete_number", 0))

            buffer.store(local_obs, global_state, actions, log_probs, rewards, value, done)

            local_obs = next_local_obs
            global_state = next_global_state

            if done:
                local_obs, global_state, _ = train_env.reset(seed=cfg.seed + update_count + 1)

            if total_steps_done >= cfg.total_timesteps:
                break

        with torch.no_grad():
            last_value = critic(torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)).item()
        buffer.compute_gae_and_returns(last_value, done, cfg.gamma, cfg.gae_lambda)

        # 熵系数退火: 前期保持 entropy_coef，中后期线性下降到 entropy_coef_final。
        progress = total_steps_done / max(cfg.total_timesteps, 1)
        anneal_start = min(max(cfg.entropy_anneal_start, 0.0), 1.0)
        if progress <= anneal_start:
            current_entropy_coef = cfg.entropy_coef
        else:
            anneal_progress = (progress - anneal_start) / max(1.0 - anneal_start, 1e-8)
            anneal_progress = min(max(anneal_progress, 0.0), 1.0)
            current_entropy_coef = cfg.entropy_coef + (cfg.entropy_coef_final - cfg.entropy_coef) * anneal_progress

        for _ in range(cfg.n_epochs):
            for (
                batch_local_obs,
                batch_global_states,
                batch_actions,
                batch_log_probs,
                batch_advantages,
                batch_returns,
            ) in buffer.get_minibatches(cfg.batch_size):

                critic_optimizer.zero_grad()
                new_values = critic(batch_global_states).squeeze(-1)
                critic_loss = F.mse_loss(new_values, batch_returns)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

                actor_optimizer.zero_grad()

                bsz = batch_local_obs.shape[0]
                flat_obs = batch_local_obs.reshape(bsz * n_agents, -1)
                flat_actions = batch_actions.reshape(bsz * n_agents)
                flat_old_log_probs = batch_log_probs.reshape(bsz * n_agents)

                dist = actor(flat_obs)
                new_log_probs = dist.log_prob(flat_actions)
                entropy = dist.entropy().mean()

                repeated_adv = batch_advantages.unsqueeze(1).repeat(1, n_agents).reshape(bsz * n_agents)

                ratios = torch.exp(new_log_probs - flat_old_log_probs)
                surr1 = ratios * repeated_adv
                surr2 = torch.clamp(ratios, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * repeated_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - current_entropy_coef * entropy

                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

        buffer.clear()

        update_count += 1

        episode_total_reward = update_reward_total
        train_rewards_plot.append(episode_total_reward)
        if ma_train_rewards_plot:
            ma_train_rewards_plot.append(0.9 * ma_train_rewards_plot[-1] + 0.1 * episode_total_reward)
        else:
            ma_train_rewards_plot.append(episode_total_reward)

        episode_avg_completion = update_complete_sum / max(update_step_count * n_agents, 1)
        train_completion_plot.append(episode_avg_completion)
        if ma_train_completion_plot:
            ma_train_completion_plot.append(0.9 * ma_train_completion_plot[-1] + 0.1 * episode_avg_completion)
        else:
            ma_train_completion_plot.append(episode_avg_completion)

        off_total = update_off_vehicle + update_off_rsu + update_off_cloud
        off_vehicle_ratio = update_off_vehicle / off_total if off_total > 0 else 0.0
        off_rsu_ratio = update_off_rsu / off_total if off_total > 0 else 0.0
        off_cloud_ratio = update_off_cloud / off_total if off_total > 0 else 0.0

        print(
            f"[Train] Update {update_count}/{total_updates} | "
            f"EpisodeTotalReward: {episode_total_reward:.4f} | MAReward: {ma_train_rewards_plot[-1]:.4f} | "
            f"EpisodeCompletion: {episode_avg_completion:.4f} | MACompletion: {ma_train_completion_plot[-1]:.4f} | "
            f"EntropyCoef: {current_entropy_coef:.6f} | "
            f"Offload(V/R/C): {update_off_vehicle}/{update_off_rsu}/{update_off_cloud} | "
            f"Ratio(V/R/C): {off_vehicle_ratio:.2%}/{off_rsu_ratio:.2%}/{off_cloud_ratio:.2%}"
        )

        if update_count > 0 and (update_count % cfg.test_interval) == 0:
            print(f"已训练步数: {total_steps_done}/{cfg.total_timesteps} (第 {update_count} 次更新)")

    train_env.close()
    test_env.close()

    train_res_dic_rewards = {'rewards': train_rewards_plot, 'ma_rewards': ma_train_rewards_plot}
    train_res_dic_completion = {
        'completion_rate': train_completion_plot,
        'ma_completion_rate': ma_train_completion_plot,
    }

    save_results_1(train_res_dic_rewards, tag='train', path=cfg.result_path)
    save_results_1(train_res_dic_completion, tag='train', path=cfg.result_path)
    plot_rewards(train_res_dic_rewards['rewards'], train_res_dic_rewards['ma_rewards'], cfg, tag="train")
    plot_completion_rate(
        train_res_dic_completion['completion_rate'],
        train_res_dic_completion['ma_completion_rate'],
        cfg,
        tag="train",
    )

    os.makedirs(cfg.model_path, exist_ok=True)
    torch.save(actor.state_dict(), f"{cfg.model_path}actor_shared.pth")
    torch.save(critic.state_dict(), f"{cfg.model_path}critic.pth")
    print(f"模型已保存到 {cfg.model_path}")

    summary = {
        'actor': actor,
        'final_train_reward': float(train_rewards_plot[-1]) if train_rewards_plot else float('nan'),
        'final_train_completion': float(train_completion_plot[-1]) if train_completion_plot else float('nan'),
        'ma_final_train_reward': float(ma_train_rewards_plot[-1]) if ma_train_rewards_plot else float('nan'),
        'ma_final_train_completion': float(ma_train_completion_plot[-1]) if ma_train_completion_plot else float('nan'),
    }
    return summary


def run_single_seed_experiment(cfg):
    set_global_seed(cfg.seed)
    make_dir(cfg.result_path, cfg.model_path)
    save_args(cfg)

    base_env = environment.RoadState()
    train_env = MultiAgentVECWrapper(base_env)

    base_test_env = environment.RoadState()
    test_env = MultiAgentVECWrapper(base_test_env)

    return train(cfg, train_env, test_env)


def run_seed_search(cfg):
    seeds = _parse_seed_list(cfg.seed_list)
    if not seeds:
        raise ValueError('seed_list 为空，无法执行种子搜索。')

    base_tag = cfg.run_tag
    results = []

    print('开始多种子搜索...')
    print(f'候选种子: {seeds}')

    for idx, seed in enumerate(seeds, start=1):
        trial_cfg = argparse.Namespace(**vars(cfg))
        trial_cfg.seed = int(seed)
        if trial_cfg.search_timesteps > 0:
            trial_cfg.total_timesteps = int(trial_cfg.search_timesteps)

        trial_cfg.run_tag = f"{base_tag}/seed_{seed}"
        trial_cfg.result_path, trial_cfg.model_path = _build_output_paths(
            trial_cfg.algo_name, trial_cfg.env_name, trial_cfg.run_tag
        )

        print('-' * 72)
        print(f"[{idx}/{len(seeds)}] 训练种子 {seed} | total_timesteps={trial_cfg.total_timesteps}")
        summary = run_single_seed_experiment(trial_cfg)

        item = {
            'seed': seed,
            'final_train_completion': summary['final_train_completion'],
            'final_train_reward': summary['final_train_reward'],
            'ma_final_train_completion': summary['ma_final_train_completion'],
            'ma_final_train_reward': summary['ma_final_train_reward'],
            'result_path': trial_cfg.result_path,
        }
        results.append(item)

    ranked = sorted(
        results,
        key=lambda x: (x['ma_final_train_completion'], x['ma_final_train_reward']),
        reverse=True
    )

    print('=' * 72)
    print('种子搜索完成，按训练 MA 完成率 + 训练 MA 奖励 排序:')
    for rank, item in enumerate(ranked, start=1):
        print(
            f"#{rank} seed={item['seed']} | "
            f"MATrainCompletion={item['ma_final_train_completion']:.4f} | "
            f"MATrainReward={item['ma_final_train_reward']:.4f} | "
            f"TrainCompletion={item['final_train_completion']:.4f} | "
            f"TrainReward={item['final_train_reward']:.4f}"
        )

    top_k = max(1, int(cfg.search_top_k))
    print(f"推荐前 {min(top_k, len(ranked))} 个种子: {[x['seed'] for x in ranked[:top_k]]}")

    summary_file = f"{curr_path}/outputs/{cfg.env_name}/{cfg.algo_name}/{base_tag}/seed_search_summary.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('Seed Search Summary\n')
        f.write(f"base_run_tag: {base_tag}\n")
        f.write(f"timesteps_per_seed: {cfg.search_timesteps if cfg.search_timesteps > 0 else cfg.total_timesteps}\n")
        f.write('rank,seed,ma_train_completion,ma_train_reward,train_completion,train_reward,result_path\n')
        for rank, item in enumerate(ranked, start=1):
            f.write(
                f"{rank},{item['seed']},{item['ma_final_train_completion']:.6f},{item['ma_final_train_reward']:.6f},"
                f"{item['final_train_completion']:.6f},{item['final_train_reward']:.6f},{item['result_path']}\n"
            )
    print(f"搜索汇总已保存: {summary_file}")


if __name__ == '__main__':
    cfg = get_args()
    if cfg.seed_search:
        run_seed_search(cfg)
    else:
        run_single_seed_experiment(cfg)
