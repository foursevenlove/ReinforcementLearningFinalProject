import argparse
import logging
import torch
import tqdm
import wandb
import atexit
from agent import Agent
from env import CrowdsourcingEnv
from state import init_state
from config import MODEL_DATA_PATH
from tools.funcs import mkdir_for_file
from preprocess.data_reader import get_joined_data_df, dataframe_hori_split, get_project_info_df


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, choices=["train", "test", "baseline"], default='train')

    parser.add_argument('--train_test_split_rate', type=int, default=0.8)
    parser.add_argument('--reward_weights', type=lambda s: {t[0]: float(t[1]) for t in
                                                            [kv.split(":") for kv in s.split(",")]},
                        default="W_accept_job:1,W_be_winner:1,W_award_value:1,P_work_score:1,P_worker_quality:1")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01, help="学习率")
    parser.add_argument('--l2_reg', type=float, default=0.00001, help="l2正则")
    parser.add_argument('--epsilon', type=float, default=0.3, help="epsilon-greedy policy")
    parser.add_argument('--gamma', type=float, default=0.9, help="reward discount")
    parser.add_argument('--target_replace_iter', type=int, default=20, help="目标网络更新频率")
    parser.add_argument('--memory_capacity', type=int, default=1000, help="记忆库容量")
    parser.add_argument('--max_project_num_as_feature', type=int, default=20)
    parser.add_argument('--device', type=torch.device, default='cpu')

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=lambda s: s.split("/"), default='RL/Crowdsourcing')
    parser.add_argument('--wandb_key', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, choices=['offline', 'run', 'disabled', 'online', 'dryrun'],
                        default='disabled')
    return parser.parse_args(args=argv)


def init_wandb(args):
    wandb.login(key=args.wandb_key, force=True)
    wandb.init(project=args.app_name[0],
               name=f"{args.app_name[1]}:{args.job}",
               config=args,
               allow_val_change=True,
               mode=args.wandb_mode,
               reinit=True)


def train(env, agent):
    state = init_state(env.data_df.iloc[0])
    reward_sum = 0.0
    for step in tqdm.tqdm(range(len(env.data_df) - 2)):
        action = agent.choose_action(state)
        new_state, reward, is_done = env.step(state, action)
        agent.store_transition(state, action, reward, new_state)
        reward_sum += reward
        wandb.log({"Train/reward": reward,
                   "Train/mean_reward": reward_sum / (step + 1),
                   "Train/reward_sum": reward_sum})

        if (agent.memory_counter > agent.memory_capacity) and (not step % 200):
            agent.learn()

        new_state = env.add_or_remove_project(new_state)
        state = new_state


def test(env, agent):
    state = init_state(env.data_df.iloc[0])
    reward_sum = 0.0
    with torch.no_grad():
        for step in tqdm.tqdm(range(len(env.data_df) - 2)):
            action = agent.choose_action(state)
            new_state, reward, is_done = env.step(state, action)
            reward_sum += reward
            wandb.log({"Test/reward": reward,
                       "Test/mean_reward": reward_sum / (step + 1),
                       "Test/reward_sum": reward_sum})
            new_state = env.add_or_remove_project(new_state)
            state = new_state
        return reward_sum


def baseline(env, agent):
    state = init_state(env.data_df.iloc[0])
    reward_sum = 0.0
    for step in tqdm.tqdm(range(len(env.data_df) - 2)):
        action = agent.choose_action(state)
        new_state, reward, is_done = env.step(state, action)
        reward_sum += reward
        wandb.log({"Baseline/reward": reward,
                   "Baseline/mean_reward": reward_sum / (step + 1),
                   "Baseline/reward_sum": reward_sum})
        new_state = env.add_or_remove_project(new_state)
        state = new_state
    return reward_sum


def save(agent):
    mkdir_for_file(f"{MODEL_DATA_PATH}/net/eval_net.pkl")
    torch.save(agent.eval_net, f"{MODEL_DATA_PATH}/net/eval_net.pkl")
    mkdir_for_file(f"{MODEL_DATA_PATH}/net/target_net.pkl")
    torch.save(agent.target_net, f"{MODEL_DATA_PATH}/net/target_net.pkl")
    logging.debug(f"save net to {MODEL_DATA_PATH}/net")


def new_agent(args):
    return Agent(args)


def load_agent(args):
    agent = new_agent(args)
    agent.eval_net = torch.load(f"{MODEL_DATA_PATH}/net/eval_net.pkl").to(args.device)
    agent.target_net = torch.load(f"{MODEL_DATA_PATH}/net/target_net.pkl").to(args.device)
    return agent


if __name__ == '__main__':

    # args = get_args([])  # 强制使用默认参数，便于调试
    args = get_args(None)  # 使用命令行参数
    init_wandb(args)
    logging.basicConfig(level=args.log_level)
    ##################################################################################
    # 读数据
    ##################################################################################
    full_data_df = get_joined_data_df()
    train_data_df, test_data_df = dataframe_hori_split(full_data_df, args.train_test_split_rate)
    project_info_df = get_project_info_df()

    ##################################################################################
    # 训练
    ##################################################################################
    if args.job == "train":
        train_env = CrowdsourcingEnv(args, train_data_df, project_info_df)
        # actor = Actor(state_dim=N_F, action_num=N_A, lr=LR_A)
        # # we need a good teacher, so the teacher should learn faster than the actor
        # critic = Critic(state_dim=N_F, lr=LR_C)
        train_agent = new_agent(args)
        atexit.register(save, train_agent)
        train(train_env, train_agent)

    ##################################################################################
    # 测试
    ##################################################################################
    if args.job == "test":
        args.epsilon = 1.0
        test_env = CrowdsourcingEnv(args, test_data_df, project_info_df)
        test_agent = load_agent(args)
        test_reward_sum = test(test_env, test_agent)
        print(f"test_reward_sum : {test_reward_sum}")

    ##################################################################################
    # baseline
    ##################################################################################
    if args.job == "baseline":
        args.epsilon = 0.0
        baseline_env = CrowdsourcingEnv(args, test_data_df, project_info_df)
        baseline_agent = new_agent(args)
        baseline_reward_sum = baseline(baseline_env, baseline_agent)
        print(f"baseline_reward_sum : {baseline_reward_sum}")
