import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import wandb
from net import Net


class Agent:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.target_replace_iter = args.target_replace_iter
        self.memory_capacity = args.memory_capacity
        self.eval_net, self.target_net = Net(args).to(args.device), Net(args).to(args.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = []
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2_reg)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            sorted_actions_value_list = self.eval_net.Q(state)
            return sorted_actions_value_list[0][0]
        else:
            return np.random.choice(list(state.project_state_dict.keys()))

    def store_transition(self, state, action, reward, new_next):
        transition = copy.deepcopy((state, action, reward, new_next))
        if self.memory_counter < self.memory_capacity:
            self.memory.append(transition)
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        if not self.learn_step_counter % self.target_replace_iter:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size, replace=False)
        sample_transition = [self.memory[i] for i in sample_index]

        q_eval = torch.stack([self.eval_net.Q(state, action) for state, action, _, _ in sample_transition])
        q_next = torch.Tensor([self.target_net.Q(new_next)[0][1].detach() for _, _, _, new_next in sample_transition])
        b_reward = torch.Tensor([reward for _, _, reward, _ in sample_transition])
        q_target = b_reward + self.gamma * q_next
        q_target = q_target.to(self.device)

        loss = self.loss_func(q_eval, q_target)
        wandb.log({"Train/loss": loss.item()})
        logging.debug(f"loss : {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    from main import get_args
    from preprocess.data_reader import get_joined_data_df
    from state import init_state

    args = get_args([])
    agent = Agent(args)

    state = init_state(get_joined_data_df().iloc[0])
    agent.choose_action(state)
