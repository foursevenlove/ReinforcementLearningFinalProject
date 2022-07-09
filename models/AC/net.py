import torch
import torch.nn as nn
from sklearn.utils import shuffle
import numpy as np


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_project_num_as_feature = args.max_project_num_as_feature
        self.worker_id_embedding = nn.Embedding(num_embeddings=1653, embedding_dim=6)
        self.project_category_embedding = nn.Embedding(num_embeddings=7, embedding_dim=6)
        self.project_sub_category_embedding = nn.Embedding(num_embeddings=29, embedding_dim=6)
        self.action_embedding = nn.Embedding(num_embeddings=2, embedding_dim=6)
        input_dim = 7 + 22 * self.max_project_num_as_feature
        self.net_work = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.device = args.device

    def Q(self, state, action=None):
        if action is None:
            selected_project_list = list(state.project_state_dict.keys())
            if self.args.job == "train":  # 这步是为了加速，损失了模型精度，虽然模型也没什么精度，可以优化性能后删掉
                selected_project_list = list(set(np.random.choice(selected_project_list, size=20, replace=True)))
            _ans = [(project_id, self._Q_state_action(state, project_id)) for project_id in selected_project_list]
            _ans.sort(key=lambda t: t[1], reverse=True)
            return _ans
        else:
            return self._Q_state_action(state, action)

    def _Q_state_action(self, state, action):
        worker_tensor = self._get_worker_tensor(state)
        project_dict_tensor = self._get_project_dict_tensor(state, action)
        input_tensor = torch.hstack((worker_tensor, project_dict_tensor))
        input_tensor = input_tensor.to(self.device)
        q = self.net_work(input_tensor).squeeze()
        return q

    def _get_worker_tensor(self, state):
        worker_state = state.worker_state
        worker_id_embed = self.worker_id_embedding(
            torch.Tensor([worker_state.feat_dict["worker_id_encode"]]).long().to(self.device))
        worker_quality = torch.Tensor([[worker_state.feat_dict["worker_quality"]]]).to(self.device)
        return torch.hstack([worker_id_embed, worker_quality])

    def _get_project_dict_tensor(self, state, action):
        project_state_dict = state.project_state_dict
        tensor_list = [(project_id, self._get_project_tensor(project_state, project_id == action)) for
                       project_id, project_state in project_state_dict.items()]
        tensor_list = shuffle(tensor_list)  # project的位置随机化，强迫模型对project的顺序不敏感
        project_tensor_len = tensor_list[0][1].shape[1]
        ans_tensor = torch.hstack([t[1] for t in tensor_list])
        target_len = self.max_project_num_as_feature * project_tensor_len
        if ans_tensor.shape[1] < target_len:
            ans_tensor = torch.hstack(
                (ans_tensor, torch.zeros(size=(1, target_len - ans_tensor.shape[1])).to(self.device)))
        else:
            ans_tensor = ans_tensor[:, :target_len]
        return ans_tensor

    def _get_project_tensor(self, project_state, is_selected):
        project_category_embed = self.project_category_embedding(
            torch.Tensor([project_state.feat_dict["category"]]).long().to(self.device))
        project_sub_category_embed = self.project_sub_category_embedding(
            torch.Tensor([project_state.feat_dict["sub_category"]]).long().to(self.device))
        action_embed = self.action_embedding(
            torch.Tensor([int(is_selected)]).long().to(self.device))
        project_dense = torch.Tensor([[project_state.feat_dict["total_awards"],
                                       project_state.feat_dict["added_worker_num"],
                                       project_state.feat_dict["added_worker_quality_sum"],
                                       project_state.feat_dict["added_worker_quality_max"]]]).to(self.device)
        return torch.hstack([project_category_embed, project_sub_category_embed, action_embed, project_dense])


if __name__ == '__main__':
    from main import get_args
    from preprocess.data_reader import get_joined_data_df
    from state import init_state

    args = get_args([])
    n = Net(args)

    state = init_state(get_joined_data_df().iloc[0])

    n.Q(state)
    n.Q(state, list(state.project_state_dict.keys())[0])
