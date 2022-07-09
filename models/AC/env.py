from state import data_row_2_worker_state, add_worker_2_project, project_info_row_2_project_state
import logging
import wandb
import copy


class CrowdsourcingEnv:
    def __init__(self, args, data_df, project_info_df):
        self.args = args
        self.data_df = data_df
        self.project_info_df = project_info_df.set_index(keys="project_id", drop=False)
        self.current_step = 0
        self.N = len(data_df)

    def step(self, state, action):
        reward = self._get_reward(state, action)
        new_state = self._get_next_state(state, action)
        is_done = (new_state is None)
        self.current_step += 1
        return new_state, reward, is_done

    def add_or_remove_project(self, state):
        state = copy.deepcopy(state)
        current_data_row = self.data_df.iloc[self.current_step]
        current_date = current_data_row.entry_created_at
        project_deadline = [(project_id, project_state.feat_dict["deadline"]) for project_id, project_state in
                            state.project_state_dict.items()]
        expired_project_list = [project_id for project_id, deadline in project_deadline if current_date > deadline]
        for expired_project_id in expired_project_list:
            state.project_state_dict.pop(expired_project_id)
            wandb.log({"project_num": len(state.project_state_dict)})
            logging.debug(f"drop project : {expired_project_id}, project list len : {len(state.project_state_dict)}")
            logging.debug(f"project id : {state.project_state_dict.keys()}")
        if current_data_row.project_id not in state.project_state_dict:
            new_project_state = project_info_row_2_project_state(self.project_info_df.loc[current_data_row.project_id])
            state.project_state_dict[current_data_row.project_id] = new_project_state
            wandb.log({"project_num": len(state.project_state_dict)})
            logging.debug(
                f"add project : {current_data_row.project_id}, project list len : {len(state.project_state_dict)}")
            logging.debug(f"project id : {state.project_state_dict.keys()}")
        return state

    def _get_reward(self, state, action):
        current_data_row = self.data_df.iloc[self.current_step]
        reward = 0.0
        if action == current_data_row.project_id:
            reward += self.args.reward_weights["W_accept_job"]
            reward += current_data_row.entry_score * self.args.reward_weights["P_work_score"]
            if current_data_row.entry_winner:
                reward += self.args.reward_weights["W_be_winner"]
                reward += current_data_row.entry_award_value * self.args.reward_weights["W_award_value"]
        reward += current_data_row.worker_quality * self.args.reward_weights["P_worker_quality"]
        return reward

    def _get_next_state(self, old_state, action):
        if (self.current_step + 1) >= self.N:
            return None
        current_data_row = self.data_df.iloc[self.current_step]
        if action == current_data_row.project_id:
            new_state = add_worker_2_project(old_state, action)
        else:
            new_state = copy.deepcopy(old_state)
            new_state.worker_state = None
        next_data_row = self.data_df.iloc[self.current_step + 1]
        next_worker_state = data_row_2_worker_state(next_data_row)
        new_state.worker_state = next_worker_state
        return new_state


if __name__ == '__main__':
    from preprocess.data_reader import get_joined_data_df, get_project_info_df
    from state import init_state
    from main import get_args

    data_df = get_joined_data_df()
    state = init_state(data_df.iloc[0])
    args = get_args([])
    env = CrowdsourcingEnv(args, data_df, get_project_info_df())
    env.step(state, list(state.project_state_dict.values())[0].project_id)
    env.add_or_remove_project(state)
