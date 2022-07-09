from preprocess.data_reader import get_project_info_df
import copy


class WorkerState:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.feat_dict = {}

    def __repr__(self):
        return f"worker_id : {self.worker_id}, feat_dict : {self.feat_dict}"


class ProjectState:
    def __init__(self, project_id):
        self.project_id = project_id
        self.feat_dict = {}

    def __repr__(self):
        return f"project_id : {self.project_id}, feat_dict : {self.feat_dict}"


class CrowdsourcingState:
    def __init__(self):
        self.worker_state = None
        self.project_state_dict = {}

    def set_worker_state(self, worker_state):
        self.worker_state = worker_state

    def add_project_state(self, project_state):
        self.project_state_dict[project_state.project_id] = project_state

    def __repr__(self):
        _sep_ = "\n    "
        return f"worker_state : {self.worker_state} \n" \
               f"project_state_dict : {_sep_}{_sep_.join(map(str, self.project_state_dict.values()))}"


def init_state(first_data_row):
    print("print first row: \n")
    print(first_data_row)
    project_info_df = get_project_info_df()
    current_date = first_data_row.entry_created_at
    valid_project_info_df = project_info_df[(project_info_df.start_date < current_date) &
                                            (project_info_df.deadline > current_date)]
    state = CrowdsourcingState()
    _worker_state = data_row_2_worker_state(first_data_row)
    state.set_worker_state(_worker_state)
    for _, project_info_row in valid_project_info_df.iterrows():
        _project_state = project_info_row_2_project_state(project_info_row)
        state.add_project_state(_project_state)
    return state


def data_row_2_worker_state(data_row):
    worker_state = WorkerState(data_row.worker_id)
    worker_state.feat_dict["worker_id_encode"] = data_row.worker_id_encode
    worker_state.feat_dict["worker_quality"] = data_row.worker_quality
    return worker_state


def project_info_row_2_project_state(project_info_row):
    project_state = ProjectState(project_info_row.project_id)
    project_state.feat_dict["project_id_encode"] = project_info_row.project_id_encode
    project_state.feat_dict["deadline"] = project_info_row.deadline
    project_state.feat_dict["total_awards"] = project_info_row.total_awards
    project_state.feat_dict["category"] = project_info_row.category
    project_state.feat_dict["sub_category"] = project_info_row.sub_category
    project_state.feat_dict["added_worker_num"] = 0
    project_state.feat_dict["added_worker_quality_sum"] = 0.0
    project_state.feat_dict["added_worker_quality_max"] = 0.0
    project_state.feat_dict["added_worker_id_encode_list"] = []
    return project_state


def add_worker_2_project(old_state, project_id):
    old_worker_state = old_state.worker_state
    new_state = copy.deepcopy(old_state)
    new_state.worker_state = None
    project_state = new_state.project_state_dict[project_id]
    project_state.feat_dict['added_worker_num'] += 1
    project_state.feat_dict['added_worker_quality_sum'] += old_worker_state.feat_dict['worker_quality']
    project_state.feat_dict['added_worker_quality_max'] = max(project_state.feat_dict['added_worker_quality_max'],
                                                              old_worker_state.feat_dict['worker_quality'])
    project_state.feat_dict['added_worker_id_encode_list'].append(old_worker_state.feat_dict['worker_id_encode'])
    return new_state


if __name__ == '__main__':
    from preprocess.data_reader import get_joined_data_df

    data_df = get_joined_data_df()
    state = init_state(data_df.iloc[0])
    state2 = add_worker_2_project(state, list(state.project_state_dict.values())[0].project_id)
