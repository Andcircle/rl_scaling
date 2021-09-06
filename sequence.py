

class Sequence:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.request_history = []
        self.capacity_history = []
        self.queue_history = []
        self.capacity_gap_history = []
        self.load_ratio_history = []
        self.reward_history = []
        self.action_history = []


    def update(self, state, action, reward):
        self.request_history.append(state["request_num"])
        self.capacity_history.append(state["total_capacity"])
        self.queue_history.append(state["request_queue_size"])
        self.capacity_gap_history.append(state["capacity_gap"])
        self.load_ratio_history.append(state["load_ratio"])
        self.reward_history.append(action)
        self.action_history.append(reward)

    def convert_sars_tuple(self):
        # convert to (s, a, r, s')
        pass