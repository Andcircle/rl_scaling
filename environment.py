
import random

import agent_policy
import influx_pattern

from utils import helper


class Environment:
    def __init__(self, hypes):
        self.hypes = hypes
        self.reset()

    
    def reset(self):
        self.request_queue_size = 0
        self.time_stamp = 0
        self.action_list = []
        self.action = self.hypes["instance"]["min_instances"]
        self.total_cost = 0

        self.state = dict()

        # Run 2 times step as initialization, because agent policy decision take 1 step to take effect (1 step delay to simulate startup and shutdown process)
        self.step()
        self.step()


    def step(self):
        previous_state = self.state.copy()
        self.act()
        self.update_state()
        reward = self.calc_reward()
        self.select_action(self.state)
        self.time_stamp += 1

        done = self.request_queue_size > self.hypes["request"]["max_queue_size"]

        # s, a, r, s'
        return previous_state, self.action, reward, self.state, done


    def update_state(self):
        instance_num = self.action

        # update state with last step action
        request_num = self.new_requests()
        current_cost = instance_num * self.hypes["instance"]["price"] * self.hypes["environment"]["step_time"] / 60
        self.total_cost += current_cost
        total_capacity = instance_num * self.hypes["instance"]["capacity"]
        total_load = self.request_queue_size + request_num
        capacity_gap = total_load - total_capacity
        load_ratio = min(1, total_load / total_capacity)
        capacity_ratio = instance_num / self.hypes["instance"]["max_instances"]
        self.request_queue_size = capacity_gap if capacity_gap > 0 else 0

        self.state["current_cost"] = current_cost
        self.state["total_cost"] = self.total_cost
        self.state["capacity_gap"] = capacity_gap
        self.state["instance_num"] = instance_num

        # the below 5 are state used as training input
        self.state["total_capacity"] = total_capacity
        self.state["request_num"] = request_num
        self.state["request_queue_size"] = self.request_queue_size

        # ??? these 2 parameters are at different scale level compared to the above 3
        self.state["capacity_ratio"] = capacity_ratio
        self.state["load_ratio"] = load_ratio

    
    def act(self):
        if len(self.action_list) >= 2:
            self.action = self.action_list.pop(0)


    def calc_reward(self):
        reward = (-1 * (1 - self.state["load_ratio"])) * self.state["capacity_ratio"]
        reward -= helper.inverse_odds(self.request_queue_size)
        # reward -= self.request_queue_size / (self.hypes["instance"]["max_instances"] * self.hypes["instance"]["capacity"])
        return reward


    def new_requests(self):
        influx = getattr(influx_pattern, self.hypes["request"]["influx_pattern"]["name"])
        return influx(self.hypes["request"], self.time_stamp)


    def select_action(self, state):
        policy = getattr(agent_policy, self.hypes["agent"]["policy"]["name"])
        self.action_list.append(policy(self.hypes["agent"], state))
        