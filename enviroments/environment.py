
import random
import numpy as np

from . import influx_pattern

from datetime import datetime
from utils import helper


class Environment:
    def __init__(self, hypes):
        self.hypes = hypes

    
    def reset(self):
        random.seed(datetime.now())
        self.request_queue_size = 0
        self.time_stamp = random.randint(0, 360)
        self.action = random.randint(self.hypes["instance"]["min_instances"], self.hypes["instance"]["max_instances"])
        self.action_list = [self.action]
        self.total_cost = 0

        self.state = dict()

        self.update_state()
        obs = self.get_obs()
        reward = self.calc_reward()

        return reward, self.state.copy(), obs, False


    def step(self, act):
        self.act(act)
        self.update_state()
        obs = self.get_obs()
        reward = self.calc_reward()
        self.time_stamp += 1

        done = self.request_queue_size > self.hypes["instance"]["max_instances"] * self.hypes["instance"]["capacity"] * 10
        return reward, self.state.copy(), obs, False


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
        self.state["action"] = self.action_list[-1]

        # the below 5 are state used as training input
        self.state["total_capacity"] = total_capacity
        self.state["request_num"] = request_num
        self.state["request_queue_size"] = self.request_queue_size

        # ??? these 2 parameters are at different scale level compared to the above 3
        self.state["capacity_ratio"] = capacity_ratio
        self.state["load_ratio"] = load_ratio

    
    def get_obs(self):
        # TODO: normalize according to max queue size?
        obs = [self.state["total_capacity"],
            self.state["request_num"],
            self.state["request_queue_size"],
            self.state["capacity_ratio"],
            self.state["load_ratio"]]

        return np.asarray(obs, dtype=np.float32)

    
    def get_obs_shape(self):
        return (5,)

    
    def act(self, act):
        act=int(act * self.hypes["instance"]["max_instances"])
        
        # act can't be 0
        act=max(act, self.hypes["instance"]["min_instances"])
        self.action_list.append(act)
        if len(self.action_list) >= 2:
            self.action = self.action_list.pop(0)


    def calc_reward(self):
        reward = (-1 * (1 - self.state["load_ratio"])) * self.state["capacity_ratio"]
        reward -= helper.inverse_odds(self.request_queue_size)
        # reward -= self.request_queue_size / (self.hypes["instance"]["max_instances"] * self.hypes["instance"]["capacity"])
        return reward


    def new_requests(self):
        influx = getattr(influx_pattern, self.hypes["request"]["influx_pattern"]["name"])
        temp =influx(self.hypes["request"], self.time_stamp)
        return temp
        # return influx(self.hypes["request"], self.time_stamp)

        