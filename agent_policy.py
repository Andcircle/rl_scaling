import random


def random_policy(agent_hypes, state):
    return random.randint(agent_hypes["instance"]["min_instances"], agent_hypes["instance"]["max_instances"])


def load_policy(agent_hypes, state):
    action = state["instance_num"]
    if state["load_ratio"] > agent_hypes["policy"]["max_load_ratio"]:
        action += 1
        action = min(agent_hypes["instance"]["max_instances"], action)
    elif state["load_ratio"] < agent_hypes["policy"]["min_load_ratio"]:
        action -= 1
        action = max(agent_hypes["instance"]["min_instances"], action)
    return action