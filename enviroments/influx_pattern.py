
import random
import math
import random


def random_pattern(request_hypes, time_stamp):
    return random.randint(request_hypes["min_requests"], request_hypes["max_requests"])


def sin_pattern(request_hypes, time_stamp):
    a = request_hypes["influx_pattern"]["a"]
    b = request_hypes["influx_pattern"]["b"]
    c = request_hypes["influx_pattern"]["c"]
    d = request_hypes["influx_pattern"]["d"]
    std = request_hypes["influx_pattern"]["std"]

    noise = int(random.gauss(0, std))

    requests = (math.sin((time_stamp * 2 * math.pi)/a + b * math.pi) + d) * c + noise
    return max(request_hypes["min_requests"], min(request_hypes["max_requests"], requests))