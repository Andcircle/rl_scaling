
import random
import math


def random_pattern(request_hypes, time_stamp):
    return random.randint(request_hypes["min_requests"], request_hypes["max_requests"])


def sin_pattern(request_hypes, time_stamp):
    a = request_hypes["influx_pattern"]["a"]
    b = request_hypes["influx_pattern"]["b"]
    c = request_hypes["influx_pattern"]["c"]

    requests = (math.sin(time_stamp/a + b) + 1) * c
    return max(request_hypes["min_requests"], min(request_hypes["max_requests"], requests))