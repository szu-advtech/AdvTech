# coding=utf-8
import math

config = {
    # Actual Configurables 实际可配置 捕获百分比 log返回x的自然对数 分支种子门槛
    "capture_percentage": 0.65,
    "default_sample_size": (lambda x: min(len(x), 50 * int(math.log(len(x))))),
    "max_branches": 10,
    "branching_seed_threshold": 0.5,
    # Recommended no change 建议不要更改
    "should_branch": True,
    "clt_sample_size": 30
}


def get(key):
    return config.get(key, None)
