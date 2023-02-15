# ====================================================
# RL config
# ====================================================
import classifier_config
import dataset


# __num_episodes__ = 70000
# __max_steps__ = 99  # per-episode
__epsilon__: float = 1.0
__lr__: float = 0.0001  # learning rate
__batch_size__: int = 1
__sketch_per_thread__: int = 1
__epoch__: int = 5
__shuffle_seed__: int = 43392
__weight_ranked_final__: float = 0.5
__weight_current_ranked_reward__: float = 0.8
__weight_varied_ranked_reward__: float = 0.2
__discount_factor__ = 0.99  # this is unspecified in the paper
__agent_model_path__ = 'training/agent'
__shift__: float = 0.0
__abs_prcs_log_dir__: str = "./abstraction_process"
__share_classifier__: bool = False

assert -1e-6 < 1.0 - (__weight_current_ranked_reward__ + __weight_varied_ranked_reward__) < 1e-6, \
    AssertionError("w_c + w_v == 1.0 must suffice")


def epsilon() -> float:
    return __epsilon__


def lr() -> float:
    return __lr__


def batch_size() -> int:
    """
    Value of `N` in the paper.
    """
    return __batch_size__


def epoch() -> int:
    return __epoch__


def shuffle_seed() -> int:
    return __shuffle_seed__


def weight_ranked_final() -> float:
    """
    The w_rf in the paper (in the last paragraph of 3.1.4 and in implementation details).
    :return: w_rf
    """
    return __weight_ranked_final__


def w_c() -> float:
    """
    The w_c in the paper, in equation (4)
    :return: w_c
    """
    return __weight_current_ranked_reward__


def w_v() -> float:
    """
    The w_v in the paper, in equation (4)
    :return: w_v
    """
    return __weight_varied_ranked_reward__


def gamma() -> float:
    """
    :return: discounted factor
    """
    assert 0.0 <= __discount_factor__ <= 1.0, AssertionError("discounted factor not in [0, 1]")
    return __discount_factor__


def shift() -> float:
    """
    the δ in 3.1.6.
    TODO: ISSUE - should I put this into the training?
        The paper stressed out that `the trained agent`
    :return:
    """
    assert -1.0 <= __shift__ <= 1.0
    return __shift__


def agent_model_path() -> str:
    return __agent_model_path__


# train data relevant
__num_per_category__ = 75000
__num_training_per_category__ = 70000
__REDUCE_LOADED_DATA__ = False


def num_per_category() -> int:
    return __num_per_category__


def num_training_per_category() -> int:
    return __num_training_per_category__


def reduce_loaded_data(reduce_by: float = 1.0):
    if reduce_by <= 1.0:
        return

    global __num_per_category__, __num_training_per_category__

    __num_per_category__ = int(__num_per_category__ // reduce_by)
    __num_training_per_category__ = int(__num_training_per_category__ // reduce_by)
    print(__num_per_category__, __num_training_per_category__)
    return


def sketch_per_thread() -> int:
    assert __sketch_per_thread__ > 0
    return __sketch_per_thread__


def abstraction_process_log_dir() -> str:
    return __abs_prcs_log_dir__


def share_classifier() -> bool:
    """
    If True, all SAEnv use one common loaded classifier model. issue: This leads to Tracing in keras model.
    Otherwise, all SAEnv would own copy of a classifier model.
    """
    return __share_classifier__
