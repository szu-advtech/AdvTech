# ====================================================
# Classifier config.
# ====================================================

__batch_size__ = 128
__num_per_category__: int = 75000
__num_training_per_category__: int = 70000

__lstm_hidden_size__ = 256
__epoch__ = 6
__shuffle_seed__ = 94228
__csv_filepath__ = "training/classifier_model/log.csv"
__dataset_dir__ = "./dataset/"
__filename_prefix__ = "full_binary_"
__filename_suffix__ = ".bin"
__classifier_model_path__ = 'training/classifier_model/'


def batch_size() -> int:
    return __batch_size__


def num_per_category() -> int:
    return __num_per_category__


def num_training_per_category() -> int:
    return __num_training_per_category__


def epoch() -> int:
    return __epoch__


def shuffle_seed() -> int:
    return __shuffle_seed__


# def checkpoint_path() -> str:
#     return __checkpoint_path__


def csv_filepath() -> str:
    return __csv_filepath__


def classifier_model_path() -> str:
    return __classifier_model_path__


def dataset_filename(category_name: str) -> str:
    result = __dataset_dir__ + __filename_prefix__ + category_name + __filename_suffix__
    return result


def reduce_loaded_data(reduce_by: float = 1.0):
    if reduce_by <= 1.0:
        return

    global __num_per_category__, __num_training_per_category__

    __num_per_category__ = int(__num_per_category__ // reduce_by)
    __num_training_per_category__ = int(__num_training_per_category__ // reduce_by)
    print(__num_per_category__, __num_training_per_category__)
    return


def lstm_hidden_size() -> int:
    return __lstm_hidden_size__
