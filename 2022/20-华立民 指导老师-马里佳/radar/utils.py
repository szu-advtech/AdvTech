import numpy
import random

def get_argument(argv, index, default_value):  #返回argv中的domain name
    if len(argv) > index:
        return argv[index]
    else:
        return default_value

def get_float_argument(argv, index, default_value):  #返回敌对率adversary_ratio
    if len(argv) > index:
        return float(argv[index])
    else:
        return default_value

def get_int_argument(argv, index, default_value):
    if len(argv) > index:
        return int(argv[index])
    else:
        return default_value

def argmax(values):
    max_value = max(values)
    default_index = numpy.argmax(values)
    candidate_indices = []
    for i,value in enumerate(values):
        if value >= max_value:
            candidate_indices.append(i)
    if not candidate_indices:
        return default_index
    return random.choice(candidate_indices)

def argmin(values):
    min_value = min(values)
    default_index = numpy.argmin(values)
    candidate_indices = []
    for i,value in enumerate(values):
        if value <= min_value:
            candidate_indices.append(i)
    if not candidate_indices:
        return default_index
    return random.choice(candidate_indices)

def get_param_or_default(params, label, default):
    if label in params:
        return params[label]
    return default

def get_value_if(value, condition, default_value):
    if condition:
        return value
    return default_value

def check_value_not_none(value, label):
    if value is None:
        raise ValueError("No '{}' provided!".format(label))
    return value

def pad_or_truncate_sequences(sequences, max_length, default_elements):
    sequence_length = len(sequences)
    if sequence_length > max_length:
        sequences = sequences[-max_length:]
    if sequence_length < max_length:
        items_missing = max_length - sequence_length
        sequences += [default_elements for _ in range(items_missing)]
    assert len(sequences) == max_length
    return sequences