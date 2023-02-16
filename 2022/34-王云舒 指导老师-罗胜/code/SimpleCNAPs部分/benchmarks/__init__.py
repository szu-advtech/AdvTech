#!/usr/bin/env python3

"""
The benchmark modules provides a convenient interface to standardized benchmarks in the literature.
It provides train/validation/test TaskDatasets and TaskTransforms for pre-defined datasets.

This utility is useful for researchers to compare new algorithms against existing benchmarks.
For a more fine-grained control over tasks and data, we recommend directly using `l2l.data.TaskDataset` and `l2l.data.TaskTransforms`.
"""

import os

import datapre

from collections import namedtuple
from .digits_benchmark import digits_tasksets
from .omniglot_benchmark import omniglot_tasksets
from .mini_imagenet_benchmark import mini_imagenet_tasksets
# from .tiered_imagenet_benchmark import tiered_imagenet_tasksets
# from .fc100_benchmark import fc100_tasksets
# from .cifarfs_benchmark import cifarfs_tasksets
# from .digitsc_benchmark import digitsc_tasksets
# from .normaltest_digits_benchmark import normal_digits_tasksets
# from .activity_benchmark import activity_tasksets


__all__ = ['list_tasksets', 'get_tasksets']


BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test'))
BenchmarkTasksets_new = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test_train1', 'test1', 'test_train2', 'test2'))
BenchmarkTasksets_two = namedtuple('BenchmarkTasksets', ('train', 'test'))

_TASKSETS = {
    'digits': digits_tasksets,
    'omniglot': omniglot_tasksets,
    'mini-imagenet': mini_imagenet_tasksets,
    # 'tiered-imagenet': tiered_imagenet_tasksets,
    # 'fc100': fc100_tasksets,
    # 'cifarfs': cifarfs_tasksets,
    # 'digitsc': digitsc_tasksets,
    # 'digits_normal': normal_digits_tasksets,
    # 'act': activity_tasksets
}


def list_tasksets():
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns a list of all available benchmarks.

    **Example**
    ~~~python
    for name in l2l.vision.benchmarks.list_tasksets():
        print(name)
        tasksets = l2l.vision.benchmarks.get_tasksets(name)
    ~~~
    """
    return _TASKSETS.keys()

def get_tasksets_two(
    name,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=-1,
    root='~/data',
    device=None,
    **kwargs,
):
    root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    # Load task-specific data and transforms
    datasets, transforms = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           **kwargs)
    train_dataset, test_dataset = datasets
    train_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = datapre.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = datapre.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets_two(train_tasks, test_tasks)

def get_tasksets_act(
    name,
    setname,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=-1,
    root='~/data',
    device=None,
    **kwargs,
):
    root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    # Load task-specific data and transforms
    datasets, transforms = _TASKSETS[name](dataset=setname,
                                           train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = datapre.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = datapre.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = datapre.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)


def get_tasksets(
    name,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=-1,
    root='~/data',
    device=None,
    **kwargs,
):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns the tasksets for a particular benchmark, using literature standard data and task transformations.

    The returned object is a namedtuple with attributes `train`, `validation`, `test` which
    correspond to their respective TaskDatasets.
    See `examples/vision/maml_miniimagenet.py` for an example.

    **Arguments**

    * **name** (str) - The name of the benchmark. Full list in `list_tasksets()`.
    * **train_ways** (int, *optional*, default=5) - The number of classes per train tasks.
    * **train_samples** (int, *optional*, default=10) - The number of samples per train tasks.
    * **test_ways** (int, *optional*, default=5) - The number of classes per test tasks. Also used for validation tasks.
    * **test_samples** (int, *optional*, default=10) - The number of samples per test tasks. Also used for validation tasks.
    * **num_tasks** (int, *optional*, default=-1) - The number of tasks in each TaskDataset.
    * **root** (str, *optional*, default='~/data') - Where the data is stored.

    **Example**
    ~~~python
    train_tasks, validation_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = train_tasks.sample()

    or:

    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = tasksets.train.sample()
    ~~~
    """
    root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    # Load task-specific data and transforms
    datasets, transforms = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = datapre.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = datapre.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = datapre.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)

def get_tasksets_new(
    name,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=-1,
    root='~/data',
    device=None,
    **kwargs,
):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns the tasksets for a particular benchmark, using literature standard data and task transformations.

    The returned object is a namedtuple with attributes `train`, `validation`, `test` which
    correspond to their respective TaskDatasets.
    See `examples/vision/maml_miniimagenet.py` for an example.

    **Arguments**

    * **name** (str) - The name of the benchmark. Full list in `list_tasksets()`.
    * **train_ways** (int, *optional*, default=5) - The number of classes per train tasks.
    * **train_samples** (int, *optional*, default=10) - The number of samples per train tasks.
    * **test_ways** (int, *optional*, default=5) - The number of classes per test tasks. Also used for validation tasks.
    * **test_samples** (int, *optional*, default=10) - The number of samples per test tasks. Also used for validation tasks.
    * **num_tasks** (int, *optional*, default=-1) - The number of tasks in each TaskDataset.
    * **root** (str, *optional*, default='~/data') - Where the data is stored.

    **Example**
    ~~~python
    train_tasks, validation_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = train_tasks.sample()

    or:

    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = tasksets.train.sample()
    ~~~
    """
    root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    # Load task-specific data and transforms
    datasets, transforms = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           **kwargs)
    train_dataset, validation_dataset, test_train1, test_train2, test_dataset1, test_dataset2 = datasets
    train_transforms, validation_transforms, testtrain_transforms1, test_transforms1, testtrain_transforms2, test_transforms2 = transforms

    # Instantiate the tasksets
    train_tasks = datapre.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = datapre.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    testtrain1 = datapre.TaskDataset(
        dataset=test_train1,
        task_transforms=testtrain_transforms1,
        num_tasks=num_tasks,
    )
    test_tasks1 = datapre.TaskDataset(
        dataset=test_dataset1,
        task_transforms=test_transforms1,
        num_tasks=num_tasks,
    )
    testtrain2 = datapre.TaskDataset(
        dataset=test_train2,
        task_transforms=testtrain_transforms2,
        num_tasks=num_tasks,
    )
    test_tasks2 = datapre.TaskDataset(
        dataset=test_dataset2,
        task_transforms=test_transforms2,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets_new(train_tasks, validation_tasks, testtrain1, test_tasks1, testtrain2, test_tasks2)