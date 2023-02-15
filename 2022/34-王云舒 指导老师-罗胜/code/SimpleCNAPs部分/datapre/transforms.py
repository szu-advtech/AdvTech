from contextlib import nullcontext
import copy
import random
import collections
import functools
import array
import itertools

from .taskdataset import DataDescription


class TaskTransform:

    def __init__(self, dataset):
        self.dataset = dataset

    def new_task(self):
        n = len(self.dataset)
        task_description = [None] * n
        for i in range(n):
            task_description[i] = DataDescription(i)
        return task_description

# 加载数据
class LoadData(TaskTransform):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Loads a sample from the dataset given its index.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset):
        super(LoadData, self).__init__(dataset)
        self.dataset = dataset

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        for data_description in task_description:
            data_description.transforms.append(lambda x: self.dataset[x])
        return task_description


class CythonFilterLabels(TaskTransform):

    def __init__(self, dataset, labels):
        super(CythonFilterLabels, self).__init__(dataset)
        indices_to_labels = dict(dataset.indices_to_labels)
        len_dataset = len(dataset)
        self.labels = labels
        self.filtered_indices = array.array('i', [0] * len_dataset)
        for i in range(len_dataset):
            self.filtered_indices[i] = int(indices_to_labels[i] in self.labels)

    def __reduce__(self):
        return CythonFilterLabels, (self.dataset, self.labels)

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        result = []
        for dd in task_description:
            if self.filtered_indices[dd.index]:
                result.append(dd)
        return result


class FilterLabels(CythonFilterLabels):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**
    从给定索引的数据集加载样本。
    Removes samples that do not belong to the given set of labels.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample. 要从中加载示例的数据集。
    * **labels** (list) - The list of labels to include. 要包括的标签列表。

    """

    def __init__(self, dataset, labels):
        super(FilterLabels, self).__init__(dataset, labels)


class ConsecutiveLabels(TaskTransform):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Re-orders the samples in the task description such that they are sorted in
    consecutive order.

    Note: when used before `RemapLabels`, the labels will be homogeneously clustered,
    but in no specific order.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset):
        super(ConsecutiveLabels, self).__init__(dataset)
        self.dataset = dataset

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        pairs = [(dd, self.dataset.indices_to_labels[dd.index])
                 for dd in task_description]
        pairs = sorted(pairs, key=lambda x: x[1])
        return [p[0] for p in pairs]


class RemapLabels(TaskTransform):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Given samples from K classes, maps the labels to 0, ..., K.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset, shuffle=True):
        super(RemapLabels, self).__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle

    def remap(self, data, mapping):
        data = [d for d in data]
        data[1] = mapping(data[1])
        return data

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        labels = list(set(self.dataset.indices_to_labels[dd.index] for dd in task_description))
        if self.shuffle:
            random.shuffle(labels)

        def mapping(x):
            return labels.index(x)
            # return x

        for dd in task_description:
            remap = functools.partial(self.remap, mapping=mapping)
            dd.transforms.append(remap)
        return task_description


class CythonNTask(TaskTransform):

    def __init__(self, dataset):
        super(CythonNTask, self).__init__(dataset)
        self.indices_to_labels = dict(dataset.indices_to_users)

    def __reduce__(self):
        return CythonNTask, (self.dataset)

    def new_task(self):  # Efficient initializer 有效的初始化
        tasks = self.dataset.tasks
        task_description = []
        tasks_to_indices = dict(self.dataset.tasks_to_indices)
        task = random.sample(tasks, k=1)
        task = task[0]
        for idx in tasks_to_indices[task]:
            task_description.append(DataDescription(idx))
        return task_description


class NTask(CythonNTask):

    def __init__(self, dataset):
        super(NTask, self).__init__(dataset=dataset)

    def __call__(self, task_description):
        if task_description is None:
            return self.new_task()
        else:
            print("wtf")
            return nullcontext


# 要修改nways里的，让继承前面的description
class CythonNWays(TaskTransform):

    def __init__(self, dataset, n=2):
        super(CythonNWays, self).__init__(dataset)
        self.n = n
        self.indices_to_labels = dict(dataset.indices_to_labels)

    def __reduce__(self):
        return CythonNWays, (self.dataset, self.n)

    def new_task(self):  # Efficient initializer 有效的初始化
        labels = self.dataset.labels
        task_description = []
        labels_to_indices = dict(self.dataset.labels_to_indices)
        # print(labels)
        # print(self.n)
        classes = random.sample(labels, k=self.n)
        for cl in classes:
            for idx in labels_to_indices[cl]:
                task_description.append(DataDescription(idx))
        return task_description


class NWays(CythonNWays):

    def __init__(self, dataset, n=2):
        super(NWays, self).__init__(dataset=dataset, n=n)

    def __call__(self, task_description):
        if task_description is None:
            return self.new_task()
        classes = []
        result = []
        set_classes = set()
        for dd in task_description:
            set_classes.add(self.indices_to_labels[dd.index])
        classes = set_classes
        classes = random.sample(classes, k=self.n)
        for dd in task_description:
            if self.indices_to_labels[dd.index] in classes:
                result.append(dd)
        return result


class CythonKShots(TaskTransform):

    def __init__(self, dataset, k=1, replacement=False):
        super(CythonKShots, self).__init__(dataset)
        self.dataset = dataset
        self.k = k
        self.replacement = replacement

    def __reduce__(self):
        return CythonKShots, (self.dataset, self.k, self.replacement)

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        # TODO: The order of the data samples is not preserved.
        class_to_data = collections.defaultdict(list)
        for dd in task_description:
            cls = self.dataset.indices_to_labels[dd.index]
            class_to_data[cls].append(dd)
        if self.replacement:
            def sampler(x, k):
                return [copy.deepcopy(dd)
                        for dd in random.choices(x, k=k)]
        else:
            sampler = random.sample

        return list(itertools.chain(*[sampler(dds, k=self.k) for dds in class_to_data.values()]))


class KShots(CythonKShots):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps K samples for each present labels.40/5000
    为每个现在的标签保留K个样品。

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample. 要从中加载示例的数据集。
    * **k** (int, *optional*, default=1) - The number of samples per label. 每个标签的样品数量。
    * **replacement** (bool, *optional*, default=False) - Whether to sample with replacement.

    """

    def __init__(self, dataset, k=1, replacement=False):
        super(KShots, self).__init__(dataset=dataset, k=k, replacement=replacement)


class CythonFusedNWaysKShots(TaskTransform):

    def __init__(self, dataset, n=2, k=1, replacement=False, filter_labels=None):
        super(CythonFusedNWaysKShots, self).__init__(dataset)
        self.n = n
        self.k = k
        self.replacement = replacement
        if filter_labels is None:
            filter_labels = self.dataset.labels
        self.filter_labels = filter_labels
        self.filter = FilterLabels(self.dataset, self.filter_labels)
        self.nways = NWays(self.dataset, self.n)
        self.kshots = KShots(self.dataset, k=self.k, replacement=self.replacement)

    def __reduce__(self):
        return CythonFusedNWaysKShots, (self.dataset,
                                        self.n,
                                        self.k,
                                        self.replacement,
                                        self.filter_labels)

    def new_task(self):
        task_description = []
        labels = self.filter_labels
        selected_labels = random.sample(labels, k=self.n)
        for sl in selected_labels:
            indices = self.dataset.labels_to_indices[sl]
            if self.replacement:
                selected_indices = [random.choice(indices) for _ in range(self.k)]
            else:
                selected_indices = random.sample(indices, k=self.k)
            for idx in selected_indices:
                task_description.append(DataDescription(idx))
        return task_description

    def __call__(self, task_description):
        if task_description is None:
            return self.new_task()
        # Not fused
        return self.kshots(self.nways(self.filter(task_description)))


class FusedNWaysKShots(CythonFusedNWaysKShots):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Efficient implementation of FilterLabels, NWays, and KShots.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.
    * **k** (int, *optional*, default=1) - The number of samples per label.
    * **replacement** (bool, *optional*, default=False) - Whether to sample shots with replacement.
    * **filter_labels** (list, *optional*, default=None) - The list of labels to include. Defaults to
        all labels in the dataset.
    """

    def __init__(self, dataset, n=2, k=1, replacement=False, filter_labels=None):
        super(FusedNWaysKShots, self).__init__(
            dataset,
            n=n,
            k=k,
            replacement=replacement,
            filter_labels=filter_labels,
        )
