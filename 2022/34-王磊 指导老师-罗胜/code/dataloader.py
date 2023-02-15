import threading
import random

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import BatchSampler
from torch.utils.data import _utils
# from torch.utils.data.dataloader import _DataLoaderIter

from torch.utils.data._utils import collate
from torch.utils.data._utils import signal_handling
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils import IS_WINDOWS
from torch.utils.data._utils.worker import ManagerWatchdog

from torch._six import queue

class _DataLoaderIter(object):
    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = multiprocessing.Event()

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = multiprocessing.Process(
                    target=_utils.worker._worker_loop,
                    args=(self.dataset, index_queue,
                          self.worker_result_queue, self.done_event,
                          self.collate_fn, base_seed + i,
                          self.worker_init_fn, i))
                w.daemon = True
                # NB: Process.start()
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            if self.pin_memory:
                self.data_queue = queue.Queue()
                pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop,
                    args=(self.worker_result_queue, self.data_queue,
                          torch.cuda.current_device(), self.done_event))
                pin_memory_thread.daemon = True
                pin_memory_thread.start()
                # Similar to workers (see comment above), we only register pin_memory_thread once it is started.
                self.pin_memory_thread = pin_memory_thread
            else:
                self.data_queue = self.worker_result_queue

            _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def _get_batch(self):
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        elif self.pin_memory:
            while self.pin_memory_thread.is_alive():
                try:
                    return self.data_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                except queue.Empty:
                    continue
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
        else:
            return self.data_queue.get()

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, _utils.ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        raise NotImplementedError("_DataLoaderIter cannot be pickled")

    def _shutdown_workers(self):
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted. See (1) and the second half of the note.
        if not self.shutdown:
            self.shutdown = True
            # Removes pids from the C side data structure first so worker termination afterwards won't trigger false positive error report.
            if self.worker_pids_set:
                _utils.signal_handling._remove_worker_pids(id(self))
                self.worker_pids_set = False

            self.done_event.set()

            # Exit `pin_memory_thread` first because exiting workers may leave
            # corrupted data in `worker_result_queue` which `pin_memory_thread` reads from.
            if hasattr(self, 'pin_memory_thread'):
                self.worker_result_queue.cancel_join_thread()
                self.worker_result_queue.put(None)
                self.pin_memory_thread.join()
                self.worker_result_queue.close()

            # Exit workers now.
            for q in self.index_queues:
                q.put(None)
                # Indicate that no more data will be put on this queue by the current process.
                q.close()
            for w in self.workers:
                w.join()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


def _ms_loop(dataset, index_queue, data_queue, done_event, collate_fn, scale, seed, init_fn, worker_id):
    try:
        collate._use_shared_memory = True
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if r is None:
                assert done_event.is_set()
                return
            elif done_event.is_set():
                continue

            idx, batch_indices = r
            try:
                idx_scale = 0
                if len(scale) > 1 and dataset.train:
                    idx_scale = random.randrange(0, len(scale))
                    dataset.set_scale(idx_scale)

                samples = collate_fn([dataset[i] for i in batch_indices])
                samples.append(idx_scale)
            except Exception:
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples

    except KeyboardInterrupt:
        pass

class _MSDataLoaderIter(_DataLoaderIter):

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = multiprocessing.Event()

            base_seed = torch.LongTensor(1).random_()[0]

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        index_queue,
                        self.worker_result_queue,
                        self.done_event,
                        self.collate_fn,
                        self.scale,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                w.daemon = True
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            if self.pin_memory:
                self.data_queue = queue.Queue()
                pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop,
                    args=(
                        self.worker_result_queue,
                        self.data_queue,
                        torch.cuda.current_device(),
                        self.done_event
                    )
                )
                pin_memory_thread.daemon = True
                pin_memory_thread.start()
                self.pin_memory_thread = pin_memory_thread
            else:
                self.data_queue = self.worker_result_queue

            _utils.signal_handling._set_worker_pids(
                id(self), tuple(w.pid for w in self.workers)
            )
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

            for _ in range(2 * self.num_workers):
                self._put_indices()


class MSDataLoader(DataLoader):

    def __init__(self, cfg, *args, **kwargs):
        super(MSDataLoader, self).__init__(
            *args, **kwargs, num_workers=cfg.n_threads
        )
        self.scale = cfg.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)