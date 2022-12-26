import os
import argparse
import torch
import numpy as np
import reader
from tqdm import tqdm
from bonito.multiprocessing import process_cancel, process_itemmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fast5_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--chunksize', type=int, default=3600, )
    parser.add_argument('--overlap', type=int, default=400, )
    parser.add_argument('--nproc', type=int, default=1, )
    args = parser.parse_args()

    if not os.path.isdir(args.fast5_path):
        raise NotADirectoryError('fast5 directory is not found')

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    fast5_reader = reader.Reader(args.fast5_path, recursive=True)
    reads = fast5_reader.get_reads(
        args.fast5_path,
        n_proc=args.nproc,
        recursive=True,
        read_ids=None,
        skip=False,
        do_trim=True,
        cancel=process_cancel(),
    )

    chunksize = args.chunksize
    overlap = args.overlap
    print('chunksize = {}; overlap = {}'.format(chunksize, overlap))

    fail_read_ids = []
    success_read_ids = []
    success_read_signal_lengths = []

    for read in tqdm(reads):
        signal = torch.from_numpy(read.signal)
        T = len(signal)
        if T < chunksize:
            fail_read_ids.append(read.read_id)
            continue
        else:
            success_read_ids.append(read.read_id)
            success_read_signal_lengths.append(T)
            stub = (T - overlap) % (chunksize - overlap)
            chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
            if stub > 0:
                chunks = torch.cat([signal[None, :chunksize], chunks], dim=0)
            np.save(os.path.join(args.output_path, str(read.read_id) + '.npy'), chunks.numpy())

    with open(os.path.join(args.output_path, 'success_read_ids.txt'), 'w') as f:
        for rid in success_read_ids:
            f.write('{}\n'.format(rid))

    with open(os.path.join(args.output_path, 'fail_read_ids.txt'), 'w') as f:
        for rid in fail_read_ids:
            f.write('{}\n'.format(rid))

    success_read_signal_lengths = np.array(success_read_signal_lengths, dtype=int)
    np.save(os.path.join(args.output_path, 'success_read_signal_lengths.npy'), success_read_signal_lengths)
