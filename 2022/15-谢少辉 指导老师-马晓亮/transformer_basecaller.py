import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import fast_ctc_decode
from tqdm import tqdm
from transformer_basecaller import SACallModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python transformer_basecaller.py')
    parser.add_argument('input_path', type=str, help='fast5 preprocessed directory')
    parser.add_argument('output_path', type=str, help='output directory')
    parser.add_argument('--model', type=str, default=None, help='model checkpoint path')
    parser.add_argument('--use-conv-transformer-encoder', action='store_true', help='use convolution + self-attention transformer encoder')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device id')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=40, help='seed number (default: 40)')
    parser.add_argument('--chunksize', type=int, default=3600, help='signal chunk length (default: 3600)')
    parser.add_argument('--overlap', type=int, default=400, help='signal chunk overlap (default: 400)')
    parser.add_argument('--stride', type=int, default=4, help='model time down-sampling stride (default: 4)')
    parser.add_argument('--alphabet', type=str, default='NACGT', help='nucleic acid alphabet (default: NACGT)')
    parser.add_argument('--beam-size', type=int, default=30, help='beam search size (default: 30)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        raise NotADirectoryError('input directory is not found')

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    else:
        print('output directory {} is exist'.format(args.output_path))
        exit()

    if not os.path.isfile(args.model):
        raise FileNotFoundError('model path {} is not found'.format(args.model))

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.gpu is None:
        print('CPU is used for basecalling')
    else:
        print('GPU {} is used for basecalling'.format(args.gpu))
        torch.cuda.set_device(args.gpu)

    model = SACallModel(use_conv_transformer_encoder=args.use_conv_transformer_encoder)
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    model.eval()

    success_reads_dict = dict()
    success_read_signal_lengths = np.load(os.path.join(args.input_path, 'success_read_signal_lengths.npy'), mmap_mode='r')
    with open(os.path.join(args.input_path, 'success_read_ids.txt'), 'r') as f:
        i = 0
        for line in f:
            success_reads_dict[line.rstrip()] = int(success_read_signal_lengths[i])
            i += 1

    chunksize = args.chunksize
    overlap = args.overlap
    stride = args.stride
    fasta = open(os.path.join(args.output_path, 'basecaller.fasta'), 'w')

    for read_id, T in tqdm(iter(success_reads_dict.items())):
        chunks = np.load(os.path.join(args.input_path, str(read_id) + '.npy'))
        chunks = torch.from_numpy(chunks)

        i = 0
        read_prob = None
        with torch.inference_mode():
            while i < chunks.size(0):
                batch = chunks[i: min(i + args.batch_size, chunks.size(0))]
                if args.gpu is not None:
                    batch = batch.cuda(args.gpu, non_blocking=True)
                output = model(batch)
                logits = SACallModel.get_normalized_probs(output, log_probs=False).contiguous()  # T x B x C
                logits = logits.transpose(0, 1)  # B x T x C
                if i == 0:
                    read_prob = logits
                else:
                    read_prob = torch.cat([read_prob, logits], dim=0)
                i += args.batch_size

        prob = None
        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (T - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end
        prob = read_prob[0, :first_chunk_end]
        for i in range(1, chunks.size(0) - 1):
            prob = torch.cat([prob, read_prob[i, start:end]], dim=0)
        if chunks.size(0) > 1:
            prob = torch.cat([prob, read_prob[chunks.size(0) - 1, start:]], dim=0)

        seq, _ = fast_ctc_decode.beam_search(prob.cpu().numpy(), args.alphabet, beam_size=args.beam_size, )
        fasta.write('>{}\n'.format(str(read_id)))
        fasta.write('{}\n'.format(seq))

    fasta.close()
    print('Done......')
