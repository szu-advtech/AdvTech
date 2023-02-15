import os
import argparse
import json
import random
import warnings
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformer_basecaller import LabelDataset, get_tri_stage_scheduler
from transformer_basecaller import SACallModel, CTCCriterionConfig, CTCCriterion
import fast_ctc_decode
import parasail
import re
import collections


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    if category.__name__ != 'FutureWarning':
        return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
    else:
        return ''


warnings.formatwarning = warning_on_one_line

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, working_directory, epoch, step=None):
    directory = os.path.join(os.path.abspath(working_directory), 'weights')
    if not os.path.exists(directory):
        os.makedirs(directory)
    if step is None:
        torch.save(state, os.path.join(directory, 'epoch_{}.checkpoint.pth.tar'.format(epoch)))
    else:
        torch.save(state, os.path.join(directory, 'epoch_{}_step_{}.checkpoint.pth.tar'.format(epoch, step)))


def train(epoch, train_loader, model, criterion, optimizer, scheduler, logger, args, gpu):
    model.train()
    for step, (signal, targets, target_lengths) in enumerate(train_loader):
        signal = signal.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        target_lengths = target_lengths.cuda(gpu, non_blocking=True)
        global_step = epoch * len(train_loader) + step
        output = model(signal)

        loss, sample_size, logging_output = criterion(output, targets, target_lengths)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            # Gradients are unscaled during context manager exit
        # Now it's safe to clip.  Replace
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # with
        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=2.0)

        optimizer.step()
        scheduler.step()

        if logger is not None and (global_step + 1) % args.print_freq == 0:
            logger.add_scalar('train/loss', loss.item(), global_step + 1)
            logger.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step + 1)


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr


def align_accuracy(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = collections.defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        acc_ = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        acc_ = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return acc_ * 100


def convert_seq_to_str(s, s_len, alphabet):
    ret = [alphabet[c] for c in s[0: s_len]]
    return "".join(ret)


def accuracy(logit, target, target_len, alphabet):
    batch_size = len(logit)
    acc_one_batch_ = []
    for i in range(batch_size):
        log_prob = logit[i]
        for j in range(len(log_prob)):
            log_prob[j] = np.exp(log_prob[j])
        seq, path = fast_ctc_decode.beam_search(log_prob, alphabet, beam_size=30, )
        target_seq = convert_seq_to_str(target[i], target_len[i], alphabet)
        acc_ = align_accuracy(target_seq, seq, min_coverage=0.5)
        acc_one_batch_.append(acc_)
    return acc_one_batch_


def evaluate(val_loader, model, criterion, logger, epoch, gpu, args):
    model.eval()
    acc = []
    with torch.no_grad():
        for step, (signal, targets, target_lengths) in enumerate(val_loader):
            signal = signal.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            target_lengths = target_lengths.cuda(gpu, non_blocking=True)
            output = model(signal)

            logits = SACallModel.get_normalized_probs(output, log_probs=True).contiguous()  # T x B x C
            logits = logits.transpose(0, 1)  # B x T x C
            acc_one_batch = accuracy(logits.cpu().numpy(),
                                     targets.cpu().numpy(),
                                     target_lengths.cpu().numpy(),
                                     args.alphabet)
            acc.extend(acc_one_batch)
            loss, sample_size, logging_output = criterion(output, targets, target_lengths)

            if logger is not None and (step + 1) % args.print_freq == 0:
                logger.add_scalar('eval_epoch{}/acc_mean'.format(epoch + 1), np.mean(acc_one_batch), step + 1)
                logger.add_scalar('eval_epoch{}/loss'.format(epoch + 1), loss.item(), step + 1)

    if logger is not None:
        logger.add_scalar('eval/acc_mean', np.mean(acc), epoch + 1)
        logger.add_scalar('eval/acc_median', np.median(acc), epoch + 1)
    print('GPU{}/epoch{} mean_acc={:.3f}% median_acc={:.3f}%'.format(gpu, epoch + 1, np.mean(acc), np.median(acc)))


def main_worker(gpu, args):
    torch.cuda.set_device(gpu)

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        set_global_seed(args.seed)
        print('setting seed={} on cuda:{}'.format(args.seed, gpu))

    args.batch_size = args.batch_size // args.ngpus_per_node
    args.rank = args.rank * args.ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    print('GPU rank{} use batch size {}'.format(args.rank, args.batch_size))

    train_dataset = LabelDataset(args.data, read_limit=args.limit, is_validate=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                                    rank=args.rank, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                               num_workers=0, pin_memory=True, sampler=train_sampler)

    valid_dataset = LabelDataset(args.data, read_limit=None, is_validate=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=args.world_size,
                                                                    rank=args.rank, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                               drop_last=True, num_workers=0, pin_memory=True, sampler=valid_sampler)

    total_training_step = args.epochs * len(train_loader)

    if gpu == 0:
        print('train dataset size {}'.format(len(train_dataset)))
        print('valid dataset size {}'.format(len(valid_dataset)))

    model = SACallModel(use_conv_transformer_encoder=args.use_conv_transformer_encoder)
    if args.sync_bn:
        import apex
        print('using apex synced BN')
        model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda(gpu)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    warmup_steps = int(total_training_step * args.lr_warmup_ratio)
    hold_steps = int(total_training_step * args.lr_hold_ratio)
    decay_steps = int(total_training_step * args.lr_decay_ratio)
    scheduler = get_tri_stage_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_hold_steps=hold_steps,
        num_decay_steps=decay_steps,
        lr_end=args.lr_end,
    )

    model = DDP(model, delay_allreduce=True)

    criterion_config = CTCCriterionConfig(
        zero_infinity=True,
        blank_idx=0,
        reduction="mean",
    )
    criterion = CTCCriterion(criterion_config).cuda(gpu)

    logger = None
    if args.rank % args.ngpus_per_node == 0:
        logger_path = os.path.join(os.path.abspath(args.output), 'log')
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        logger = SummaryWriter(logger_path)

    if args.restore is not None:
        if os.path.isfile(args.restore):
            checkpoint = torch.load(args.restore, map_location=lambda storage, loc: storage.cuda(gpu))
            args.start_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            amp.load_state_dict(checkpoint['amp'])
            print('GPU_{} restore model'.format(gpu))
        else:
            raise RuntimeError('-> no checkpoint found at {}'.format(args.restore))

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train(epoch, train_loader, model, criterion, optimizer, scheduler, logger, args, gpu)

        if (epoch + 1) % args.eval_freq == 0:
            evaluate(valid_loader, model, criterion, logger, epoch, gpu, args)

        if args.rank % args.ngpus_per_node == 0:
            save_checkpoint({
                'epoch': epoch + 1,  # 用于作为下一次训练的start_epoch
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp': amp.state_dict(),
            }, args.output, epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basecaller fine-tune for labeled signal dataset')
    parser.add_argument('data', metavar='DATASET', help='bonito labeled dataset directory (containing chunks.npy, reference_lengths.npy and references.npy)')
    parser.add_argument('output', metavar='OUTPUT', help='output directory (save log and model weights)')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs (default: 5)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restart training)')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size for all GPUs on the current node (default: 32)')
    parser.add_argument('--eval-batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size for each GPU when evaluating model')
    parser.add_argument('--restore', metavar='PATH', type=str, default=None, help='restore checkpoint')
    parser.add_argument('--use-conv-transformer-encoder', action='store_true', help='use convolution + self-attention transformer encoder')

    parser.add_argument('--lr-warmup-ratio', default=0.05, type=float, help='learning rate scheduler warmup steps ratio')
    parser.add_argument('--lr-hold-ratio', default=0.35, type=float, help='learning rate scheduler hold steps ratio')
    parser.add_argument('--lr-decay-ratio', default=0.2, type=float, help='learning rate scheduler decay steps ratio')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='final learning rate')

    parser.add_argument('--limit', metavar='N', type=int, default=None, help='read limit for fine-tune training')
    parser.add_argument('--alphabet', metavar='ALPHABET', type=str, default='NACGT', help='Canonical base alphabet (default: NACGT)')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', default=1, type=int, help='evaluation frequency (default: 1)')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:12345', type=str, help='master url (eg, tcp://224.66.41.62:23456) to setup distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend (default: nccl)')
    parser.add_argument('--seed', default=40, type=int, help='seed for deterministic training (default: 40)')
    parser.add_argument('--ngpus-per-node', default=1, type=int, metavar='N', help='number of gpus per node, the value <= torch.cuda.device_count() (default: 1)')
    parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str, default="O1", help="apex optimization level, O1 is recommended for typical use")
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None, help="Keeping the batchnorms in FP32 improves stability and allows Pytorch to use cudnn batchnorms")
    parser.add_argument('--loss-scale', type=str, default=None)
    args_ = parser.parse_args()

    if not os.path.exists(args_.data):
        raise NotADirectoryError('bonito dataset directory is not valid')

    if not os.path.exists(args_.output):
        raise NotADirectoryError('output directory is not valid')

    with open(os.path.join(args_.output, 'config.json'), 'w') as f:
        json.dump(args_.__dict__, f, indent=2)

    torch.backends.cudnn.benchmark = True
    if args_.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args_.seed)  # 环境变量可以在主函数设置

    if args_.dist_url == "env://":
        print("Master URL is tcp://{}:{}".format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
        if args_.world_size == -1:
            args_.world_size = int(os.environ['WORLD_SIZE'])
        if args_.rank == -1:
            args_.rank = int(os.environ["RANK"])
    else:
        print("Master URL is {}".format(args_.dist_url))
        if args_.world_size == -1:
            parser.error('--world-size is not valid')
        if args_.rank == -1:
            parser.error('--rank is not valid')
    print("Node number: {}; Current Node ID: {}".format(args_.world_size, args_.rank))

    if args_.ngpus_per_node > torch.cuda.device_count():
        parser.error('--ngpus-per-node must be <= {}'.format(torch.cuda.device_count()))
    print('Current Node use {} gpus'.format(args_.ngpus_per_node))

    args_.world_size = args_.ngpus_per_node * args_.world_size
    mp.spawn(main_worker, nprocs=args_.ngpus_per_node, args=(args_,))
