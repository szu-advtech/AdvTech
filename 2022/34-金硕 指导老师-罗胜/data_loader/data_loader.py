from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import math
from .HHARDataset import HHARDataset
from .DSADataset import DSADataset
from .MetaSense_ActivityDataset import MetaSense_ActivityDataset
from .MetaSense_SpeechDataset import MetaSense_SpeechDataset

import copy





def split_data(entire_data, valid_split, test_split, train_max_rows, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(entire_data) * valid_split)
    test_size = math.floor(len(entire_data) * test_split)

    train_size = len(entire_data) - valid_size - test_size
    if valid_size > 0 and test_size > 0 : # this meets when it's mostly target condition
        if train_size < 20: # for 'src = all' && meta learning cases, source should have at least 5 supports and queries, while target has at least 10 shots for evaluation.
            gap = 20 - train_size
            train_size = 20
            valid_size -= gap # decrease valid size
    assert(train_size >=0 and valid_size >=0 and test_size >=0)

    train_data, valid_data, test_data = torch.utils.data.random_split(entire_data,
                                                                      [train_size, valid_size, test_size])

    if len(entire_data) > train_max_rows:
        train_data = torch.utils.data.Subset(train_data, range(train_max_rows))
    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return train_data, valid_data, test_data

def datasets_to_dataloader(datasets, batch_size, concat=True, num_workers=8, shuffle=True, drop_last=False):
    if concat:
        data_loader = None
        if len(datasets):
            if type(datasets) is torch.utils.data.dataset.Subset:
                datasets = [datasets]
            if sum([len(dataset) for dataset in datasets]) > 0:  # at least one dataset has data
                data_loader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers, drop_last=drop_last, pin_memory=False)
        return data_loader
    else:
        data_loaders = []
        for dataset in datasets:
            if len(dataset) == 0:
                continue
            else:
                data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, drop_last=drop_last, pin_memory=False))
        return data_loaders


def domain_data_loader(args, domains, file_path, batch_size, train_max_rows=np.inf, valid_max_rows=np.inf,
                       test_max_rows=np.inf, valid_split=0.1, test_split=0.1, separate_domains=False, is_src=True):
    entire_datasets = []
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    st = time.time()
    if isinstance(domains, (list,)):
        processed_domains = domains
    else:
        processed_domains = [domains]
    ##-- preprocessing 'rest' domain
    if domains == ['all']:
        domains = ['rest'] # same processing for 'all' case
    if domains == ['rest']:  # convert 'rest' into list of domains
        test_domains = []
        if args.dataset in ['hhar_scaled']:
            # hhar_model_device_user ex) hhar.gear.gear1.a, hhar.nexus..c
            test_domain = args.tgt.split('.')

            users = set([test_domain[2]])  # bracket is required to append a tuple
            # devices = set(HHAROpt['devices']) - set(complementary_options[2])
            models = set([test_domain[0]])

            for user in users:
                for model in models:
                    test_domains.append(
                        model + '..' + user)  # does not support device instance because we don't need at the moment

        elif args.dataset in ['dsa_scaled']:
            # dsa_user_position ex) dsa.p2.T, dsa.p8.LL
            test_domain = args.tgt.split('.')

            users = set([test_domain[0]])  # bracket is required to append a tuple
            positions = set([test_domain[1]])

            for user in users:
                for position in positions:
                    test_domains.append(
                        user + '.' + position)

        elif args.dataset in ['metasense_activity_scaled']:
            # metasense_activity.domain ex) metasense_activity.PH0007-jskim, metasense_activity.PH0014-wjlee
            test_domain = args.tgt.split('.')

            domains_ = set([test_domain[0]])  # bracket is required to append a tuple

            for domain_ in domains_:
                test_domains.append(domain_)

        elif args.dataset in ['metasense_speech_scaled']:
            # metasense_speech.domain ex) metasense_speech.PH0007-jskim, metasense_speech.PH0014-wjlee
            test_domain = args.tgt.split('.')

            domains_ = set([test_domain[0]])  # bracket is required to append a tuple

            for domain_ in domains_:
                test_domains.append(domain_)

        print('Test domains:' + str(test_domains))
        processed_domains = test_domains

    ##-- load dataset per each domain
    print('Domains:{}'.format(processed_domains))
    for domain in processed_domains:

        if args.dataset in ['hhar_scaled']:
            # hhar_model_device_user ex) hhar.gear.gear1.a, hhar.nexus..c
            options = domain.split('.')
            if separate_domains:
                for train_data in HHARDataset(file=file_path, model=options[0], device=options[1], user=options[2],
                                              complementary=True if domains == [
                                                  'rest'] else False).get_datasets_per_domain():
                    entire_datasets.append(train_data)

            else:
                train_data = HHARDataset(file=file_path, model=options[0], device=options[1], user=options[2],
                                         complementary=True if domains == ['rest'] else False)
                if len(train_data) == 0:
                    print('Zero train data: {:s}'.format(domain))
                    continue
                else:
                    entire_datasets.append(train_data)
        elif args.dataset in ['dsa_scaled']:
            # dsa_user_position ex) dsa.p2.T, dsa.p8.LL
            options = domain.split('.')
            if separate_domains:
                for train_data in DSADataset(file=file_path, user=options[0], position=options[1],
                                             complementary=True if domains == [
                                                 'rest'] else False).get_datasets_per_domain():
                    entire_datasets.append(train_data)

            else:
                train_data = DSADataset(file=file_path, user=options[0], position=options[1],
                                        complementary=True if domains == ['rest'] else False)
                if len(train_data) == 0:
                    print('Zero train data: {:s}'.format(domain))
                    continue
                else:
                    entire_datasets.append(train_data)
        elif args.dataset in ['metasense_activity_scaled']:
            # metasense_activity_domain ex) metasense_acitivity.PH0007-jskim
            options = domain.split('.')
            is_ecdf = True if 'ecdf' in args.model else False

            if separate_domains:
                for train_data in MetaSense_ActivityDataset(file=file_path, domain=options[0],
                                                           complementary=True if domains == ['rest'] else False,
                                                           max_source=args.num_source, ecdf=is_ecdf, num_bin = args.num_bin).get_datasets_per_domain():
                    entire_datasets.append(train_data)

            else:
                train_data = MetaSense_ActivityDataset(file=file_path, domain=options[0],
                                                      complementary=True if domains == ['rest'] else False,
                                                      max_source=args.num_source, ecdf=is_ecdf, num_bin = args.num_bin)
                if len(train_data) == 0:
                    print('Zero train data: {:s}'.format(domain))
                    continue
                else:
                    entire_datasets.append(train_data)

        elif args.dataset in ['metasense_speech_scaled']:
            # metasense_speech_domain ex) metasense_speech.PH0007-jskim
            options = domain.split('.')
            if separate_domains:
                for train_data in MetaSense_SpeechDataset(file=file_path, domain=options[0],
                                                         complementary=True if domains == [
                                                             'rest'] else False,
                                                         max_source=args.num_source).get_datasets_per_domain():
                    entire_datasets.append(train_data)

            else:
                train_data = MetaSense_SpeechDataset(file=file_path, domain=options[0],
                                                    complementary=True if domains == ['rest'] else False,
                                                    max_source=args.num_source)
                if len(train_data) == 0:
                    print('Zero train data: {:s}'.format(domain))
                    continue
                else:
                    entire_datasets.append(train_data)

    ##-- split each dataset into train, valid, and test
    for train_data in entire_datasets:
        train_data, valid_data, test_data = split_data(train_data, valid_split, test_split, train_max_rows,
                                                       valid_max_rows, test_max_rows)
        train_datasets.append(train_data)
        valid_datasets.append(valid_data)
        test_datasets.append(test_data)

        print('#Multi?:{:d} data_loader len:{:d} Train: {:d}\t# Valid: {:d}\t# Test: {:d}'.format(
            True if domains == ['rest'] else False, len(train_data), len(train_data), len(valid_data),
            len(test_data)))

    train_datasets = train_datasets[:args.num_source]
    valid_datasets = valid_datasets[:args.num_source]
    test_datasets = test_datasets[:args.num_source]

    print('# Time: {:f} secs'.format(time.time() - st))

    if separate_domains:
        # actual batch size is multiplied by num_class
        train_data_loaders = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=False, drop_last=True)
        # if 'PN' in args.method:
        #     test_batch_size = 5
        # else:
        valid_data_loaders = datasets_to_dataloader(valid_datasets, batch_size=32,
                                                    concat=False)  # set validation batch_size = 32 to boost validation speed
        if is_src:
            test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False, drop_last=True)
        else:
            test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False, drop_last=False)
        # assert (len(train_data_loaders) == len(valid_data_loaders) == len(test_data_loaders))
        data_loaders = []
        for i in range(len(train_data_loaders)):
            data_loader = {
                'train': train_data_loaders[i],
                'valid': valid_data_loaders[i] if len(valid_data_loaders) == len(train_data_loaders) else None,
                'test': test_data_loaders[i],
                'num_domains': len(train_data_loaders)
            }
            data_loaders.append(data_loader)
        print('num_domains:' + str(len(train_data_loaders)))

        return data_loaders
    else:
        train_data_loader = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=True, drop_last=True) #Drop_last for avoding one-sized minibatches for batchnorm layers
        valid_data_loader = datasets_to_dataloader(valid_datasets, batch_size=batch_size, concat=True)
        test_data_loader = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=True)

        data_loader = {
            'train': train_data_loader,
            'valid': valid_data_loader,
            'test': test_data_loader,
            'num_domains': sum([dataset.dataset.get_num_domains() for dataset in train_datasets]),
        }
        print('num_domains:' + str(data_loader['num_domains']))
        return data_loader


def support_query_data_loader(target_data_loader, query_split,  batch_size, nshot, src_separate_domains = True):
    '''
    :param target_data_loader: original target_data_loader
    :param query_split: src_query_split
    :param batch_size: trg_batch_size
    :param nshot: src_batch_size
    :param src_separate_domains:
    :return: (support+query)_data_loader, final_target_data_loader
    '''
    # Distribute original train data of target_data_loader to source(support+query) and target(train)
    source_dataset = []
    empty_dataset = []
    train_target_dataset = []

    for i in range(len(target_data_loader)):
        temp_source_size = math.floor(len(target_data_loader[i]['train'].dataset.indices) * 0.5) # split by 50:50

        temp_source_data = copy.copy(target_data_loader[i]['train'].dataset)
        temp_source_data_indices = target_data_loader[i]['train'].dataset.indices[:temp_source_size]
        temp_source_data.indices = temp_source_data_indices
        temp_target_train_data = copy.copy(target_data_loader[i]['train'].dataset)
        temp_target_data_indices = target_data_loader[i]['train'].dataset.indices[temp_source_size:]
        temp_target_train_data.indices = temp_target_data_indices

        source_dataset.append(temp_source_data)
        train_target_dataset.append(temp_target_train_data)

    # Distribute source_dataset to support_set, valid_set(empty), query_set
    support_sets=[]
    valid_sets=[]
    query_sets=[]
    for i in range(len(source_dataset)):
        temp_query_size = math.floor(len(source_dataset[i].indices)*query_split)
        if query_split > 0: # for meta-learning methods
            temp_query_data = copy.copy(source_dataset[i])
            temp_query_data_indices = source_dataset[i].indices[:temp_query_size]
            temp_query_data.indices = temp_query_data_indices
            query_sets.append(temp_query_data)

        temp_support_data = copy.copy(source_dataset[i])
        temp_support_data_indices = source_dataset[i].indices[temp_query_size:]
        temp_support_data.indices = temp_support_data_indices
        support_sets.append(temp_support_data)


    # each data_loader for temp_source_data_loaders
    if src_separate_domains:
        support_data_loaders = datasets_to_dataloader(support_sets, batch_size=nshot, concat=False)
        valid_data_loaders = datasets_to_dataloader(valid_sets, batch_size=32, concat=False)
        query_data_loaders = datasets_to_dataloader(query_sets, batch_size=nshot, concat=False)

        # temp_source_data_loader = ['train'//support]:0.5 ['valid']:0 ['test'//query]:0.5
        temp_source_data_loaders = []
        for i in range(len(support_data_loaders)):
            temp_source_data_loader = {
                'train': support_data_loaders[i],
                'valid': valid_data_loaders[i] if len(valid_data_loaders) == len(support_data_loaders) else None,
                'test': query_data_loaders[i] if len(query_data_loaders) != 0 else None,
                'num_domains': len(support_data_loaders)
            }
            temp_source_data_loaders.append(temp_source_data_loader)
        print('num_domains:' + str(len(support_data_loaders)))
    else:
        support_data_loaders = datasets_to_dataloader(support_sets, batch_size=nshot, concat=True, drop_last=True)
        valid_data_loaders = datasets_to_dataloader(valid_sets, batch_size=nshot, concat=True)
        query_data_loaders = datasets_to_dataloader(query_sets, batch_size=nshot, concat=True)

        temp_source_data_loaders = {
            'train': support_data_loaders,
            'valid': valid_data_loaders,
            'test': query_data_loaders,
            'num_domains': sum([dataset.dataset.get_num_domains() for dataset in support_sets]),
        }
        print('num_domains:' + str(temp_source_data_loaders['num_domains']))

    # Update target_data_loader[0]['train']
    target_valid_set=[]
    target_test_set=[]
    for i in range(len(target_data_loader)):
        temp_valid_set = target_data_loader[i]['valid'].dataset
        temp_test_set = target_data_loader[i]['test'].dataset

        target_valid_set.append(temp_valid_set)
        target_test_set.append(temp_test_set)

    target_train_data_loaders = datasets_to_dataloader(train_target_dataset, batch_size=batch_size, concat=False)
    target_valid_data_loaders = datasets_to_dataloader(target_valid_set, batch_size=32, concat=False)
    target_test_data_loaders = datasets_to_dataloader(target_test_set, batch_size=batch_size, concat=False)

    temp_target_data_loaders = []
    for i in range(len(target_train_data_loaders)):
        temp_target_data_loader = {
            'train': target_train_data_loaders[i],
            'valid': target_valid_data_loaders[i] if len(target_valid_data_loaders) == len(target_train_data_loaders) else None,
            'test': target_test_data_loaders[i],
            'num_domains': len(target_train_data_loaders)
        }
        temp_target_data_loaders.append(temp_target_data_loader)
    print('num_domains:' + str(len(target_train_data_loaders)))

    return temp_source_data_loaders, temp_target_data_loaders



if __name__ == '__main__':
    pass