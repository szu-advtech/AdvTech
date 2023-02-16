import copy, sys
import heapq
import numpy as np
from tqdm import tqdm
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))


from update import LocalUpdate, save_protos, test_inference_new_het_lt, test_inference_avg
from utils import proto_aggregation, agg_func, average_weights

def FedProto(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    global_protos = []
    idxs_users = range(args.num_users)
    train_loss, train_accuracy = [], []
    index_min = []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos, local_protos_a, local_protos_b = [], [], {}, {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, index_min, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)

            agg_protos = agg_func(protos)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
            proto_loss += loss['2']

        arg_min = heapq.nsmallest(15, local_losses)
        index_min = map(local_losses.index, arg_min)
        index_min = list(index_min)
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    if args.dataset == 'mnist' and args.mode == 'task_heter':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedAvg(args, train_dataset, test_dataset, user_groups, global_model, user_groups_lt):
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')
        global_model.train()
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc = local_model.update_weights(idx,
                model=copy.deepcopy(global_model), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        global_model.eval()
    test_acc, test_loss = test_inference_avg(args, global_model, test_dataset, user_groups_lt)
    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
            np.mean(test_acc), np.std(test_acc)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(
            np.mean(test_loss), np.std(test_loss)))
    if args.dataset == 'mnist':
        local_model_list = [global_model] * args.num_users
        save_protos(args, local_model_list, test_dataset, user_groups_lt)