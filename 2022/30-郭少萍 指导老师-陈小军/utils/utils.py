import numpy as np
import torch
import torch.utils.data.DataLoader


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)




@torch.no_grad()
def get_features_eval(val_loader, model):
    model.eval()
    targets, features, indices = [], [], []
    for i, batch in enumerate(val_loader):
        if val_loader.dataset.filename == 'text':
            input_, target_, index_ = batch
            input_ = input_.cuda()
            target_ = target_.cuda()
            feature_ = model(input_, forward_pass='backbone_t')
        else:
            input_ = batch['image'].cuda()
            target_ = batch['target'].cuda()
            index_ = batch['index']
            feature_ = model(input_, forward_pass='backbone_i')

        targets.append(target_)
        features.append(feature_.cpu())
        indices.append(index_)

    targets = torch.cat(targets).int()
    features = torch.cat(features)
    indices = torch.cat(indices)

    # Sort features and targets according to indices
    features_order, targets_order = torch.zeros_like(features), torch.zeros_like(targets)
    features_order[indices] = features
    targets_order[indices] = targets

    return features_order, targets_order

@torch.no_grad()
def get_knn_indices(model, train_dataloader, val_dataloader, topk):

    train_features, train_targets = get_features_eval(train_dataloader, model)
    val_features, val_targets = get_features_eval(val_dataloader, model)
    train_features = train_features.float()
    val_features = val_features.float()

    train_indices, train_accuracy = mine_nearest_neighbors(train_features.numpy(), train_targets.cpu().numpy(), topk)
    val_indices, val_accuracy = mine_nearest_neighbors(val_features.numpy(), val_targets.cpu().numpy(), 5)

    return train_indices, train_accuracy, val_indices, val_accuracy


def mine_nearest_neighbors(features, targets, topk):
    # mine the topk nearest neighbors for every sample
    import faiss
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatL2(dim)  # index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included

    # evaluate
    neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    accuracy = np.mean(neighbor_targets == anchor_targets)
    return indices, accuracy
