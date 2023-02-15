import torch

def dataset_split(datasets, classes_labels, images_ids=None):
    try:
        targets = datasets.targets
    except:
        # get class labels for dataset svhn
        targets = datasets.labels

    ids = []
    if images_ids == None:
        for i, target in enumerate(targets):
            if target in classes_labels:
                ids += [i]
    else:
        for i in images_ids.keys():
            if int(i) in classes_labels:
                ids += images_ids[i]

    return torch.utils.data.Subset(datasets, ids)


def CACLoss(distances, gt, opt, device):
    true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
    non_gt = torch.Tensor(
        [[i for i in range(opt.num_classes) if gt[x] != i] for x in range(len(distances))]).long().to(device)
    others = torch.gather(distances, 1, non_gt)

    anchor = torch.mean(true)

    tuplet = torch.exp(-others + true.unsqueeze(1))
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

    total = opt._lambda * anchor + tuplet

    return total, anchor, tuplet