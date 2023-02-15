"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter

def graph_train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, epoch_num, update_cluster_head_only=False):
    """
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, contrastive_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    # from utils.utils import get_features_train
    # outputs, targets = get_features_train(train_loader, model)


    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    cl = []
    cl2 = []
    tl = []

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)

        # network output
        if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

        # Loss for every head
        total_loss, consistency_loss,contrastive_loss, entropy_loss = [], [], [],[]
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, contrastive_loss_, entropy_loss_ = criterion(anchors_output_subhead, neighbors_output_subhead, epoch, epoch_num)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            contrastive_loss.append(contrastive_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        contrastive_losses.update(np.mean([v.item() for v in contrastive_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))


        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # if i == (len(train_loader) - 1):
        #     progress.display(i + 1)
        if i % 25 == 0:
            progress.display(i)

        tl.append(round(total_loss.item(), 3))
        cl.append(round(np.mean([v.item() for v in consistency_loss]), 3))
        cl2.append(round(np.mean([v.item() for v in contrastive_loss]), 3))

    f_total_loss = round(np.mean([v for v in tl]), 3)
    f_consistency_loss = round(np.mean([v for v in cl]), 3)
    f_contrastive_loss = round(np.mean([v for v in cl2]), 3)

    final_loss = [f_total_loss, f_consistency_loss, f_contrastive_loss]

    return final_loss


def selflabel_train(train_loader, model, criterion, optimizer, epoch, epoch_num, ema=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                             prefix="Epoch: [{}]".format(epoch + 1))

    model.train()

    tl = []
    sum = 0
    predicitons = []
    labels = []
    for i, batch in enumerate(train_loader):

        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        if i % 5 == 0:
            from utils.utils import get_features_train
            outputs, targets = get_features_train(train_loader, model)
            # features, _ = get_features_train(train_loader, model, forward_pass='backbone')

        indices = batch['index']
        neighbors_indices = batch['anchor_neighbors_indices']
        anchor_output = outputs[indices].cuda(non_blocking=True)
        neighbors_output = outputs[neighbors_indices].cuda(non_blocking=True)
        # neighbors_features = features[neighbors_indices].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
            # images_features = model(images, forward_pass = 'default')
        output_augmented = model(images_augmented)[0]

        loss, prediction, label, conf_num = criterion(output, output_augmented, anchor_output, neighbors_output,  batch['target'], batch['index'])
        losses.update(loss.item())
        predicitons.append(prediction)
        labels.append(label)
        sum += conf_num

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:  # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        # if i == (len(train_loader) - 1):
        #     progress.display(i + 1)
        if i % 25 == 0:
            progress.display(i)

        tl.append(round(loss.item(), 3))

    f_total_loss = round(np.mean([v for v in tl]), 3)
    final_loss = [f_total_loss]

    N = len(train_loader.dataset)

    # count true pseudo-label
    predictions = torch.cat(predicitons, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    from utils.evaluate_utils import _hungarian_match
    match = _hungarian_match(predictions, labels, preds_k=10, targets_k=10)
    reordered_preds = torch.zeros(len(predictions), dtype=predictions.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
    pseudo_label_acc = int((reordered_preds == labels).sum()) / float(N)

    return final_loss, round(1.0*sum/N, 3), round(pseudo_label_acc, 3)


# def label_propagation():



