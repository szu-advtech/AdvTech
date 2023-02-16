import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import utils


def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, train_loss_c_list, train_loss_s_list, valid_loss_list, valid_loss_c_list,
                 valid_loss_s_list, global_steps_list):
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'train_loss_c_list': train_loss_c_list,
                  'train_loss_s_list': train_loss_s_list,
                  'valid_loss_list': valid_loss_list,
                  'valid_loss_c_list': valid_loss_c_list,
                  'valid_loss_s_list': valid_loss_s_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path is None:
        return

    metrics_state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return metrics_state_dict


# a is the weight alpha in the final loss
def train(model,
          optimizer,
          device,
          train_loader,
          valid_loader,
          file_path,
          criterion=nn.CrossEntropyLoss(),
          a=0.2,
          num_epochs=100,
          best_valid_loss=float("Inf")):

    running_loss = 0.0
    running_c_loss = 0.0
    running_s_loss = 0.0
    valid_running_loss = 0.0
    valid_running_c_loss = 0.0
    valid_running_s_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    train_loss_c_list = []
    valid_loss_c_list = []
    train_loss_s_list = []
    valid_loss_s_list = []
    global_steps_list = []

    valid_acc = 0.0
    memory_bank = pd.DataFrame({'Id': [], 'memory': []})

    eval_every = len(train_loader) // 2
    early_stop_count = 0
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1} training"):
            Id, label, i11, i12, i13, i14, i15, i01, i02, i03, i04, i05 = batch
            label = label.type(torch.LongTensor)
            label = label.to(device)
            Id = Id.to(device)
            output, x = model(Id)
            memory_bank = utils.write_memory(Id, x, memory_bank)
            sim1 = utils.consim(x, i11, i12, i13, i14, i15, memory_bank, device)
            sim0 = utils.consim(x, i01, i02, i03, i04, i05, memory_bank, device)
            sim = (sim0 - sim1) / 2
            # print(f"Debug | shape: {output.shape} and {label.shape}")
            loss_c = criterion(output, label.view(-1))
            loss_s = ((label - 1) * sim + label * sim) + 0.5
            loss_s = torch.mean(loss_s)
            loss = (1 - a) * loss_c + a * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_c_loss += loss_c.item()
            running_s_loss += loss_s.item()
            global_step += 1

            # validation
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for batch in tqdm(valid_loader, desc="validation"):
                        Id, label, i11, i12, i13, i14, i15, i01, i02, i03, i04, i05 = batch
                        label = label.type(torch.LongTensor)
                        label = label.to(device)
                        Id = Id.to(device)
                        output, x = model(Id)
                        memory_bank = utils.write_memory(Id, x, memory_bank)
                        sim1 = utils.consim(x, i11, i12, i13, i14, i15, memory_bank, device)
                        sim0 = utils.consim(x, i01, i02, i03, i04, i05, memory_bank, device)
                        sim = (sim0 - sim1) / 2
                        loss_c = criterion(output, label.view(-1))
                        loss_s = ((label - 1) * sim + label * sim) + 0.5
                        loss_s = torch.mean(loss_s)
                        loss = (1 - a) * loss_c + a * loss_s
                        valid_running_loss += loss.item()
                        valid_running_c_loss += loss_c.item()
                        valid_running_s_loss += loss_s.item()

                        # acc
                        valid_acc += utils.flat_accuracy(output.detach().cpu().numpy(), label.detach().cpu().numpy())

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_train_c_loss = running_c_loss / eval_every
                average_valid_c_loss = valid_running_c_loss / len(valid_loader)
                average_train_s_loss = running_s_loss / eval_every
                average_valid_s_loss = valid_running_s_loss / len(valid_loader)
                valid_acc = valid_acc / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                train_loss_c_list.append(average_train_c_loss)
                valid_loss_c_list.append(average_valid_c_loss)
                train_loss_s_list.append(average_train_s_loss)
                valid_loss_s_list.append(average_valid_s_loss)
                global_steps_list.append(global_step)

                # print progress
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Loss (C): {:.4f},Train Loss (S): {:.4f},Valid Loss: {:.4f},Valid Loss (C)): {:.4f},Valid Loss (S): {:.4f}'
                    .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                            average_train_loss, average_train_c_loss, average_train_s_loss,
                            average_valid_loss, average_valid_c_loss, average_valid_s_loss))
                print(f"Validation acc: {valid_acc}")

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                running_c_loss = 0.0
                valid_running_c_loss = 0.0
                running_s_loss = 0.0
                valid_running_s_loss = 0.0
                valid_acc = 0.0
                model.train()

                # checkpoint
                # 加上 early stop
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model_BTIC_Swin.pth', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics_BTIC_Swin.pth', train_loss_list, train_loss_c_list,
                                 train_loss_s_list, valid_loss_list, valid_loss_c_list, valid_loss_s_list,
                                 global_steps_list)
                else:
                    early_stop_count += 1
                    if early_stop_count >= 5:
                        # 结束训练
                        break
        if early_stop_count >= 5:
            break

    save_metrics(file_path + '/' + 'metrics_BTIC_Swin.pth', train_loss_list, train_loss_c_list, train_loss_s_list,
                 valid_loss_list, valid_loss_c_list, valid_loss_s_list, global_steps_list)
    print('Finished Training!')


def visualize_train_process(savepath, device):
    metrics_state_dict = load_metrics(savepath + '/metrics_BTIC_Swin.pth', device)
    global_steps_list = metrics_state_dict['global_steps_list']

    plt.plot(global_steps_list, metrics_state_dict['train_loss_list'], label='Train')
    plt.plot(global_steps_list, metrics_state_dict['valid_loss_list'], label='Valid')
    plt.plot(global_steps_list, metrics_state_dict['train_loss_c_list'], label='Train_criterion')
    plt.plot(global_steps_list, metrics_state_dict['valid_loss_c_list'], label='Valid_criterion')
    plt.plot(global_steps_list, metrics_state_dict['train_loss_s_list'], label='Train_similarity')
    plt.plot(global_steps_list, metrics_state_dict['valid_loss_s_list'], label='Valid_similarity')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(savepath + '/T&Vloss_BTIC_Swin.jpg', dpi=300)
    plt.show()


def evaluate(model, test_loader, device, savepath):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluation"):
            Id, label, i11, i12, i13, i14, i15, i01, i02, i03, i04, i05 = batch
            label = label.type(torch.LongTensor)
            label = label.to(device)
            Id = Id.to(device)
            output, x = model(Id)

            output = output.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            y_pred.extend(np.argmax(output, axis=1).flatten())
            y_true.extend(label.flatten())
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=["fake", "true"], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    p2 = sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(["fake", "true"])
    ax.yaxis.set_ticklabels(["fake", "true"])
    s2 = p2.get_figure()
    s2.savefig(savepath + '/HeatMap_BTIC_Swin.jpg', dpi=300)
