import torch


def train(model, train_data, optimizer,epoch):
    model.train()

    for epoch in range(0, epoch):
        model.zero_grad()
        score,x,y = model(train_data)
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss(score, train_data['cd_matix'].cuda())
        loss.backward()
        optimizer.step()
        print(loss.item())
    # torch.save(model.state_dict(), '../')
    return model

