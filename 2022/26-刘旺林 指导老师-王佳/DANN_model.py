
import torch.utils.data


loss_class = torch.nn.MSELoss()

def test(input, targets):
    cuda = True
    alpha = 0
    my_net = torch.load('DANN_model.pt')
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    t_img = input
    t_label = targets

    if cuda:
        t_img = t_img.cuda()
        t_label = t_label.cuda()

    class_output, _ = my_net(input_data=t_img, alpha=alpha)
    loss = loss_class(class_output,t_label)

    return loss