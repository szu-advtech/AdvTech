import torch
import models
from torchvision import datasets
from torchvision import transforms

batch_size=4

test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

classifier = models.create('alexnet', 200, False)
classifier.load_state_dict(torch.load('tiny200_ori_classifier.pth'))
classifier.eval()
file = open('accuracy.txt', 'w')

for qp in range(0, 52):
    test_set = datasets.ImageFolder(root=f'./dataset/ori_img_qp{qp}_dec/val', transform=test_trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)

    correct = 0
    miss = 0
    dataset_size = 0
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            output = classifier(input)
            pred = torch.argmax(output, dim=1)
            print(f'qp:{qp}, pred:{pred}')
            print(f'qp:{qp}, label:{label}')
            correct += pred.eq(label).sum().item()
            miss += label.shape[0] - pred.eq(label).sum().item()
            dataset_size += batch_size

    file.write(f'qp:{qp}\n')
    file.write(f'dataset_size:{dataset_size}\n')
    file.write(f'correct:{correct}\n')
    file.write(f'miss:{miss}\n')
    file.write(f'accuracy:{correct/dataset_size}\n\n')
    file.flush()

file.close()