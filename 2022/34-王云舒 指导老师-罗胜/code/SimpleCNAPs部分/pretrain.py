import os
import random
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

#from simple_cnaps_src import resnet_new_12
from simple_cnaps_src import resnet_new
import benchmarks as benchmarks
from collections import OrderedDict
from path_index import *

def copyStateDict(state_dict):    
    if list(state_dict.keys())[0].startswith('module'):        
        start_idx = 1    
    else:        
        start_idx = 0    
    new_state_dict = OrderedDict()    
    for k,v in state_dict.items():        
        name = '.'.join(k.split('.')[start_idx:])        
        new_state_dict[name] = v    
    return new_state_dict
    
class Net_resnet_pretrained(nn.Module):
    def __init__(self, model):
        super(Net_resnet_pretrained, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, pretrain_trainways)
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x

def main(
        ways=pretrain_trainways,
        shots=1,
        cuda=True,
        lr=1e-3,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Tasksets using the benchmark interface
    # tasksets = benchmarks.get_tasksets_two('digits_normal',
    #                                    train_samples=2*shots,
    #                                    train_ways=ways,
    #                                    test_samples=2*shots,
    #                                    test_ways=ways,
    #                                    root='~/data',
    # )
    # tasksets = benchmarks.get_tasksets_act('act',
    #                                    'dsa',
    #                                    train_samples=2*shots,
    #                                    train_ways=ways,
    #                                    test_samples=2*shots,
    #                                    test_ways=ways,
    #                                    root='~/data',
    # )
    tasksets = benchmarks.get_tasksets('digits',
                                    train_samples=2*shots,
                                    train_ways=ways,
                                    test_samples=shots,
                                    test_ways=ways,
                                    root='~/data',
    )
    

    if USEPreModel == True:
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        if pretrain_kernel_size == 5:
            model.conv1=nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=3, bias=False) #5x5
        model.fc = nn.Linear(512, pretrain_trainways)
        print(model)
    else:
        
        # model = shufflenet_v2_x1_0(num_classes=10)
        
        model = resnet_new.resnet18()
        print(model)
        
        #import torchvision.models as models
        #model = models.resnet18(pretrained=False)
        #if pretrain_kernel_size == 5:
            #model.conv1=nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=3, bias=False) #5x5
        #model.fc = nn.Linear(512, pretrain_trainways)
        #print(model)
    
    optimizer = optim.Adam(model.parameters(), lr)
    lossfn = nn.CrossEntropyLoss(reduction='mean')

    device = torch.device('cuda')
    model.to(device)
    train = tasksets.train.dataset
    # testset = tasksets.validation.dataset
    trainsize = int(len(train)*0.7)
    testsize = len(train)-trainsize
    trainset, testset = random_split(train, [trainsize, testsize])
    trainloader = DataLoader(trainset, batch_size=8,shuffle=True)
    testloader = DataLoader(testset, batch_size=8,shuffle=True)

    minloss = 100
    it = 0
    allsince = time.time()
    for epoch in range(500):
 
        model.train()                                   
        for batchidx, (x, label, usrs, task) in enumerate(trainloader):
            #x: [b, 3, 32, 32]
            #label: [b]
            x = x.to(device)
            label = label.to(device)
            logits = model(x)                           #logits: [b, 10]
            loss = lossfn(logits, label)              

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()                      
        with torch.no_grad():
            
            total_correct = 0                                      
            total_num = 0
            for x, label, usrs, task in testloader:
                #x: [b, 3, 32, 32]
                #label: [b]
                x = x.to(device)
                label = label.to(device)
                logits = model(x)                              
                pred = logits.argmax(dim=1)          
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                
            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)

        if minloss > loss:
            minloss = loss
            it = 0
        else:
            it += 1
            if it > 20:
                print('stop at ', epoch)
                break

    allend = time.time() - allsince
    print(allend)
    print('All training complete in {:.0f}s'.format(allend))
    torch.save(model.state_dict(), pretrain_destination)



if __name__ == '__main__':
    main()