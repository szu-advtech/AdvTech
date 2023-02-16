from torchvision.datasets import CIFAR10
from PIL import Image


class PreDataset(CIFAR10):
    def __getitem__(self, item):
        img,target=self.data[item],self.targets[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target


if __name__=="__main__":

    import config
    train_data = PreDataset(root='dataset', train=True, transform=config.train_transform, download=True)
    print(train_data[0])