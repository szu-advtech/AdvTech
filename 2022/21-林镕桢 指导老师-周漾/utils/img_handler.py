from PIL import Image
import torch
from torchvision import transforms
from dataset.coco_handler import get_seg


class ImageHandler():
    def __init__(self, model, clip_model, clip_preprocess, k_means=None) -> None:
        
        self.model = model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.transform = transforms.Compose([
            transforms.Resize([256, 256], 
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def gen_img(self, label_path, img_feature):
        # target = Image.open("./target.png")
        # target = transforms.ToTensor()(target)
        # target = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(target)
        # target = target.unsqueeze(0).cuda()
        label = Image.open(label_path)
        seg = (self.transform(label)*255).long()
        label = seg[0,:,:]
        # transforms.ToPILImage()(label).save("test.png")
        label[label == 255] = 182
        label = label.unsqueeze(0).unsqueeze(0).cuda()
        inst = seg[1,:,:]
        inst = inst.unsqueeze(0).unsqueeze(0).cuda()
        seg = get_seg(0, label, inst)
        # img = model(seg, img=target)
        self.model.eval()
        img = self.model(seg, clip_feature=img_feature)
        img = ((img+1)/2).clamp_(0, 1)

        transforms.ToPILImage()(img.squeeze(0).cpu()).save("gen_img.png")

        # return img.squeeze(0).cpu()

    def get_inst(self, label):
        arr = []
        inst = torch.zeros_like(label)
        for i in range(label.size()[1]):
            for j in range(label.size()[2]):
                cur_pix = label[0, i, j].item()
                if cur_pix == 182:
                    continue
                if  cur_pix not in arr:
                    arr.append(cur_pix)
                cur_index = arr.index(cur_pix)+1
                inst[0, i, j] = cur_index

        return inst

    def get_edge(self, inst):
        edge = torch.zeros_like(inst)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (inst[:, :, :, 1:] != inst[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (inst[:, :, :, 1:] != inst[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (inst[:, :, 1:, :] != inst[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (inst[:, :, 1:, :] != inst[:, :, :-1, :])
        return edge.float()

    def get_feature(self, img_path):
        img = Image.open(img_path)
        img = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            clip_feature = self.clip_model.encode_image(img)

        return clip_feature.cuda()

    def gen_from_seg(self, label_path, img_feature):
        label = Image.open(label_path)
        label = (self.transform(label)*255).long()
        label[label == 255] = 182

        inst = self.get_inst(label)
        inst = inst.unsqueeze(0)
        edge = self.get_edge(inst)

        label = label.unsqueeze(0)
        seg = torch.zeros([label.shape[0], 183, *label.shape[2:]])
        seg = seg.scatter_(1, label, 1.0)
        seg = torch.cat([seg, edge], dim=1).cuda()


        self.model.eval()

        gen = self.model(seg, clip_feature=img_feature)
        gen = ((gen+1)/2).clamp_(0, 1)
        transforms.ToPILImage()(gen.squeeze(0).cpu()).save("gen_img.png")

