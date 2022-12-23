--  Copyright (c) 2015, Omkar M. Parkhi
--  All rights reserved.

require 'image'
require 'nn'
net = torch.load('./VGG_FACE.t7')
net:evaluate()
im = image.load('./ak.png',3,'float')
im = im*255
mean = {129.1863,104.7624,93.5940}
for i=1,3 do im_bgr[i]:add(-mean[i]) end
im_bgr = im:index(1,torch.LongTensor{3,2,1})
prob = net(im_bgr)
maxval,maxid = prob:max(1)
print(maxid)
