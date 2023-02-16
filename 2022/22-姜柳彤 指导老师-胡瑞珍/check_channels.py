import os.path

import torch
import matplotlib.pyplot as plt
import material_decomposition_resnetUnet as mdr
from PIL import Image
from torchvision.transforms import transforms

transform = transforms.Compose([
    # transforms.Resize([256, 256]),
    transforms.ToTensor()
])

# read img

# render image
render = Image.open(os.path.join('D:/dataset/abo-benchmark-material/train/B000S6N026/render/0/render_0.jpg'))

# base_color
base_color = Image.open(os.path.join('D:/dataset/abo-benchmark-material/train/B000S6N026/base_color/base_color_0.jpg'))
base_color = transform(base_color)
# split channels
base_color = base_color.chunk(3)

# metallic_roughness
metallic_roughness = Image.open(os.path.join('D:/dataset/abo-benchmark-material/train/B000S6N026/metallic_roughness'
                                             '/metallic_roughness_0.jpg'))
# split channels
results = transform(metallic_roughness).chunk(3)

# normal
normal = Image.open(os.path.join('D:/dataset/abo-benchmark-material/train/B000S6N026/normal/normal_0.png'))
# split channels
normal = transform(normal)


# channel to image
toPIL = transforms.ToPILImage()
result0 = toPIL(results[0])
result1 = toPIL(results[1])
result2 = toPIL(results[2])
result0.save('test/channel0.jpg')
result1.save('test/channel1.jpg')
result2.save('test/channel2.jpg')
base_color0 = toPIL(base_color[0])
base_color1 = toPIL(base_color[1])
base_color2 = toPIL(base_color[2])
normal = toPIL(normal)


# show
fig = plt.figure()
axes = fig.subplots(3, 3)
axes[0, 0].imshow(result0)
axes[0, 1].imshow(result1)
axes[0, 2].imshow(result2)
axes[1, 0].imshow(base_color0)
axes[1, 1].imshow(base_color1)
axes[1, 2].imshow(base_color2)
axes[2, 0].imshow(normal)
plt.show()