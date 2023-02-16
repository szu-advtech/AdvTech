#import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
#import matplotlib.pyplot as plt

# make gifs
def make_gif(path, num):
	images = []
	font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-MI.ttf', size=20)
	for i in range(num):
		file = path + 'fake_samples_epoch_%03d.png' %(i)
		src_image = Image.open(file)
		new_img = Image.new('RGB', (64*9,64*9), 'white')
		new_img.paste(src_image,(25,10))
		draw = ImageDraw.Draw(new_img)
		draw.text((64*4,64*8+35),'Epoch=%d'%(i+1), font=font, fill='blue')
		new_img.save('test.jpg')
		images.append(imageio.imread('test.jpg'))

	imageio.mimsave(path+'fake_images.gif', images, fps=5)

make_gif('checkpoints/LSGAN/Image/',45)
