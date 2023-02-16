import torch
import clip
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import cv2

def resize(img, size=(224,224)):
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    # img = center_crop(img, size)
    return img

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def _transform():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class VisualMatcher(object):
	def __init__(self) -> None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model, preprocess = clip.load("ViT-B/32", device=device)

		self.model = model
		self.preprocess = preprocess
		self.device = device
	
	def convert_images(self, images):
		data = []
		for image in images:
			img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
			data.append(img_tensor)
		data = torch.cat(data, dim=0)
		return data
	
	def convert_rgb_arr(self, images):
		data = []
		for image in images:
			image = resize(image)
			img_tensor = _transform()(image).unsqueeze(0).to(self.device)
			data.append(img_tensor)
		data = torch.cat(data, dim=0)
		return data
		
	def convert_texts(self, object_labels):
		texts = []
		for label in object_labels:
			texts.append( f"A picture of a {label}" )

		data = clip.tokenize(texts).to(self.device)
		return data
	
	def match_images(self, source_images, target_images, object_labels, use_text=True):
		with torch.no_grad():

			if type(source_images[0]) == np.ndarray:
				convert_fn = self.convert_rgb_arr
			else:
				convert_fn = self.convert_images

			source_data = convert_fn(source_images)
			target_data = convert_fn(target_images)

			source_features = self.model.encode_image(source_data)
			target_features = self.model.encode_image(target_data)

			source_features /= source_features.norm(dim=-1, keepdim=True)
			target_features /= target_features.norm(dim=-1, keepdim=True)

			if use_text:
				text_data = self.convert_texts(object_labels)
				text_features = self.model.encode_text(text_data)
				text_features /= text_features.norm(dim=-1, keepdim=True)

				source_text = (100.0 * source_features @ text_features.T).softmax(dim=-1)
				target_text = (100.0 * target_features @ text_features.T).softmax(dim=-1)
				source_target = (100.0 * source_text @ target_text.T).softmax(dim=-1).cpu().numpy()

			else:
				source_target = (100.0 * source_features @ target_features.T).softmax(dim=-1).cpu().numpy()

		# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
		source_target = -source_target
		source_ids, target_ids = linear_sum_assignment(source_target)

		return source_ids, target_ids
	
	def match(self, source_rgb, source_dep, target_rgb, target_dep, texts ):
		# UOC app
		pass
