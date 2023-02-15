from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import matplotlib.pyplot as plt
import cv2
import time

t0 = time.time()
# Load image
image = Image.open("0.png")
# Load model
processor = CLIPSegProcessor.from_pretrained("./clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("./clipseg-rd64-refined")
# processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Prepare image and texts for model
prompts = ["background", "a blue block", "a orange block", "a yellow block", "a red block"]
inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# Forward pass
## Next, let's run a forward pass and visualize the predictions the model made.
# predict
with torch.no_grad():
  outputs = model(**inputs)
  
logits = outputs.logits
print(logits.shape)

preds = outputs.logits.unsqueeze(1)

# # visualize prediction
# _, ax = plt.subplots(1, 5, figsize=(15, 4))
# [a.axis('off') for a in ax.flatten()]
# ax[0].imshow(image)
# [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
# [ax[i+1].text(0, -15, prompts[i]) for i in range(4)]

# Convert to binary mask
# One can apply a sigmoid activation function on the predicted mask 
# and use some OpenCV (cv2) to turn it into a binary mask.

filename = f"mask.png"
# here we save the second mask
# for i in range(len(prompts)):
plt.imsave(filename,torch.sigmoid(preds[0][0]))

img2 = cv2.imread(filename)

gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# fix color format
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

im = Image.fromarray(bw_image)
im.save("mask2.png")

t1 = time.time()
print("time:", t1 - t0)