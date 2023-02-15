import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open("1.jpeg")).unsqueeze(0).to(device)

#lb=["red envelop that was used in new year", "red envelop that was used in christmas", "envelop that was used in new year" ]
lb=['sad','Neutral','happiness']
text = clip.tokenize(lb).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

for i in range(len(lb)):
    print(lb[i]+' : {:.2%}'.format(probs[0][i]))


