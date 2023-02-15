import json
import os

json_path = "./split_train/lvis_common.json"
save_path = "./split_train/lvis_common_category.json"
# image_folder = "./val2017"


with open(json_path, 'r') as f:
    data = json.load(f)

cats = data['categories']
annos = data['annotations']
images = data['images']

common_data = {}
all_valid_ids = []

for cat in cats:
    # common, freq, rare
    if cat['frequency'] == 'c' and cat['image_count'] > 1:
        all_valid_ids.append(cat['id'])

        common_data[ cat['id'] ] = {
            'name': cat['name'],
            'synset': cat['synset'],
            'images': [],
            'bboxes': []
        }
        
for anno in annos:
    cat_id = anno['category_id']
    if cat_id in all_valid_ids and anno['area'] > 32*32:
        # image_path = os.path.join( image_folder, '%012d.jpg' % anno['image_id'] )
        # if os.path.exists(image_path):
        if True:
            common_data[cat_id]['images'].append(
                anno['image_id']
            )
            common_data[cat_id]['bboxes'].append(
                anno['bbox']
            )

# remove empty list
cat_ids = list(common_data.keys())
for cat_id in cat_ids:
    if len(common_data[cat_id]['images']) < 10:
        common_data.pop(cat_id)

print(len(common_data.keys()))

with open(save_path, "w") as fp:
    json.dump(common_data, fp)
