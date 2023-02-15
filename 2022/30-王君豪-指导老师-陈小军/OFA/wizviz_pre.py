import csv
from io import BytesIO
from PIL import Image
import base64
import json
import os
def img2base64(imgpath,cache):
    if imgpath in cache.keys():
        return cache[imgpath]
    else:
        img=Image.open(imgpath)
        img_buffer=BytesIO()
        img.save(img_buffer,format=img.format)
        byte_data = img_buffer.getvalue()
        base64_bytes = base64.b64encode(byte_data) # bytes
        base64_str = base64_bytes.decode("utf-8") # str
        cache[imgpath]=base64_str
    return base64_str

if __name__ == '__main__':
    split='val'
    json_path=split+'.json'
    data_dir=r'./dataset/wizviz'
    json_path=os.path.join(data_dir,'annotations',json_path)
    data_path=os.path.join(data_dir,split)
    img_base_cache={}
    save_dir='dataset/wizviz/annotations/'+split+'_wizviz.tsv'

    if split=='train':
        offset=0
    elif split=='val':
        offset=23431
    elif split=='test':
        offset=31181
    print(json_path)
    print(data_path)
    with open(json_path,'r',encoding='utf-8') as f:
        j=json.load(f)
        info=j['info']
        images=j['images']
        ann=j['annotations']
    total_sum=0
    for i,item in enumerate(ann):
        if '\n' in item['caption']:
            print(item['caption'])
            item['caption']=item['caption'].replace('\r',',').replace('\n',',')#win下用\r\n表示换行，所以在去除的时候\r\n需连用
            item['caption']=item['caption'].replace('-','')
            item['caption'] = item['caption'].replace('"', '')
            print(item['caption'])
        total_sum+=1
        item['file_path']='VizWiz_'+split+'_'+str(item['image_id']-offset).zfill(8)+'.jpg'
        item['abs_file_path']=os.path.join(data_path,item['file_path'])
        item['img']=img2base64(item['abs_file_path'],img_base_cache)

    with open(save_dir, 'w',newline="") as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in range(total_sum):
            item=ann[i]
            tsv_w.writerow([item['id'],item['caption'],item['img']])
