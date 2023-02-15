"""
This script for split val label from original PubTabNet_2.0.0.jsonl
"""

import json
import json_lines
from tqdm import tqdm

class PubtabnetParser(object):
    def __init__(self, jsonl_path, split='val'):
        self.split = split
        self.jsonl_path = jsonl_path
        self.save_root = '/root/zf/DAVAR-Lab-OCR/output/PubTabNet_2.0.0_val.jsonl'

    @property
    def data_generator(self):
        return json_lines.reader(open(self.jsonl_path, 'rb'))

    def get_json(self, split='val'):
        with open(self.save_root, "a", encoding="utf-8") as writer:
            for item in tqdm(self.data_generator):
                if item['split'] == split:
                    json.dump(item, writer, ensure_ascii=False)
                    writer.write('\n')
                    print("{} have been written to json file".format(item['filename']))
        print("Finish")


if __name__ == '__main__':
    """

        
    """

    jsonl_path = r'/root/zf/TableMASTER-mmocr/data/pubtabnet/PubTabNet_2.0.0.jsonl'
    
    parser = PubtabnetParser(jsonl_path, split='val')

    parser.get_json(split='val')














