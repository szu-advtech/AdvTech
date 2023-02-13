import argparse
import codecs
import os
parser = argparse.ArgumentParser()
parser.add_argument('--ref_data_dir',                   type=str,   default='data/tinyshakespeare')
parser.add_argument('--gen_data_dir',                   type=str,   default='data/tinyshakespeare')
parser.add_argument('--test_number',                    type=int,   default=50)
args = parser.parse_args()


f = open('config.xml', 'w+')
    
f.write('<ROUGE-EVAL version="1.55">')

for i in range(9999999):
    if not os.path.exists('{}/gen_review.{}.txt'.format(args.gen_data_dir, i)):
        break
    peer_elems = "<P ID=\"{id}\">{name}</P>".format(id=None, name= 'gen_review.' + str(i) + '.txt')
    model_elems = "<M ID=\"{id}\">{name}</M>".format(id='A', name= 'true_review.A.' + str(i) + '.txt')
    eval_string = """
    <EVAL ID="{task_id}">
        <MODEL-ROOT>{model_root}</MODEL-ROOT>
        <PEER-ROOT>{peer_root}</PEER-ROOT>
        <INPUT-FORMAT TYPE="SPL">
        </INPUT-FORMAT>
        <PEERS>
            {peer_elems}
        </PEERS>
        <MODELS>
            {model_elems}
        </MODELS>
    </EVAL>
""".format(
            task_id=i,
            model_root=args.ref_data_dir, model_elems=model_elems,
            peer_root=args.gen_data_dir, peer_elems=peer_elems)
    f.write(eval_string) 
f.write("</ROUGE-EVAL>")
f.close()

