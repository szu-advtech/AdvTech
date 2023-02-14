from six.moves import cPickle
import six
import numpy as np
df_mode='coco-test-words.p'
pkl_file = cPickle.load(open(df_mode, 'rb'), **(dict(encoding='latin1') if six.PY3 else {}))
ref_len = np.log(float(pkl_file['ref_len']))
document_frequency = pkl_file['document_frequency']
log=open('document_frequency.txt','w')
print(ref_len)
print(document_frequency,file=log)
log.close()