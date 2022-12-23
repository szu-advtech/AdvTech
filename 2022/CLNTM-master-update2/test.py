import os
import file_handling as fh
import compute_tfidf


input_dir = 'D:/Code/Python/CLNTM-master/data/20ng/processed/'
input_file = 'train'
compute_tfidf.set_tfidf(input_dir,input_file)

tfidf = compute_tfidf.get_tfidf(input_dir, input_file)


print(tfidf)