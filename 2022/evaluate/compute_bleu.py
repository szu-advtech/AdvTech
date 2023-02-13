from nltk.translate.bleu_score import corpus_bleu
import codecs
from rouge import rouge


#test_num = 3693
#City2
test_num = 745
dir_path = '../City4_log/dianping/RAW_MSE_CAML_FN_FM/18-12-18-33-07-LBA/expalanation/'
#获取gen和true文件
gen_sentence = []
true_sentence = []
for i in range(test_num):
	gen_filename = dir_path + 'gen_review.'+str(i)+'.txt'
	true_filename = dir_path + 'true_review.A.'+str(i)+'.txt'
	gen_file = codecs.open(gen_filename, 'rb', 'utf-8').readlines()
	true_file = codecs.open(true_filename, 'rb', 'utf-8').readlines()
	
	for j in range(len(gen_file)):
		gen_str = gen_file[j]
		gen_split = gen_str.split(' ')
		new_sentence = []
		for k in range(len(gen_split)):
			new_sentence.append(gen_split[k])
		gen_sentence.append(new_sentence)
		true_str = true_file[j]
		true_split = true_str.split(' ')
		new_sentence = []
		for k in range(len(true_split)):
			new_sentence.append(true_split[k])
		true_sentence.append([new_sentence])
#print(gen_sentence)

#true_sentence = [['我','不喜欢','这间','店','服务员','的','态度','太差','。'],['菜','太','咸','了','，','而且','分量','很少','，','很贵','。']]
#gen_sentence = [['我','去过','一次','感觉','一般','。']]


score = corpus_bleu(true_sentence, gen_sentence, weights=(0.25, 0.25, 0.25, 0.25))
print('BLEU score:'+str(score))
print(rouge(gen_sentence, true_sentence))
with open('City4_15-12-14-05-20-UniLM_bleu_result.txt', 'w+') as bleu_result:
	bleu_result.write('BLEU score:'+str(score))
bleu_result.close()
		
