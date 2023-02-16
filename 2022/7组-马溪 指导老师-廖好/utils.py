import numpy as np

class ABSADataset():
    def __init__(self, fname, source_word2idx, opt):
        max_length = opt.max_sentence_len
        lm = opt.lm
        topk = opt.topk

        data = []
        review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
        ae_data = open(fname + r'target.txt', 'r', encoding='utf-8').readlines()
        if lm == 'internal':
            forward_lm_data = open(fname + r'internal_forward_top10.txt', 'r', encoding='utf-8').readlines()
            backward_lm_data = open(fname + r'internal_backward_top10.txt', 'r', encoding='utf-8').readlines()
        elif lm == 'external':
            forward_lm_data = open(fname + r'external_forward_top10.txt', 'r', encoding='utf-8').readlines()
            backward_lm_data = open(fname + r'external_backward_top10.txt', 'r', encoding='utf-8').readlines()
        elif lm == 'bert_base':
            lm_data = open(fname + r'bert_base_top10.txt', 'r', encoding='utf-8').readlines()
        elif lm == 'bert_pt':
            lm_data = open(fname + r'bert_pt_top10.txt', 'r', encoding='utf-8').readlines()
        else:
            pass
        for index, _ in enumerate(review):
            '''
            Word Index
            '''
            sptoks = review[index].strip().split()

            idx = []
            mask = []
            len_cnt = 0
            for sptok in sptoks:
                if len_cnt < max_length:
                    idx.append(source_word2idx[sptok.lower()])
                    mask.append(1.)
                    len_cnt += 1
                else:
                    break

            source_data_per = (idx + [0] * (max_length - len(idx)))
            source_mask_per = (mask + [0.] * (max_length - len(idx)))

            ae_labels = ae_data[index].strip().split()
            aspect_label = []
            for l in ae_labels:
                l = int(l)
                if l == 0 :
                    aspect_label.append([1, 0, 0])
                elif l == 1:
                    aspect_label.append([0, 1, 0])
                elif l == 2:
                    aspect_label.append([0, 0, 1])
                else:
                    raise ValueError

            aspect_y_per = (aspect_label + [[0, 0, 0]] * (max_length - len(idx)))

            if lm in ['internal', 'external']:
                'Forward'
                forward_segments = forward_lm_data[index].strip().split('###')
                forward_words_list = []
                forward_probs_list = []
                for forward_segment in forward_segments:
                    forward_pairs = forward_segment.split('@@@')
                    forward_words = []
                    forward_probs = []
                    topk_cnt = 0
                    for forward_pair in forward_pairs:
                        if topk_cnt >= topk:
                            break
                        forward_word = source_word2idx[forward_pair.split()[0]]
                        forward_prob = float(forward_pair.split()[1])
                        forward_words.append(forward_word)
                        forward_probs.append(forward_prob)
                        topk_cnt += 1
                    forward_words_list.append(forward_words)
                    forward_probs_list.append(forward_probs)
                forward_words_per = (forward_words_list + [[0]*topk] * (max_length - len(idx)))
                forward_probs_per = (forward_probs_list + [[0.]*topk] * (max_length - len(idx)))
    
                'Backward'
                backward_segments = backward_lm_data[index].strip().split('###')
                backward_words_list = []
                backward_probs_list = []
                for backward_segment in backward_segments:
                    backward_pairs = backward_segment.split('@@@')
                    backward_words = []
                    backward_probs = []
                    topk_cnt = 0
                    for backward_pair in backward_pairs:
                        if topk_cnt >= topk:
                            break
                        backward_word = source_word2idx[backward_pair.split()[0]]
                        backward_prob = float(backward_pair.split()[1])
                        backward_words.append(backward_word)
                        backward_probs.append(backward_prob)
                        topk_cnt += 1
                    backward_words_list.append(backward_words)
                    backward_probs_list.append(backward_probs)
                backward_words_per = (backward_words_list + [[0]*topk] * (max_length - len(idx)))
                backward_probs_per = (backward_probs_list + [[0.]*topk] * (max_length - len(idx)))
    
                data_per = {'x': np.array(source_data_per, dtype='int64'),
                            'mask': np.array(source_mask_per, dtype='float32'),
                            'aspect_y': np.array(aspect_y_per, dtype='int64'),
                            'fw_lmwords':np.array(forward_words_per, dtype='int64'),
                            'fw_lmprobs':np.array(forward_probs_per, dtype='float32'),
                            'bw_lmwords':np.array(backward_words_per, dtype='int64'),
                            'bw_lmprobs':np.array(backward_probs_per, dtype='float32')}
            elif lm in ['bert_base', 'bert_pt']:
                'Backward'
                segments = lm_data[index].strip().split('###')
                words_list = []
                probs_list = []
                for segment in segments:
                    pairs = segment.split('@@@')
                    words = []
                    probs = []
                    topk_cnt = 0
                    for pair in pairs:
                        if topk_cnt >= topk:
                            break
                        word = source_word2idx[pair.split()[0]]
                        prob = float(pair.split()[1])
                        words.append(word)
                        probs.append(prob)
                        topk_cnt += 1
                    words_list.append(words)
                    probs_list.append(probs)
                words_per = (words_list + [[0] * topk] * (max_length - len(idx)))
                probs_per = (probs_list + [[0.] * topk] * (max_length - len(idx)))

                data_per = {'x': np.array(source_data_per, dtype='int64'),
                            'mask': np.array(source_mask_per, dtype='float32'),
                            'aspect_y': np.array(aspect_y_per, dtype='int64'),
                            'lmwords': np.array(words_per, dtype='int64'),
                            'lmprobs': np.array(probs_per, dtype='float32')}
            else:
                data_per = {'x': np.array(source_data_per, dtype='int64'),
                            'mask': np.array(source_mask_per, dtype='float32'),
                            'aspect_y': np.array(aspect_y_per, dtype='int64')}
                
            data.append(data_per)
        self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

