
def ali_cleaning_and_statistics(path, part, data, feature1, feature2, sep, single=True):
    random.seed(0)
    np.random.seed(0)

    # To be consistent with EUEN processing, we do not remove redundant samples
    nums_remove = 0
    sample_size, nums_treatment, nums_control, visit_treatment, visit_control = 0, 0, 0, 0, 0
    key1_seq_set, key1_seq_len = [set(), set(), set(), set(), set(), set(), set(), set()], [0, 0, 0, 0, 0, 0, 0, 0]
    key2_seq_set, key2_seq_len = set(), 0

    log_feature_dict1, log_feature_dict2, log_feature_dict3, log_feature_dict4 = dict(), dict(), dict(), dict()
    log_feature_idx1, log_feature_idx2, log_feature_idx3, log_feature_idx4 = 0, 0, 0, 0

    cleaned_data = []
    for p in part:
        k1 = pd.read_csv(path + p + feature1, sep=sep, header=0)
        d1 = k1.set_index('key1').agg(list, 1).to_dict()
        print('key feature1 load')

        k2 = pd.read_csv(path + p + feature2, sep=sep, header=0)
        d2 = k2.set_index('key2').agg(list, 1).to_dict()
        print('key feature2 load')

        with open(path + p + data, 'r', encoding='utf-8') as f:
            for line in f:
                if 'sample_id' in line:
                    continue

                _line = line.split(sep)
                # Remove samples whose features are not in key1 and key2
                if (int(_line[1]) not in d1) or (int(_line[2]) not in d2):
                    nums_remove += 1
                    sample_size += 1
                    if sample_size % 1000000 == 0:
                        print('{0} have processed'.format(sample_size))
                    continue

                if _line[10] == '0':
                    nums_control += 1
                    if _line[3] == '1':
                        visit_control += 1
                else:
                    nums_treatment += 1
                    if _line[3] == '1':
                        visit_treatment += 1

                sample_size += 1
                if sample_size % 1000000 == 0:
                    print('{0} have processed'.format(sample_size))

                # count the maximum number and maximum length of variable-length features in key1
                key1 = d1[int(_line[1])]
                for i in range(8):
                    if not pd.isnull(key1[i]):
                        key1_seq_set[i].update(key1[i].split(','))
                        if key1_seq_len[i] < len(key1[i].split(',')):
                            key1_seq_len[i] = len(key1[i].split(','))

                # count the maximum number and maximum length of variable-length features in key2
                key2 = d2[int(_line[2])]
                if not pd.isnull(key2[0]):
                    key2_seq_set.update(key2[0].split(','))
                    if key2_seq_len < len(key2[0].split(',')):
                        key2_seq_len = len(key2[0].split(','))

                # save cleaned data
                if _line[5] not in log_feature_dict1:
                    log_feature_dict1[_line[5]] = str(log_feature_idx1)
                    log_feature_idx1 += 1
                if _line[6] not in log_feature_dict2:
                    log_feature_dict2[_line[6]] = str(log_feature_idx2)
                    log_feature_idx2 += 1
                if _line[7] not in log_feature_dict3:
                    log_feature_dict3[_line[7]] = str(log_feature_idx3)
                    log_feature_idx3 += 1
                if _line[8] not in log_feature_dict4:
                    log_feature_dict4[_line[8]] = str(log_feature_idx4)
                    log_feature_idx4 += 1

                cleaned_data.append([_line[3], _line[10], log_feature_dict1[_line[5]], log_feature_dict2[_line[6]],
                                     log_feature_dict3[_line[7]], log_feature_dict4[_line[8]], _line[1], _line[2]])
        if single:
            cleaned_data = np.array(cleaned_data)
            index = np.arange(np.size(cleaned_data, 0))
            np.random.shuffle(index)
            train = cleaned_data[:int(0.13 * len(index)), :]
            valid = cleaned_data[int(0.13 * len(index)):int(0.15 * len(index)), :]
            np.savetxt(path + p + 'train_' + data, train, delimiter=';', fmt='%s')
            np.savetxt(path + p + 'valid_' + data, valid, delimiter=';', fmt='%s')
            print('train size {},{}'.format(np.size(train, 0), np.size(train, 1)))
            print('valid size {},{}'.format(np.size(valid, 0), np.size(valid, 1)))
        else:
            np.savetxt(path + p + 'cleaned_' + data, cleaned_data, delimiter=';', fmt='%s')

    print('Removed Data: {0}'.format(nums_remove))
    print('Size: {0}'.format(sample_size))
    print('Ratio of Treatment to Control: {0}'.format(nums_treatment / nums_control))
    print('Average Visit Ratio: {0}'.format((visit_control + visit_treatment) / sample_size))
    uplift_treatment = visit_treatment / nums_treatment
    uplift_control = visit_control / nums_control
    print('Relative Average Uplift: {0}'.format((uplift_treatment - uplift_control) / uplift_control))
    print('Average Uplift: {0}'.format(uplift_treatment - uplift_control))
    print('key1_seq_max: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'.format(len(key1_seq_set[0]), len(key1_seq_set[1]),
                                                                        len(key1_seq_set[2]), len(key1_seq_set[3]),
                                                                        len(key1_seq_set[4]), len(key1_seq_set[5]),
                                                                        len(key1_seq_set[6]), len(key1_seq_set[7])))
    print('key1_seq_len: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'.format(key1_seq_len[0], key1_seq_len[1],
                                                                        key1_seq_len[2], key1_seq_len[3],
                                                                        key1_seq_len[4], key1_seq_len[5],
                                                                        key1_seq_len[6], key1_seq_len[7]))
    print('key2_seq_max: {0}'.format(len(key2_seq_set)))
    print('key2_seq_len: {0}'.format(key2_seq_len))


def transform_idx_of_features(path, part, data, feature1, feature2, sep):
    key1_feature_dict = [dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(),
                         dict()]
    key1_feature_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    key1_feature_max = [3311, 1093, 556, 243, 234, 125, 28010, 23088]

    key2_feature_dict = [dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()]
    key2_feature_idx = [0, 0, 0, 0, 0, 0, 0, 0]
    key2_feature_max = 163

    for p in part:
        log_data = pd.read_csv(path + p + 'train_' + data, sep=sep,
                               names=['label', 't', 'f1', 'f2', 'f3', 'f4', 'key1', 'key2'], header=None)
        key_set = set(log_data['key1'])
        key_set2 = set(log_data['key2'])

        log_data = pd.read_csv(path + p + 'valid_' + data, sep=sep,
                               names=['label', 't', 'f1', 'f2', 'f3', 'f4', 'key1', 'key2'], header=None)
        key_set = set(log_data['key1']) | key_set
        key_set2 = set(log_data['key2']) | key_set2

        print('key1 num', len(key_set))
        print('key2 num', len(key_set2))

        transformed_key1_file = open(path + p + 'transformed_' + feature1, 'w')
        transformed_key2_file = open(path + p + 'transformed_' + feature2, 'w')

        sample_size = 0
        with open(path + p + feature1, 'r', encoding='utf-8') as f:
            for line in f:
                if 'key1' in line:
                    continue

                transformed_line = ''
                _line = line.split(sep)
                if int(_line[0]) in key_set:
                    transformed_line += _line[0] + ';'

                    for i in range(1, 9):
                        if not pd.isnull(_line[i]):
                            l = _line[i].replace(':1.0', '').split(',')
                            for j in range(len(l)):
                                if l[j] not in key1_feature_dict[ i -1]:
                                    key1_feature_dict[ i -1][l[j]] = str(key1_feature_idx[ i -1])
                                    key1_feature_idx[ i -1] += 1
                                if j != len(l ) -1:
                                    transformed_line += key1_feature_dict[ i -1][l[j]] + ','
                                else:
                                    transformed_line += key1_feature_dict[i - 1][l[j]] + ';'

                        else:
                            transformed_line += str(key1_feature_max[i]) + ';'

                    for i in range(9, 14):
                        if _line[i] not in key1_feature_dict[ i -1]:
                            key1_feature_dict[ i -1][_line[i]] = str(key1_feature_idx[ i -1])
                            key1_feature_idx[ i -1] += 1
                        if i != 13:
                            transformed_line += key1_feature_dict[ i -1][_line[i]] + ';'
                        else:
                            transformed_line += key1_feature_dict[ i -1][_line[i]]

                    transformed_key1_file.write(transformed_line + '\n')

                sample_size += 1
                if sample_size % 1000000 == 0:
                    print('{0} have processed'.format(sample_size))

        transformed_key1_file.close()
        print('{0}: key1 have processed'.format(p))

        sample_size = 0
        with open(path + p + feature2, 'r', encoding='utf-8') as f:
            for line in f:
                if 'key2' in line:
                    continue

                transformed_line = ''
                _line = line.split(sep)
                if int(_line[0]) in key_set2:
                    transformed_line += _line[0] + ';'

                    if not pd.isnull(_line[1]):
                        l = _line[1].replace(':1.0', '').split(',')
                        for j in range(len(l)):
                            if l[j] not in key2_feature_dict[0]:
                                key2_feature_dict[0][l[j]] = str(key2_feature_idx[0])
                                key2_feature_idx[0] += 1
                            if j != len(l ) -1:
                                transformed_line += key2_feature_dict[0][l[j]] + ','
                            else:
                                transformed_line += key2_feature_dict[0][l[j]] + ';'
                    else:
                        transformed_line += str(key2_feature_max) + ';'

                    for i in range(2, 9):
                        if _line[i] not in key2_feature_dict[ i -1]:
                            key2_feature_dict[ i -1][_line[i]] = str(key2_feature_idx[ i -1])
                            key2_feature_idx[ i -1] += 1
                        if i != 8:
                            transformed_line += key2_feature_dict[ i -1][_line[i]] + ';'
                        else:
                            transformed_line += key2_feature_dict[ i -1][_line[i]]

                    transformed_key2_file.write(transformed_line + '\n')

                sample_size += 1
                if sample_size % 1000000 == 0:
                    print('{0} have processed'.format(sample_size))

        transformed_key2_file.close()
        print('{0}: key2 have processed'.format(p))


def get_num_features(path, part, data, feature1, feature2, sep):
    for p in part:
        log_data = pd.read_csv(path + p + 'train_' + data, sep=sep,
                               names=['label', 't', 'f1', 'f2', 'f3', 'f4', 'key1', 'key2'], header=None)
        log_data2 = pd.read_csv(path + p + 'valid_' + data, sep=sep,
                                names=['label', 't', 'f1', 'f2', 'f3', 'f4', 'key1', 'key2'], header=None)
        print(log_data.head(5))
        print(log_data2.head(5))
        f_set = set(log_data['f1']) | set(log_data2['f1'])
        print('max f1', max(f_set))
        f_set = set(log_data['f2']) | set(log_data2['f2'])
        print('max f2', max(f_set))
        f_set = set(log_data['f3']) | set(log_data2['f3'])
        print('max f3', max(f_set))
        f_set = set(log_data['f4']) | set(log_data2['f4'])
        print('max f4', max(f_set))

        k1 = pd.read_csv(path + p + 'transformed_' + feature1, sep=';', names=['key1', 'seq1', 'seq2', 'seq3', 'seq4',
                                                                               'seq5', 'seq6', 'seq7', 'seq8', 'f1',
                                                                               'f2', 'f3', 'f4', 'f5'], header=None)
        f_set = set(k1['f1'])
        print('max key1 f1', max(f_set))
        f_set = set(k1['f2'])
        print('max key1 f2', max(f_set))
        f_set = set(k1['f3'])
        print('max key1 f3', max(f_set))
        f_set = set(k1['f4'])
        print('max key1 f4', max(f_set))
        f_set = set(k1['f5'])
        print('max key1 f5', max(f_set))

        k2 = pd.read_csv(path + p + 'transformed_' + feature2, sep=';', names=['key2', 'seq1', 'f1', 'f2', 'f3', 'f4',
                                                                               'f5', 'f6', 'f7'], header=None)
        f_set = set(k2['f1'])
        print('max key2 f1', max(f_set))
        f_set = set(k2['f2'])
        print('max key2 f2', max(f_set))
        f_set = set(k2['f3'])
        print('max key2 f3', max(f_set))
        f_set = set(k2['f4'])
        print('max key2 f4', max(f_set))
        f_set = set(k2['f5'])
        print('max key2 f5', max(f_set))
        f_set = set(k2['f6'])
        print('max key2 f6', max(f_set))
        f_set = set(k2['f7'])
        print('max key2 f7', max(f_set))
