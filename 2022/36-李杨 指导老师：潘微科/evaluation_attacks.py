import pickle
import argparse
import numpy as np
from local_path import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def cv_testing(model, X, y, cv=10, repeat=1):
    cv_score = []
    for _ in range(repeat):
        cv_score.append(cross_val_score(model, X, y, cv=cv, n_jobs=8))
    return np.reshape(cv_score, [-1, ])


if __name__ == '__main__':

    params_parser = argparse.ArgumentParser()
    params_parser.add_argument('--dataset', '-d', type=str, default='movie_lens')
    params_parser.add_argument('--cv', '-c', type=int, default=5)
    params_parser.add_argument('--repeat', '-r', type=int, default=10)
    args = params_parser.parse_args()

    result_file_name = args.dataset + '_attack_result.csv'
    result_file_name = os.path.join('results', result_file_name)
    if os.path.isfile(result_file_name) is False:
        with open(result_file_name, 'w') as f:
            f.write('Obfuscation, attr, model, attack-mean, attack-detail\n')

    # (1) Load the data
    with open(os.path.join(data_dir, args.dataset + '.pkl'), 'rb') as f:
        data = pickle.load(f)

    models = [
        ['NB', MultinomialNB()],
        ['SVM', SVC()],
        ['GBDT', GradientBoostingClassifier(n_estimators=300, max_depth=5)]
    ]

    obfuscate_keys = ['user_visit'] + [e for e in list(data.keys()) if 'obfuscate' in e]
    # obfuscate_keys = [e for e in list(data.keys()) if 'e0.01' in e]
    with open(result_file_name, 'r') as f:
        history = f.readlines()
        history = [','.join(e.strip('\n').split(',')[:3]).replace(' ', '') for e in history]

    for obfuscate_key in obfuscate_keys:
        target_attr = []
        for attr in data['attributes']:
            if attr in obfuscate_key:
                target_attr = [attr]
                break
        if len(target_attr) == 0:
            target_attr = data['attributes']
        for attr in target_attr:
            for model_name, model in models:
                if ','.join([obfuscate_key, attr, model_name]) in history:
                    continue
                print('Running', obfuscate_key, attr, model_name)
                cv_score = cv_testing(
                    model=model, X=data[obfuscate_key], y=np.argmax(data['attributes'][attr], axis=1),
                    cv=args.cv, repeat=args.repeat
                )
                print('Finished', np.mean(cv_score))
                with open(result_file_name, 'a+') as f:
                    f.write(', '.join([
                        obfuscate_key, attr, model_name, '%.5f' % np.mean(cv_score),
                        ', '.join(['%.5f' % e for e in cv_score])
                    ]) + '\n')
