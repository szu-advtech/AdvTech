for i in {1..5} ; do

  # 模型在20NG数据集上运行
  python run_scholar.py ./data/20ng/processed/ -k 50 --topk 1 --test-prefix test --device 0 --o ./outputs/scholar/20ng_50_1/${i} --epochs 500 --model trueContrastiveScholar
  python run_scholar.py ./data/20ng/processed/ -k 200 --topk 1 --test-prefix test --device 0 --o ./outputs/scholar/20ng_200_1/${i} --epochs 500 --model trueContrastiveScholar

  python run_scholar.py ./data/20ng/processed/ -k 50 --topk 5 --test-prefix test --device 0 --o ./outputs/scholar/20ng_50_5/${i} --epochs 500 --model trueContrastiveScholar
  python run_scholar.py ./data/20ng/processed/ -k 200 --topk 5 --test-prefix test --device 0 --o ./outputs/scholar/20ng_200_5/${i} --epochs 500 --model trueContrastiveScholar

  python run_scholar.py ./data/20ng/processed/ -k 50 --topk 15 --test-prefix test --device 0 --o ./outputs/scholar/20ng_50_15/${i} --epochs 500 --model trueContrastiveScholar
  python run_scholar.py ./data/20ng/processed/ -k 200 --topk 15 --test-prefix test --device 0 --o ./outputs/scholar/20ng_200_15/${i} --epochs 500 --model trueContrastiveScholar

  # 模型在IMDb数据集上运行
#  python run_scholar.py ./data/imdb/processed/ --topk 1 -k 50 --test-prefix test --labels sentiment --device 0 --o ./outputs/scholar/imdb_50_1/${i} c --epochs 500 --model trueContrastiveScholar
#  python run_scholar.py ./data/imdb/processed/ --topk 1 -k 200 --test-prefix test --labels sentiment --device 0 --o ./outputs/scholar/imdb_200_1/${i} c --epochs 500 --model trueContrastiveScholar
#
#  python run_scholar.py ./data/imdb/processed/ --topk 5 -k 50 --test-prefix test --labels sentiment --device 0 --o ./outputs/scholar/imdb_50_5/${i} c --epochs 500 --model trueContrastiveScholar
#  python run_scholar.py ./data/imdb/processed/ --topk 5 -k 200 --test-prefix test --labels sentiment --device 0 --o ./outputs/scholar/imdb_200_5/${i} c --epochs 500 --model trueContrastiveScholar
#
#  python run_scholar.py ./data/imdb/processed/ --topk 15 -k 50 --test-prefix test --labels sentiment --device 0 --o ./outputs/scholar/imdb_50_15/${i} c --epochs 500 --model trueContrastiveScholar
#  python run_scholar.py ./data/imdb/processed/ --topk 15 -k 200 --test-prefix test --labels sentiment --device 0 --o ./outputs/scholar/imdb_200_15/${i} c --epochs 500 --model trueContrastiveScholar
#
  # 模型在Wiki数据集上运行
#  python run_scholar.py ./data/wiki/processed/ --topk 1 -k 50 --test-prefix test --device 0 --o ./outputs/scholar/wiki_50_1/${i} --epochs 500 --model trueContrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01
#  python run_scholar.py ./data/wiki/processed/ --topk 1 -k 200 --test-prefix test --device 0 --o ./outputs/scholar/wiki_200_1/${i} --epochs 500 --model trueContrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01
#
#  python run_scholar.py ./data/wiki/processed/ --topk 5 -k 50 --test-prefix test --device 0 --o ./outputs/scholar/wiki_50_5/${i} --epochs 500 --model trueContrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01
#  python run_scholar.py ./data/wiki/processed/ --topk 5 -k 200 --test-prefix test --device 0 --o ./outputs/scholar/wiki_200_5/${i} --epochs 500 --model trueContrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01
#
#  python run_scholar.py ./data/wiki/processed/ --topk 15 -k 50 --test-prefix test --device 0 --o ./outputs/scholar/wiki_50_15/${i} --epochs 500 --model trueContrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01
#  python run_scholar.py ./data/wiki/processed/ --topk 15 -k 200 --test-prefix test --device 0 --o ./outputs/scholar/wiki_200_15/${i} --epochs 500 --model trueContrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01

done