for i in {1..5} ; do

  # 计算在20NG数据集上所得结果的npmi
  python compute_npmi.py ./outputs/scholar/20ng_50_1/${i}/topics.txt .data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json
  python compute_npmi.py ./outputs/scholar/20ng_200_1/${i}/topics.txt .data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json
  python compute_npmi.py ./outputs/scholar/20ng_50_5/${i}/topics.txt .data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json
  python compute_npmi.py ./outputs/scholar/20ng_200_5/${i}/topics.txt .data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json
  python compute_npmi.py ./outputs/scholar/20ng_50_15/${i}/topics.txt .data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json
  python compute_npmi.py ./outputs/scholar/20ng_200_15/${i}/topics.txt .data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json

  # 计算在IMDb数据集上所得结果的npmi
#  python compute_npmi.py ./outputs/scholar/imdb_50_1/${i}/topics.txt ./data/imdb/processed/train.npz ./data/imdb/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/imdb_200_1/${i}/topics.txt ./data/imdb/processed/train.npz ./data/imdb/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/imdb_50_5/${i}/topics.txt ./data/imdb/processed/train.npz ./data/imdb/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/imdb_200_5/${i}/topics.txt ./data/imdb/processed/train.npz ./data/imdb/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/imdb_50_15/${i}/topics.txt ./data/imdb/processed/train.npz ./data/imdb/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/imdb_200_15/${i}/topics.txt ./data/imdb/processed/train.npz ./data/imdb/processed/train.vocab.json


  # 计算在Wiki数据集上所得结果的npmi
#  python compute_npmi.py ./outputs/scholar/wiki_50_1/${i}/topics.txt ./data/wiki/processed/train.npz ./data/wiki/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/wiki_200_1/${i}/topics.txt ./data/wiki/processed/train.npz ./data/wiki/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/wiki_50_5/${i}/topics.txt ./data/wiki/processed/train.npz ./data/wiki/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/wiki_200_5/${i}/topics.txt ./data/wiki/processed/train.npz ./data/wiki/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/wiki_50_15/${i}/topics.txt ./data/wiki/processed/train.npz ./data/wiki/processed/train.vocab.json
#  python compute_npmi.py ./outputs/scholar/wiki_200_15/${i}/topics.txt ./data/wiki/processed/train.npz ./data/wiki/processed/train.vocab.json

done