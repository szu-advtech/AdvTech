#CUDA_VISIBLE_DEVICES='2' python run_kbert_cls.py --pretrained_model_path=./models/google_model.bin --config_path=./models/google_config.json --vocab_path=./models/google_vocab.txt --train_path=./datasets/book_review/train.tsv --dev_path=./datasets/book_review/dev.tsv --test_path=./datasets/book_review/test.tsv --epochs_num=5 --batch_size=32 --kg_name=CnDbpedia
CUDA_VISIBLE_DEVICES='2' python run_kbert_cls.py --pretrained_model_path=./models/google_model.bin --config_path=./models/google_config.json --vocab_path=./models/google_vocab.txt --train_path=./datasets/shopping/train.tsv --dev_path=./datasets/shopping/dev.tsv --test_path=./datasets/shopping/test.tsv --epochs_num=5 --batch_size=32 --kg_name=CnDbpedia


# CUDA_VISIBLE_DEVICES='2' python run_kbert_ner.py --pretrained_model_path=./models/google_model.bin --config_path=./models/google_config.json --vocab_path=./models/google_vocab.txt --train_path=./datasets/msra_ner/train.tsv --dev_path=./datasets/msra_ner/dev.tsv --test_path=./datasets/msra_ner/test.tsv --epochs_num=5 --batch_size=16 --kg_name=CnDbpedia


#sh run.sh