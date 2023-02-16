#### 数据预处理
运行下列命令，或直接去对应的py文件修改参数
```
create_train.py [-h] [--qrels_file QRELS_FILE] [--query_file QUERY_FILE] [--collection_file COLLECTION_FILE] [--save_to SAVE_TO] [--tokenizer_name TOKENIZER_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --qrels_file QRELS_FILE
                        qrels file
  --query_file QUERY_FILE
                        query file
  --collection_file COLLECTION_FILE
                        collections file
  --save_to SAVE_TO     processd train json file
  --tokenizer_name TOKENIZER_NAME
                        pretrained model tokenizer
```

产生的train.json格式如下
```
{"spans": [["QUERY_TOKENIZED"], ["PASSAGE_TOKENIZED"]]}
{"spans": [["QUERY_TOKENIZED"], ["PASSAGE_TOKENIZED"]]}
```

#### 开始训练DPR
运行下列命令
```
sh run_train.sh
```

#### 语料库编码
我们需要利用训练好的模型来事先处理语料库，以节省检索时间
```
usage: encoder_corpus.py [-h] [--model_path MODEL_PATH] [--max_sequence_length MAX_SEQUENCE_LENGTH] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE] [--pool_type POOL_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        sbert or prop
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        sbert or prop
  --input_file INPUT_FILE
                        input file with raw text
  --output_file OUTPUT_FILE
                        output file save embeddings
  --pool_type POOL_TYPE
                        pool type of the final text repsesentation
```

#### 检索
这一步是实际的文档检索过程
```
usage: retrieval.py [-h] [--model_path MODEL_PATH] [--pool_type POOL_TYPE] [--max_sequence_length MAX_SEQUENCE_LENGTH] [--index_file INDEX_FILE] [--topk TOPK] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --pool_type POOL_TYPE
                        pool type of the final text repsesentation
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        use pb model or tf checkpoint
  --index_file INDEX_FILE
  --topk TOPK
  --input_file INPUT_FILE
  --output_file OUTPUT_FILE
```

#### 评估检索结果
```
usage: evaluate.py [-h] [--result_path RESULT_PATH] [--qrel_path QREL_PATH] [--reverse REVERSE] [--topk TOPK] [--topk_list TOPK_LIST]

optional arguments:
  -h, --help            show this help message and exit
  --result_path RESULT_PATH
                        search result
  --qrel_path QREL_PATH
  --reverse REVERSE     reverse score during sorting
  --topk TOPK
  --topk_list TOPK_LIST
```
