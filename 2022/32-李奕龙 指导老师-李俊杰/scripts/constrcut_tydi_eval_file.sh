python3 run_tydi.py \
  --predict_file=data/tydi/tydiqa-v1.0-dev.jsonl.gz \
  --precomputed_predict_file=data/tydi/dev.h5df \
  --do_eval_file_construct \
  --max_seq_length=2048 \
  --max_answer_length=100 \
  --candidate_beam=30 \
  --output_dir=data/tydiqa_baseline_model/predict \
  --output_prediction_file=data/tydiqa_baseline_model/predict/pred.jsonl