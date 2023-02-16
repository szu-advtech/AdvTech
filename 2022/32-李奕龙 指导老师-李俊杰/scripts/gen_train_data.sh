python3 -m tydi_canine.prepare_tydi_data \
  --input_jsonl="**/tydiqa-v1.0-train.jsonl.gz" \
  --output_dir=data/tydi/train.h5df \
  --max_seq_length=2048 \
  --doc_stride=512 \
  --max_question_length=256 \
  --include_unknowns=0.1 \
  --is_training=true