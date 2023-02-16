mkdir -p data/paddle_weight || echo "dir exist"
python -m reproduction_utils.weight_convert_files.convert_weight \
  --pytorch_checkpoint_path=data/torch_weight/pytorch_model.bin \
  --paddle_dump_path=data/paddle_weight/model_state.pdparams \
  --layer_mapping_file=reproduction_utils/weight_convert_files/torch_paddle_layer_map.json