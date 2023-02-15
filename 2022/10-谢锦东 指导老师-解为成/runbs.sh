
# # test 1: run correctly, ok
# python main_nerfbs.py ./data/0008 --workspace ./result/bs8 -O --finetune_lips \
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000

# # test 2: GUI work, ok
# python main_nerfbs.py ./data/0008 --workspace ./result/bs8 -O --gui --finetune_lips \
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000

# # test 3: test mode 
# python main_nerfbs.py ./data/0008 --workspace ./result/bs8 -O --test --finetune_lips \
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000

# python main_nerfbs.py ./data/obama --workspace ./result/bsObama -O --finetune_lips --test\
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000
# python main_nerfbs.py ./data/0003 --workspace ./result/bs3 -O --finetune_lips --test \
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000
# python main_nerfbs.py ./data/0004 --workspace ./result/bs4 -O --finetune_lips --test\
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000
# python main_nerfbs.py ./data/0005 --workspace ./result/bs5 -O --finetune_lips --test\
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000
# python main_nerfbs.py ./data/0006 --workspace ./result/bs6 -O --finetune_lips --test\
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000
# python main_nerfbs.py ./data/0007 --workspace ./result/bs7 -O --finetune_lips --test\
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 60000
# python main_nerfbs.py ./data/0008 --workspace ./result/bs88 -O --finetune_lips --test\
#     --bound 1 --scale 4 --dt_gamma 0  --preload --iter 120000
