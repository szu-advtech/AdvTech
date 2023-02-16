#!/bin/sh
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_grasptta' -c 'cfg.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 2 -e '003_crackerbox_grasptta' -c 'cfg.yaml'  -w '2021-11-13-22-31-11/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-46-18/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 4 -e '005_tomatocan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-11-46-23/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 5 -e '006_mustard_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-39/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 6 -e '007_tunacan_grasptta' -c 'cfg.yaml'       -w '2021-11-11-11-46-41/full_2400.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 7 -e '008_puddingbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 8 -e '009_gelatinbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-49/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 9 -e '010_pottedcan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-20-54-36/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 10 -e '011_banana_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 11 -e '019_pitcher_grasptta' -c 'cfg.yaml'      -w '2021-11-12-00-34-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 12 -e '021_bleach_grasptta' -c 'cfg.yaml'       -w '2021-11-12-02-33-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 13 -e '024_bowl_grasptta' -c 'cfg.yaml'         -w '2021-11-11-21-51-03/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 14 -e '025_mug_grasptta' -c 'cfg.yaml'          -w '2021-11-11-21-34-34/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg.yaml'   -w '2021-11-12-03-27-15/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 16 -e '036_woodblock_grasptta' -c 'cfg.yaml'    -w '2021-11-12-06-29-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 17 -e '037_scissors_grasptta' -c 'cfg.yaml'     -w '2021-11-11-21-34-32/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 18 -e '040_marker_grasptta' -c 'cfg.yaml'       -w '2021-11-11-21-53-48/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_grasptta' -c 'cfg.yaml'        -w '2021-11-11-23-41-21/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 21 -e '061_foam_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-47-46/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t

#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_grasptta' -c 'cfg.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 2 -e '003_crackerbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-11-52-45/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-46-18/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 4 -e '005_tomatocan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-11-46-23/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 5 -e '006_mustard_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-39/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 6 -e '007_tunacan_grasptta' -c 'cfg.yaml'       -w '2021-11-11-11-46-41/full_2400.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 7 -e '008_puddingbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 8 -e '009_gelatinbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-49/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 9 -e '010_pottedcan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-20-54-36/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 10 -e '011_banana_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 11 -e '019_pitcher_grasptta' -c 'cfg.yaml'      -w '2021-11-12-00-34-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 12 -e '021_bleach_grasptta' -c 'cfg.yaml'       -w '2021-11-12-02-33-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 13 -e '024_bowl_grasptta' -c 'cfg.yaml'         -w '2021-11-11-21-51-03/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 14 -e '025_mug_grasptta' -c 'cfg.yaml'          -w '2021-11-11-21-34-34/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg.yaml'   -w '2021-11-12-03-27-15/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 16 -e '036_woodblock_grasptta' -c 'cfg.yaml'    -w '2021-11-12-06-29-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 17 -e '037_scissors_grasptta' -c 'cfg.yaml'     -w '2021-11-11-21-34-32/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 18 -e '040_marker_grasptta' -c 'cfg.yaml'       -w '2021-11-11-21-53-48/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_grasptta' -c 'cfg.yaml'        -w '2021-11-11-23-41-21/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 21 -e '061_foam_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-47-46/full_3000.pt' -p -sd 'data_grasptta'  -gtta