## Dependencies

```
- Ubuntu 20.04
- Python 3
- PyTorch 1.8
```

The file of the conda environment is environment.yml. We use [V-REP 3.5.0] as the simulation environment.

## Code

Experiments  requires at least 8GB of GPU memory to run the code.

First run V-REP and open the file ``simulation/simulation.ttt`` to start the simulation. Then download the pre-trained models by running

```
sh downloads.sh
```

### Training

To train from scratch, run

```
python main.py
```

You can also resume training from checkpoint and collected data

```
python main.py
--load_ckpt --critic_ckpt CRITIC-MODEL-PATH --coordinator_ckpt COORDINATOR-MODEL-PATH
--continue_logging --logging_directory SESSION-DIRECTORY --clipseg
```

```
python main.py --load_ckpt --critic_ckpt ./logs/2022-11-29.16:09:51/critic_models/critic-005000.pth --coordinator_ckpt /home/pi/Desktop/course/adcanced_stage2/grasping-invisible/logs/2022-11-29.16:09:51/coordinator_models/coordinator-005000.pth --continue_logging --logging_directory /home/pi/Desktop/course/adcanced_stage2/grasping-invisible/logs/2022-11-29.16:09:51/ --clipseg
```

### Testing

```
python main.py
--is_testing --test_preset_cases --test_target_seeking
--load_ckpt --critic_ckpt CRITIC-MODEL-PATH --coordinator_ckpt COORDINATOR-MODEL-PATH
--config_file TEST-CASE-PATH
```

using lwrf seg module

```
python main.py --is_testing --test_preset_cases --test_target_seeking --load_ckpt --critic_ckpt ./logs/lwrf_ckpt/critic_models/critic-003500.pth --coordinator_ckpt ./logs/lwrf_ckpt/coordinator_models/coordinator-003500.pth --config_file ./simulation/preset/exploration-06_no_red.txt
```

using clip seg module

```
python main.py --is_testing --test_preset_cases --test_target_seeking --load_ckpt --critic_ckpt ./logs/clipseg_ckpt/critic_models/critic-003500.pth --coordinator_ckpt ./logs/clipseg_ckpt/coordinator_models/coordinator-003500.pth --config_file ./simulation/preset/exploration-06.txt --clipseg
```

The files of the test cases are available in ``simulation/preset``.

## Acknowledgments

We use the following code in our project

- Visual Pushing and Grasping Toolbox
- CLIPSeg
- A deep learning approach to grasping invisible
- A graph-based reinforcement learning method to grasp full occulded objects
