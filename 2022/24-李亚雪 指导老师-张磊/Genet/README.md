# GENET: Automatic Curriculum Generation for Learning Adaptation in Networking
### Install python dependency
```bash
cd Genet
bash install_python_dependency.sh
```

## Unseen synthetic environments 

### ABR
```bash
cd Genet/fig_reproduce/fig9
bash run.sh
```

### CC
```bash
cd Genet # cd into the project root
python src/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir data/cc/synthetic_dataset \
  --fast
python src/plot_scripts/plot_syn_dataset.py
```

```bash
cd Genet # cd into the project root
python src/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir data/cc/synthetic_dataset
python src/plot_scripts/plot_syn_dataset.py
```

### LB
```bash
cd Genet/genet-lb-fig-upload

# example output: [-4.80, 0.07]
python rl_test.py --saved_model="results/testing_model/udr_1/model_ep_49600.ckpt"

# example output: [-3.87, 0.08]
python rl_test.py --saved_model="results/testing_model/udr_2/model_ep_44000.ckpt"

# example output: [-3.57, 0.07]
python rl_test.py --saved_model="results/testing_model/udr_3/model_ep_25600.ckpt"

# example output: [-3.02, 0.04]
python rl_test.py --saved_model="results/testing_model/adr/model_ep_20200.ckpt"

python analysis/fig9_lb.py
```

## Generalizability

### ABR
```bash
cd Genet/fig_reproduce/fig13
bash run.sh
```

### CC
```bash
cd Genet # cd into the project root
python src/plot_scripts/plot_bars_ethernet.py
python src/plot_scripts/plot_bars_cellular.py
```
## Learning curves
### CC
```bash
source genet/bin/activate
cd Genet

bash src/drivers/cc/run_for_learning_curve.sh
python src/plot_scripts/plot_learning_curve.py
```
