# Rethinking Performance Estimation in Neural Architecture Search

There is the code of the paper ``Rethinking Performance Estimation in Neural Architecture Search`` for searching. We provide the implementations of Reinforcement Learning(RL), Evolution Algorithm(EA), Random Search(RS) and Differentiable Architecture Search(DARTS) coped with the proposed ``BPE`` method.

Two hyperparameter settings for searching, named **BPE1** and **BPE2** respectively, are defined in ``param_setting.py``. ``BPE1`` takes only **0.33 GPU hours** to train a full network while ``BPE2`` takes **0.5 GPU hours**.

## Reinforcement Learning

1. Train
```bash
git clone https://github.com/CVPR2020-ID1073/Rethinking-Performance-Estimation-in-Neural-Architecture-Search.git
cd Rethinking-Performance-Estimation-in-Neural-Architecture-Search

python run_rl.py --run_id=0 --output_path=experiment/RL --n_iters=100 --lr=1e-1 --param=BPE1/BPE2
```
The parameter ``--n_iters`` indicates the number of iterations, 100 for default setting and the ``--lr`` is the learning rate for agent optimization.

2. Parse the best architecture from Json file
```bash
python parse_json.py --method=RL --param=BPE1/BPE2 --run_id=0
```


## Evolution Algorithm

1. Train
```bash
python run_evolution.py --run_id=0 --output_path=experiment/EA --n_iters=100 --pop_size=50 --param=BPE1/BPE2
```
The parameter ``--n_iters`` indicates the total number of iterations, while the ``--pop_size`` is the number iterations to generate populations.

2. Sampling the best architecture from supernet

- sampling 10 nets from the best supernet
```bash
python parse_json.py --method=EA --param=BPE1/BPE2 --run_id=0
```
- augment the sampled nets as the same way with **Random Search**, and find the best cell architecture


## Random Search

1. Randomly generate 100 cell architectures
```bash
python random_darts_generator.py --num=100
```

2. Train these random architectures from scratch

- For BPE1: 
```bash
python augment.py --name=RS_BPE1 --file=random_darts_architecture.txt --data_path=data/ --save_path=experiment/ --batch_size=128 --lr=0.03 --layers=6 --init_channels=8 --epochs=10 --cutout_length=0 --image_size=16
```
- For BPE2: 
```bash
python augment.py --name=RS_BPE2 --file=random_darts_architecture.txt --data_path=data/ --save_path=experiment/ --batch_size=128 --lr=0.03 --layers=16 --init_channels=16 --epochs=30 --cutout_length=0 --image_size=16
```


## Differentiable Architecture Search

- For BPE1: 
```bash
python search.py --name=DARTS_BPE1 --batch_size=128 --w_lr=0.03 --layers=6 --init_channels=8 --epochs=10 --cutout_length=0 --image_size=16
```
- For BPE2: 
```bash
python search.py --name=DARTS_BPE2 --batch_size=128 --w_lr=0.03 --layers=16 --init_channels=16 --epochs=30 --cutout_length=0 --image_size=16
```
