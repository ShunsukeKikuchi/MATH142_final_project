# MATH142_final_project

Assuming hoffman2 environment.

## Environment
```bash:
# if conda
conda create math156
conda activate math156
conda install 'torch<2.7.0' numpy scipy matplotlib polars pandas
# if uv
uv init
uv venv
uv add 'torch<2.7.0' numpy scipy matplotlib polars pandas
. .venv/bin/activate
```

### Training
```bash:hoffman2
# Model1:
qsub submit.sh
# Model2:
qsub submit2.sh
# Model3:
qsub submit3.sh
```

```bash: usual env
# after activating environment
nohup python train.py >log/train.log &
nohup python train2.py >log/train2.log &
nohup python train3.py >log/train3.log &
```

### inference + submit to Kaggle kernel
use https://www.kaggle.com/code/shunsukekikuchi/1d-inference
