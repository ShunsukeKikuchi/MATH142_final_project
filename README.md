# MATH156_final_project

Assuming hoffman2 environment.

## Environment
```bash:
mkdir data log weights
# if conda
conda create math156
conda activate math156
conda install 'torch<2.7.0' numpy scipy matplotlib polars pandas
pip install kaggle
# if uv
uv init
uv venv
uv add 'torch<2.7.0' numpy scipy matplotlib polars pandas kaggle
. .venv/bin/activate
```

## data install
```bash
cd data
kaggle competitions download -c hms-harmful-brain-activity-classification
unzip hms-harmful-brain-activity-classification.zip
cd ..
```

## Training
```bash:hoffman2
# Model1:
qsub submit.sh
# Model2:
qsub submit2.sh
# Model3:
qsub submit3.sh
```
or, 
```bash: usual env
# after activating environment
nohup python train.py >log/train.log &
nohup python train2.py >log/train2.log &
nohup python train3.py >log/train3.log &
```

### inference + submit to Kaggle kernel
use https://www.kaggle.com/code/shunsukekikuchi/1d-inference
