# Colab launcher

Run each block as one Colab cell. Code lives in the repo; Drive holds data and results.

## 1. Drive, repo, packages
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
%%bash
D=/content/drive/MyDrive/auto-curriculum
mkdir -p $D/data $D/results $D/logs
cd /content
[ -d autofocus-curriculum ] || git clone https://github.com/omarreis01/autofocus-curriculum.git
cd autofocus-curriculum && git pull && pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv
```

## 2. Data (once; skips files that are already there)
```bash
%%bash
cd /content/autofocus-curriculum
python -m src.data --data-dir /content/drive/MyDrive/auto-curriculum/data
```
Each new runtime: copy data to local disk (Drive reads are slow).
```bash
%%bash
mkdir -p /content/data && cp -rn /content/drive/MyDrive/auto-curriculum/data/. /content/data/
```

## 3. Smoke test
```bash
%%bash
cd /content/autofocus-curriculum
python -m src.smoke --data-dir /content/data
```

## 4. Training run in the background
Results go to `results/<condition>/<seed>/` on Drive. `--resume` continues from `last.pt` after a dropped session.
```bash
%%bash
cd /content/autofocus-curriculum
D=/content/drive/MyDrive/auto-curriculum
nohup python -m src.train --config configs/sharp_s0.json --data-dir /content/data \
  --out-dir $D/results/sharp/0 --resume > $D/logs/sharp_0.log 2>&1 &
```
Watch: `!tail -n 5 /content/drive/MyDrive/auto-curriculum/logs/sharp_0.log`

## 5. Eval
```bash
%%bash
cd /content/autofocus-curriculum
D=/content/drive/MyDrive/auto-curriculum
python -m src.eval --ckpt $D/results/sharp/0/last.pt --data-dir /content/data --out $D/results/sharp/0/eval.json
```

## Local tests (CPU)
```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python -m src.smoke --fake-data
```
