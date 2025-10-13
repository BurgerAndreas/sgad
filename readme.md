

```
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
git clone git@github.com:BurgerAndreas/hip.git
cd hip

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install --upgrade pip

uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-geometric

uv pip install --no-cache -r requirements.txt
```


T1x
```bash
git clone https://gitlab.com/matschreiner/Transition1x
cd Transition1x
uv pip install .
uv run download_t1x.py {path}
```
