#! /bin/bash

cd /ssd/Code/sgad
source .venv/bin/activate;

uv run playground/train_lj_gad.py --initial-lr 1e-4 --final-lr 1e-6;
uv run playground/train_lj_gad.py --score-clip 1e3;
# uv run playground/train_lj_gad.py --loss-type l1;
uv run playground/train_lj_gad.py --score-clip 1e3 --loss-type l1;
