#!/bin/bash

function submit_array {
    name=$1
    array=${2:-0}
    cmd=$3
    command="sbatch --job-name=$name \
        --array=$array \
        --output=slurm/$name-%A_%a.out \
        --error=slurm/$name-%A_%a.err \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --signal=USR1@600 \
        --open-mode=append \
        --time=10:00:00 \
        --gres=gpu:1 \
        --wrap=\"$cmd\""
    echo $command
    eval $command
}

CKPT_PATH=""
RESULTS_DIR=""

# The parameter `num_ranks` lets the program identify which slice of the data to operate on.

submit_array "spice_evaluation" "0-79" "python eval.py \
--test_mols data/spice_test.txt \
--true_confs data/spice_test_conformers \
--save_path ${RESULTS_DIR}/spice \
--checkpoint_path ${CKPT_PATH} \
--batch_size 128 \
--max_n_refs 512 \
--relax \
--num_ranks 80"

submit_array "drugs_evaluation" "0-79" "python eval.py \
--test_mols data/drugs_test.txt \
--true_confs data/drugs_test_conformers \
--save_path ${RESULTS_DIR}/drugs \
--checkpoint_path ${CKPT_PATH} \
--batch_size 128 \
--max_n_refs 512 \
--relax \
--num_ranks 80"
