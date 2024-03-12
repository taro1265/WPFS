#!/bin/bash
datasets=('lung' 'prostate' 'toxicity' 'cll' 'smk')

for dataset in "${datasets[@]}"
do
	echo "Running script for $dataset"
	python src/main.py \
	--model 'wpfs' \
	--max_steps 100 \
	--dataset "$dataset" \
	--use_best_hyperparams \
	--experiment_name "svd_$dataset" \
	--wpn_embedding_type 'svd' \
	--batch_size 16 \
	--run_repeats_and_cv

	python src/main.py \
	--model 'wpfs' \
	--max_steps 100 \
	--dataset "$dataset" \
	--use_best_hyperparams \
	--experiment_name "nmf_$dataset" \
	--wpn_embedding_type 'nmf' \
	--batch_size 16 \
	--run_repeats_and_cv

	python src/main.py \
	--model 'new_wpfs' \
	--max_steps 100 \
	--dataset "$dataset" \
	--use_best_hyperparams \
	--experiment_name "dnn_$dataset" \
	--wpn_embedding_type 'dnn' \
	--batch_size 16 \
	--run_repeats_and_cv
done