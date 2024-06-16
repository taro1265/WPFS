#!/bin/bash
datasets=('lung' 'prostate' 'smk' 'toxicity' 'cll')

for dataset in "${datasets[@]}"
do
	echo "Running script for $dataset"
	#python src/main.py \
	#--model 'wpfs' \
	#--max_steps 100 \
	#--dataset "$dataset" \
	#--use_best_hyperparams \
	#--experiment_name "wpfs_svd_$dataset" \
	#--wpn_embedding_type 'svd' \
	#--batch_size 16 \
	#--run_repeats_and_cv

	#python src/main.py \
	#--model 'wpfs' \
	#--max_steps 100 \
	#--dataset "$dataset" \
	#--use_best_hyperparams \
	#--experiment_name "wpfs_nmf_$dataset" \
	#--wpn_embedding_type 'nmf' \
	#--batch_size 16 \
	#--run_repeats_and_cv

	#python src/main.py \
	#--model 'experiment1' \
	#--max_steps 100 \
	#--dataset "$dataset" \
	#--use_best_hyperparams \
	#--experiment_name "ex1_raw_$dataset" \
	#--wpn_embedding_type 'raw' \
	#--batch_size 16 \
	#--run_repeats_and_cv

	python src/main.py \
	--model 'experiment2' \
	--max_steps 100 \
	--dataset "$dataset" \
	--use_best_hyperparams \
	--experiment_name "ex2_svd_$dataset" \
	--wpn_embedding_type 'svd' \
	--batch_size 16 \
	--run_repeats_and_cv

	python src/main.py \
	--model 'experiment2' \
	--max_steps 100 \
	--dataset "$dataset" \
	--use_best_hyperparams \
	--experiment_name "ex2_nmf_$dataset" \
	--wpn_embedding_type 'nmf' \
	--batch_size 16 \
	--run_repeats_and_cv

	python src/main.py \
	--model 'experiment3' \
	--max_steps 100 \
	--dataset "$dataset" \
	--use_best_hyperparams \
	--experiment_name "ex3_raw_$dataset" \
	--wpn_embedding_type 'raw' \
	--batch_size 16 \
	--run_repeats_and_cv
done