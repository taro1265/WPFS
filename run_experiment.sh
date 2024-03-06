python src/main.py \
	--model 'new_wpfs' \
	--max_steps 100 \
	--dataset 'lung' \
	--use_best_hyperparams \
	--experiment_name 'test2' \
	--wpn_embedding_type 'dnn' \
	--batch_size 16
	# --run_repeats_and_cv \  # if you want to runs 25 runs (5-fold cross-validation with 5 repeats) 