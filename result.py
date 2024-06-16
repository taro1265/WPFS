import os
import pandas as pd

datasets = ('lung', 'prostate', 'toxicity', 'cll', 'smk')
experiment_types = ('wpfs_svd', 'wpfs_nmf', 'ex1_raw', 'ex2_svd', 'ex2_nmf', 'ex3_raw')
experiment_results = []

for dataset in datasets:
    for experiment_type in experiment_types:
        experiment_name = f"{experiment_type}_{dataset}"
        folder_path = os.path.join('logs', experiment_name)

        for folder in os.listdir(os.path.join('logs', experiment_name)):
            metrics = pd.read_csv(os.path.join('logs', experiment_name, folder, 'metrics.csv'))

            train_balanced_acc = metrics['bestmodel_train/balanced_accuracy'].max()
            valid_balanced_acc = metrics['bestmodel_valid/balanced_accuracy'].max()
            test_balanced_acc = metrics['bestmodel_test/balanced_accuracy'].max()

            experiment_results.append({
                'Dataset': dataset,
                'Experiment': experiment_type,
                'Train Balanced Accuracy': train_balanced_acc,
                'Validation Balanced Accuracy': valid_balanced_acc,
                'Test Balanced Accuracy': test_balanced_acc
            })
results_df = pd.DataFrame(experiment_results)
grouped_results = results_df.groupby(['Dataset', 'Experiment']).mean().reset_index()

# Style the DataFrame for better visualization
styled_results = grouped_results.style.background_gradient(cmap='viridis', subset=['Train Balanced Accuracy', 'Validation Balanced Accuracy', 'Test Balanced Accuracy'])
styled_results = styled_results.set_caption('Experiment Results Summary')
styled_results = styled_results.set_table_styles([{
    'selector': 'caption',
    'props': [('text-align', 'center'),
              ('font-size', '16px'),
              ('font-weight', 'bold')]
}])
styled_results = styled_results.highlight_max(subset=['Train Balanced Accuracy', 'Validation Balanced Accuracy', 'Test Balanced Accuracy'], color='lightgreen')

# Display the styled DataFrame
styled_results