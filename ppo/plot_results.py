import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_experiment(exp_name, results_dir):
    exp_dir = os.path.join(results_dir, exp_name)
    if not os.path.exists(exp_dir):
        print(f"Skipping {exp_name}, directory not found.")
        return

    # Load all data for this experiment
    all_data = []
    for param_val_folder in os.listdir(exp_dir):
        param_val_path = os.path.join(exp_dir, param_val_folder)
        if not os.path.isdir(param_val_path):
            continue

        for seed_folder in os.listdir(param_val_path):
            seed_path = os.path.join(param_val_path, seed_folder)
            csv_path = os.path.join(seed_path, 'episode_logs.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Extract clean param value string
                if '_' in param_val_folder:
                    val_str = param_val_folder.split('_', 1)[1]
                else:
                    val_str = param_val_folder

                df['param_value'] = val_str
                df['seed'] = seed_folder
                all_data.append(df)

    if not all_data:
        print(f"No data found for {exp_name}")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Optional: Create a separate directory for plots to keep things clean
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating plots for {exp_name}...")

    # Set seaborn style
    sns.set_theme(style="darkgrid")

    # 1. Mean reward curve with std shading
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_all, x='episode', y='ma_50', hue='param_value', errorbar='sd')
    plt.title(f'{exp_name}: Moving Average Return (window=50)')
    plt.ylabel('Return')
    plt.savefig(os.path.join(plots_dir, 'mean_reward_curve.png'))
    plt.close()

    # 2. Final 100 episode return boxplot
    max_ep = df_all['episode'].max()
    final_100 = df_all[df_all['episode'] >= max_ep - 100]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_100, x='param_value', y='return')
    plt.title(f'{exp_name}: Final 100 Episode Return Distribution')
    plt.ylabel('Return')
    plt.savefig(os.path.join(plots_dir, 'final_100_return_boxplot.png'))
    plt.close()

    # 3. Policy entropy curve
    if 'entropy' in df_all.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='episode', y='entropy', hue='param_value', errorbar='sd')
        plt.title(f'{exp_name}: Policy Entropy')
        plt.ylabel('Entropy')
        plt.savefig(os.path.join(plots_dir, 'policy_entropy.png'))
        plt.close()

    # 4. Losses curves (actor, critic, total)
    for loss_type in ['actor_loss', 'critic_loss', 'total_loss']:
        if loss_type in df_all.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_all, x='episode', y=loss_type, hue='param_value', errorbar='sd')
            plt.title(f'{exp_name}: {loss_type}')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(plots_dir, f'{loss_type}.png'))
            plt.close()

    # 5. PPO Ratio mean/std curves
    if 'ratio_mean' in df_all.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='episode', y='ratio_mean', hue='param_value', errorbar='sd')
        plt.title(f'{exp_name}: PPO Ratio Mean')
        plt.ylabel('Ratio')
        plt.savefig(os.path.join(plots_dir, 'ratio_mean.png'))
        plt.close()

    if 'ratio_std' in df_all.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='episode', y='ratio_std', hue='param_value', errorbar='sd')
        plt.title(f'{exp_name}: PPO Ratio Std Dev')
        plt.ylabel('Ratio Std')
        plt.savefig(os.path.join(plots_dir, 'ratio_std.png'))
        plt.close()

    # 6. Clip fraction curve
    if 'clip_fraction' in df_all.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='episode', y='clip_fraction', hue='param_value', errorbar='sd')
        plt.title(f'{exp_name}: Clip Fraction')
        plt.ylabel('Fraction of ratios clipped')
        plt.savefig(os.path.join(plots_dir, 'clip_fraction.png'))
        plt.close()

    # 7. Advantage mean/std
    if 'adv_mean' in df_all.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='episode', y='adv_mean', hue='param_value', errorbar='sd')
        plt.title(f'{exp_name}: Advantage Mean')
        plt.ylabel('Advantage')
        plt.savefig(os.path.join(plots_dir, 'advantage_mean.png'))
        plt.close()

def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}. Please run experiments first.")
        return

    experiments = ['lr_sensitivity', 'clip_sensitivity', 'horizon_sensitivity', 'ablation']
    for exp in experiments:
        plot_experiment(exp, results_dir)

    print("\nAll plots generated successfully! Check the 'plots' folder inside each experiment directory.")

if __name__ == '__main__':
    main()
