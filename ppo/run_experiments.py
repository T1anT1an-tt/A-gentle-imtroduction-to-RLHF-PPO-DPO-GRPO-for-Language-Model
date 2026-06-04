import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from train import run_training


def build_tasks(base_config, experiments, seeds, results_dir):
    tasks = []
    for exp_name, exp_data in experiments.items():
        param_name = exp_data['param']
        for val in exp_data['values']:
            for seed in seeds:
                config = base_config.copy()
                config.update(exp_data['override'])

                if exp_name == 'ablation':
                    config['use_adv_norm'] = False
                    config['use_entropy'] = False
                    config['use_grad_clip'] = False

                    if val == 'adv_norm':
                        config['use_adv_norm'] = True
                    elif val == 'entropy':
                        config['use_entropy'] = True
                    elif val == 'grad_clip':
                        config['use_grad_clip'] = True
                    elif val == 'all':
                        config['use_adv_norm'] = True
                        config['use_entropy'] = True
                        config['use_grad_clip'] = True
                else:
                    config[param_name] = val

                val_str = str(val).replace('.', '_')
                run_dir = os.path.join(results_dir, exp_name, f'{param_name}_{val_str}', f'seed_{seed}')
                tasks.append({
                    'experiment': exp_name,
                    'param_name': param_name,
                    'param_value': val,
                    'seed': seed,
                    'config': config,
                    'run_dir': run_dir,
                })
    return tasks


def run_single_experiment(task):
    print(
        f"Running {task['experiment']} | "
        f"{task['param_name']}={task['param_value']} | seed={task['seed']}"
    )
    summary = run_training(task['config'], task['seed'], task['run_dir'])
    summary['experiment'] = task['experiment']
    summary['param_name'] = task['param_name']
    summary['param_value'] = task['param_value']
    return summary


def main():
    base_config = {
        'learning_rate': 0.0005,
        'gamma': 0.98,
        'lmbda': 0.95,
        'eps_clip': 0.1,
        'K_epoch': 3,
        'T_horizon': 20,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'grad_clip_norm': 0.5,
        'use_adv_norm': True,
        'use_entropy': True,
        'use_grad_clip': True,
        'episodes': 800,
        'num_envs': 32,
        'mini_batch_size': 1024,
        'vector_env_mode': os.environ.get('PPO_VECTOR_ENV_MODE', 'sync'),
        'device': os.environ.get('PPO_DEVICE', 'cpu'),
        'torch_num_threads': 1,
    }

    experiments = {
        'lr_sensitivity': {
            'param': 'learning_rate',
            'values': [1e-4, 3e-4, 5e-4, 1e-3],
            'override': {}
        },
        'clip_sensitivity': {
            'param': 'eps_clip',
            'values': [0.05, 0.1, 0.2, 0.3],
            'override': {}
        },
        'horizon_sensitivity': {
            'param': 'T_horizon',
            'values': [20, 64, 128, 256],
            'override': {}
        },
        'ablation': {
            'param': 'tricks',
            'values': ['baseline', 'adv_norm', 'entropy', 'grad_clip', 'all'],
            'override': {}
        }
    }

    seeds = [0, 1, 2, 3, 4]
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    default_workers = max(1, min(4, (os.cpu_count() or 1) // 8))
    parallel_workers = int(os.environ.get('PPO_PARALLEL_WORKERS', default_workers))
    if str(base_config['device']).startswith('cuda') and parallel_workers > 1:
        parallel_workers = 1

    tasks = build_tasks(base_config, experiments, seeds, results_dir)
    all_summaries = []

    print(f'Total runs: {len(tasks)}')
    print(f'Parallel workers: {parallel_workers}')
    print(f'Device: {base_config["device"]}')
    print(f'Vector env mode: {base_config["vector_env_mode"]}')

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(run_single_experiment, task): task for task in tasks}
        for future in as_completed(futures):
            summary = future.result()
            all_summaries.append(summary)
            print(
                f"Completed {summary['experiment']} | "
                f"{summary['param_name']}={summary['param_value']} | seed={summary['seed']} | "
                f"final_100_mean={summary['final_100_mean']:.2f}"
            )

    df_all = pd.DataFrame(all_summaries)
    df_all.to_csv(os.path.join(results_dir, 'summary_results.csv'), index=False)
    print('All experiments completed! Results saved to:', results_dir)


if __name__ == '__main__':
    main()
