import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import SugarscapeG1mt
# Assuming your Sugarscape model class is already imported
# from sugarscape_model import SugarscapeG1mt

# -------- Parameters --------
steps = 100
runs = 25

# Settings to test
p_copy_values = [0.4, 0.7, 0.95]
p_mutate_values = [0.005, 0.02, 0.05]

# Fixed parameters
p_mutate_fixed = 0.02
p_copy_fixed = 0.8


# -------- Plot 1: Vary p_copy --------
plt.figure(figsize=(10, 6))

for p_copy in p_copy_values:
    all_runs = []

    for _ in tqdm(range(runs), desc=f"Running p_copy={p_copy}"):
        model = SugarscapeG1mt(
            p_copy=p_copy,
            p_mutate=p_mutate_fixed,
            enable_staghunt=True,
            initial_population=200,
            width=50,
            height=50,
            seed=None  # Random seed each run
        )
        model.run_model(step_count=steps)

        data = model.datacollector.get_model_vars_dataframe()
        all_runs.append(data["Frac_stag"].values)

    all_runs = np.array(all_runs)
    avg = all_runs.mean(axis=0)
    std = all_runs.std(axis=0)

    plt.plot(range(steps+1), avg, label=f"p_copy={p_copy}", linewidth=2)
    plt.fill_between(range(steps+1), avg - std, avg + std, alpha=0.2)

plt.title("Effect of p_copy on Stag Coordination (100 Runs)", fontsize=14)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Fraction Choosing 'Stag'", fontsize=12)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Imitation Rate", fontsize=9)
plt.tight_layout()
plt.show()


# -------- Plot 2: Vary p_mutate --------
plt.figure(figsize=(10, 6))

for p_mutate in p_mutate_values:
    all_runs = []

    for _ in tqdm(range(runs), desc=f"Running p_mutate={p_mutate}"):
        model = SugarscapeG1mt(
            p_copy=p_copy_fixed,
            p_mutate=p_mutate,
            enable_staghunt=True,
            initial_population=200,
            width=50,
            height=50,
            seed=None
        )
        model.run_model(step_count=steps)

        data = model.datacollector.get_model_vars_dataframe()
        all_runs.append(data["Frac_stag"].values)

    all_runs = np.array(all_runs)
    avg = all_runs.mean(axis=0)
    std = all_runs.std(axis=0)

    plt.plot(range(steps+1), avg, label=f"p_mutate={p_mutate}", linewidth=2)
    plt.fill_between(range(steps+1), avg - std, avg + std, alpha=0.2)

plt.title("Effect of p_mutate on Stag Coordination (100 Runs)", fontsize=14)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Fraction Choosing 'Stag'", fontsize=12)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Mutation Rate", fontsize=9)
plt.tight_layout()
plt.show()
