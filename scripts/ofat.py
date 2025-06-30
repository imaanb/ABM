import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mesa.batchrunner import batch_run

from sugarscape.model import SugarscapeG1mt

cont_vars = [
    "initial_population",
    "metabolism_min",
    "metabolism_max",
    "endowment_min",
    "endowment_max",
    "vision_min",
    "vision_max",
]

adjusted_bounds = [
    [25, 250],  # initial population
    [0.1, 1.0],  # metabolism min
    [1.1, 2.0],  # metabolism max
    [0, 5],  # endowmnent min
    [8, 20],  # endownment max
    [1.0, 3.0],  # vision min
    [3.1, 6.0],  # vision max
]
problem = {
    "num_vars": len(cont_vars),
    "names": cont_vars,
    "bounds": adjusted_bounds,
}

integer_vars = {
    "initial_population",
    "endowment_min",
    "endowment_max",
    "vision_min",
    "vision_max",
    "metabolism_min",
    "metabolism_max",
}


discrete_factors = {
    "wealth_tax_system": [0],  # 0 ≙ "none"
    "income_tax_system": [0],  # 0 ≙ "none"
    "enable_staghunt": [0],  # OFF
}

wealth_tax_map = {0: "none", 1: "proportional", 2: "progressive", 3: "degressive"}
income_tax_map = wealth_tax_map.copy()

# Set up the problem dictionary with bounds and names
max_steps = 5  # 200
n = 2  # 5


def run_model(params, seed=None):
    """Run the SugarscapeG1mt model with given parameters and return Gini coefficient"""

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Process parameters like in the main analysis
    processed_params = params.copy()

    # 1 – round integers
    for name in integer_vars:
        if name in processed_params:
            processed_params[name] = int(round(processed_params[name]))

    # 2 – enforce ordering on min/max pairs
    for lo, hi, caster in [
        ("endowment_min", "endowment_max", int),
        ("vision_min", "vision_max", int),
        ("metabolism_min", "metabolism_max", float),
    ]:
        if lo in processed_params and hi in processed_params:
            lo_val, hi_val = sorted((processed_params[lo], processed_params[hi]))
            processed_params[lo], processed_params[hi] = caster(lo_val), caster(hi_val)

    # 3 – disable staghunt parameters
    processed_params["enable_staghunt"] = False
    processed_params["p_copy"] = 0.0
    processed_params["p_mutate"] = 0.0

    # 4 – map tax codes to strings (both are "none")
    processed_params["wealth_tax_system"] = "none"
    processed_params["income_tax_system"] = "none"

    # 5 – cast remaining floats
    processed_params["flat_rate"] = 0.0
    processed_params["income_tax_flat_rate"] = 0.0

    # 6 – run the model
    try:
        out = batch_run(
            SugarscapeG1mt,
            parameters=processed_params,
            iterations=1,
            max_steps=max_steps,
            data_collection_period=-1,
            display_progress=False,
        )[0]

        return out["Gini"]
    except Exception as e:
        print(f"Error running model with params {processed_params}: {e}")
        return np.nan


# OFAT Analysis

print("=== ONE FACTOR AT A TIME (OFAT) ANALYSIS ===")

# Baseline values = midpoint of each bound
baseline = {
    name: np.mean(bound) for name, bound in zip(problem["names"], problem["bounds"])
}

# Include discrete factors as fixed
baseline.update({k: v[0] for k, v in discrete_factors.items()})

print("Baseline parameter values:")
for param, value in baseline.items():
    if param in problem["names"]:
        print(f"  {param}: {value:.3f}")

print("\nEach parameter will be varied across 10 steps while others remain at baseline")
print(f"Replications per parameter value: {n}")
print(f"Model runs per parameter: 10 x {n} = {10 * n}")
print(f"Total model runs: {len(problem['names']) * 10 * n}")

# Prepare for results storage
results = {
    var: {"x": [], "y_mean": [], "y_std": [], "y_all": []} for var in problem["names"]
}


# Loop over each continuous variable
print("Running OFAT Analysis...")
for i, var in enumerate(problem["names"]):
    print(f"Analyzing variable {i + 1}/{len(problem['names'])}: {var}")
    var_min, var_max = problem["bounds"][i]
    # Choose N steps between min and max
    N = 10
    x_values = np.linspace(var_min, var_max, N)

    y_values = []

    for j, x in enumerate(x_values):
        # Set baseline values
        params = baseline.copy()
        # Vary one variable
        params[var] = round(x) if var in integer_vars else x

        # Run model n times with different seeds and collect all results
        y_replicates = []
        for rep in range(n):
            seed = 1000 * i + 100 * j + rep  # Unique seed for each run
            output = run_model(params, seed=seed)
            y_replicates.append(output)

        # Calculate statistics
        y_replicates = np.array(y_replicates)
        mean_gini = np.nanmean(y_replicates)
        std_gini = np.nanstd(y_replicates)

        # Store results
        results[var]["y_mean"].append(mean_gini)
        results[var]["y_std"].append(std_gini)
        results[var]["y_all"].append(y_replicates)

        # Progress indicator
        if (j + 1) % 5 == 0 or j == len(x_values) - 1:
            print(
                f"  Step {j + 1}/{N} completed (mean Gini: {mean_gini:.3f} ± {std_gini:.3f})"
            )

    # Store x values
    results[var]["x"] = x_values

print("OFAT Analysis completed!")

# Plotting results with error bars

fig, axs = plt.subplots(
    len(problem["names"]), 1, figsize=(10, 4 * len(problem["names"]))
)

if len(problem["names"]) == 1:
    axs = [axs]  # Ensure axs is iterable

for ax, var in zip(axs, problem["names"]):
    x_vals = results[var]["x"]
    y_means = results[var]["y_mean"]
    y_stds = results[var]["y_std"]

    # Plot mean with error bars (standard deviation)
    ax.errorbar(
        x_vals,
        y_means,
        yerr=y_stds,
        marker="o",
        linewidth=2,
        markersize=6,
        capsize=4,
        capthick=1,
        elinewidth=1,
        alpha=0.8,
    )

    # Also plot individual points as semi-transparent dots
    for i, (x_val, y_reps) in enumerate(zip(x_vals, results[var]["y_all"])):
        ax.scatter([x_val] * len(y_reps), y_reps, alpha=0.3, s=10, color="gray")

    ax.set_xlabel(var, fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title(
        f"OFAT: {var} vs Gini Coefficient (n={n} replications)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add some styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text box with statistics
    y_range = np.max(y_means) - np.min(y_means)
    max_std = np.max(y_stds)
    info_text = f"Range: {y_range:.3f}\nMax StdDev: {max_std:.3f}"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        fontsize=10,
    )

plt.tight_layout()
plt.show()

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "visualizations" / "ofat_plots_test"
os.makedirs(output_dir, exist_ok=True)


# Print summary statistics
print("\n=== OFAT SUMMARY STATISTICS ===")
for var in problem["names"]:
    y_means = np.array(results[var]["y_mean"])
    y_range = np.max(y_means) - np.min(y_means)
    avg_std = np.mean(results[var]["y_std"])
    print(f"{var:15s}: Range = {y_range:.4f}, Avg StdDev = {avg_std:.4f}")

    # Save plots as images

    for var in problem["names"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_vals = results[var]["x"]
        y_means = results[var]["y_mean"]
        y_stds = results[var]["y_std"]

        ax.errorbar(
            x_vals,
            y_means,
            yerr=y_stds,
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=4,
            capthick=1,
            elinewidth=1,
            alpha=0.8,
        )
        for i, (x_val, y_reps) in enumerate(zip(x_vals, results[var]["y_all"])):
            ax.scatter([x_val] * len(y_reps), y_reps, alpha=0.3, s=10, color="gray")
        ax.set_xlabel(var, fontsize=12)
        ax.set_ylabel("Gini Coefficient", fontsize=12)
        ax.set_title(
            f"OFAT: {var} vs Gini Coefficient (n={n} replications)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y_range = np.max(y_means) - np.min(y_means)
        max_std = np.max(y_stds)
        info_text = f"Range: {y_range:.3f}\nMax StdDev: {max_std:.3f}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            fontsize=10,
        )
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ofat_{var}.png"))
        plt.close(fig)
    print(f"Plots saved to '{output_dir}' directory.")
