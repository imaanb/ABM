import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import SugarscapeG1mt
# Parameters for sweep
p_copy_values = np.linspace(0, 1, 20)   # e.g., 0.0 to 1.0 in 0.1 steps
p_mutate_values = np.linspace(0, 0.1, 20)  # Mutation tends to be low

# Store results
results = np.zeros((len(p_copy_values), len(p_mutate_values)))

# Run model for each combination
for i, p_copy in enumerate(tqdm(p_copy_values, desc="p_copy sweep")):
    for j, p_mutate in enumerate(p_mutate_values):
        # Initialize model
        model = SugarscapeG1mt(
            p_copy=p_copy,
            p_mutate=p_mutate,
            enable_staghunt=True,
            initial_population=100,
            width=50,
            height=50,
            seed=42  # Optional: Fix seed for reproducibility
        )

        # Run model
        model.run_model(100)

        # Get the last fraction of "stag" players
        data = model.datacollector.get_model_vars_dataframe()
        frac_stag = data["Frac_stag"].iloc[-1]  # Take last step

        # Store result
        results[i, j] = frac_stag


# ✅ Plotting the result as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    results,
    xticklabels=np.round(p_mutate_values, 3),
    yticklabels=np.round(p_copy_values, 2),
    cmap="YlGnBu",
    annot=True,
    cbar_kws={"label": "Fraction Choosing Stag"},
)

plt.title("Stag Hunt Equilibrium vs. p_copy and p_mutate")
plt.xlabel("p_mutate (mutation rate)")
plt.ylabel("p_copy (imitation probability)")
plt.tight_layout()
plt.show()
