import matplotlib.pyplot as plt
from model import SugarscapeG1mtggtt
# After running your model:
model = SugarscapeG1mtggtt()
model.run_model(step_count=10)

# Get the collected data
results = model.datacollector.get_model_vars_dataframe()
print(results)
# Plot VAT over time
plt.figure(figsize=(10, 5))
plt.plot(results["VAT Sugar"], label="VAT Sugar")
plt.plot(results["VAT Spice"], label="VAT Spice")
plt.plot(results["VAT Total"], label="VAT Total", linewidth=2)
plt.xlabel("Step")
plt.ylabel("VAT Treasury")
plt.title("VAT Treasury Over Time")
plt.legend()
plt.grid()
plt.show()