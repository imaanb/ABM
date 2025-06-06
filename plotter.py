import matplotlib.pyplot as plt
from model import SugarscapeG1mt

# Create and run the model
model = SugarscapeG1mt()
for _ in range(100):  # Run for 100 steps
    model.step()

# Get the collected data
df = model.datacollector.get_model_vars_dataframe()

# Plot #Traders and Price
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Step')
ax1.set_ylabel('#Traders', color=color)
ax1.plot(df.index, df["#Traders"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Price', color=color)
ax2.plot(df.index, df["Price"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Sugarscape: #Traders and Price over Time")
plt.show()