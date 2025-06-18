from agents import Trader
from model import SugarscapeG1mt


def run_model(n_steps: int = 20):
    # 1 ─ create a fresh model
    # model = SugarscapeG1mt()  # uses all default parameters

    model = SugarscapeG1mt(
        wealth_tax_system="degressive",
        flat_rate=0.03,  # only used if system=="proportional"
        wealth_tax_period=10,  # adjust as you like
        income_tax_system="proportional",  # "none" | "proportional" | "progressive" | "degressive"
        income_tax_flat_rate=0.05,  # used only when system=="proport
        income_tax_brackets=None,  # default brackets will be used if None
        width=50,
        height=50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        enable_trade=True,
        seed=42,
    )
    # 2 ─ advance the simulation
    for _ in range(n_steps):
        model.step()
        print(
            f"step {model.steps:>3} | "
            f"sugar treasury = {model.government_treasury_sugar:8.2f} | "
            f"spice treasury = {model.government_treasury_spice:8.2f} | "
            f"wealth treasury = {model.government_treasury_wealth:8.2f} | "
        )

    # 3 ─ show what the DataCollector recorded
    # df = model.datacollector.get_model_vars_dataframe()

    return model


def sample_income(n_steps=100):
    model = SugarscapeG1mt(
        income_tax_system="none",  # disable tax so we only sample
        wealth_tax_system="none",
        width=50,
        height=50,
    )
    incomes = []
    for _ in range(n_steps):
        model.step()
        for agent in model.agents_by_type[Trader]:
            incomes.append(agent._last_income_sugar)  # assuming you stored it
    print("min:", min(incomes), "max:", max(incomes))


if __name__ == "__main__":
    run_model(50)  # run 30 ticks by default
    # sample_income(100)
