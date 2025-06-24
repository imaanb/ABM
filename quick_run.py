import itertools

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


def test_staghunt_payoff():
    model = SugarscapeG1mt(
        width=50,
        height=50,
        initial_population=2,
        vision_min=1,
        vision_max=1,
        enable_trade=False,
        enable_staghunt=True,
        seed=123,
    )
    a, b = list(model.agents_by_type[Trader])

    # 2) Zero out both agents so we see only the bonus
    a.sugar = a.spice = 0
    b.sugar = b.spice = 0

    # 3) Fix their strategies to a known pair
    a.strategy = "stag"
    b.strategy = "stag"

    # 4) Monkey-patch a.cell.get_neighborhood to return b’s cell
    #    so play_staghunt() will pick b as the partner.
    a.cell.get_neighborhood = lambda vision, include_center=False: [b.cell]

    # 5) Invoke the hunt
    bonus = a.play_staghunt()

    # 6) Look up expected payoffs
    expected_sugar, expected_spice = model.staghunt_payoffs[("stag", "stag")]

    # 7) Assertions
    assert bonus == expected_sugar, (
        f"Returned bonus {bonus} != expected {expected_sugar}"
    )
    assert b.sugar == expected_sugar, (
        f"Partner sugar {b.sugar} != expected {expected_sugar}"
    )
    assert b.spice == expected_spice, (
        f"Partner spice {b.spice} != expected {expected_spice}"
    )

    print("play_staghunt() is awarding (sugar, spice) correctly!")


if __name__ == "__main__":
    # run_model(50)  # run 30 ticks by default
    # sample_income(100)
    # test_staghunt_payoff()  # run the staghunt test

    # Choose your grid
    p_copy_vals = [0.5, 0.7, 0.9, 1.0]
    p_mutate_vals = [0.005, 0.02, 0.05, 0.1]

    results = {}
    for p_copy, p_mutate in itertools.product(p_copy_vals, p_mutate_vals):
        model = SugarscapeG1mt(
            enable_staghunt=True,
            p_copy=p_copy,
            p_mutate=p_mutate,
            initial_population=200,
            seed=0,
        )
        n = len(model.agents)
        n_stag = sum(1 for a in model.agents if a.strategy == "stag")
        print(f"At step 0: Frac_stag = {n_stag}/{n} = {n_stag / n:.2f}")
        # run long enough to reach equilibrium
        for _ in range(200):
            model.step()

        frac = model.datacollector.get_model_vars_dataframe()["Frac_stag"].iloc[-1]
        results[(p_copy, p_mutate)] = frac

    # Print a heat-map table
    for (pc, pm), frac in results.items():
        print(f"p_copy={pc:<4} p_mutate={pm:<5} → Frac_stag ≃ {frac:.2f}")
