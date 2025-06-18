import pkgutil
from io import StringIO

import mesa
import numpy as np
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer

from agents import Trader


# Helper Functions
def flatten(list_of_lists):
    """
    helper function for model datacollector for trade price
    collapses agent price list into one list
    """
    return [item for sublist in list_of_lists for item in sublist]


def geometric_mean(list_of_prices):
    """
    find the geometric mean of a list of prices
    """
    return np.exp(np.log(list_of_prices).mean())


def get_trade(agent):
    """
    For agent reporters in data collector

    return list of trade partners and None for other agents
    """
    if isinstance(agent, Trader):
        return agent.trade_partners
    else:
        return None


class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with Traders
    """

    def __init__(
        self,
        width=50,
        height=50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        wealth_tax_system="progressive",
        flat_rate=0.03,
        wealth_tax_period=10,
        income_tax_system: str = "proportional",  # "none" | "proportional" | "progressive" | "degressive"
        income_tax_flat_rate: float = 0.05,  # used only when system=="proportional"
        income_tax_brackets: list = None,
        enable_trade=True,
        seed=None,
        enable_staghunt: bool = False,
    ):
        super().__init__(seed=seed)
        # Initiate width and height of sugarscape
        self.width = width
        self.height = height

        # Initiate population attributes
        self.enable_trade = enable_trade
        self.running = True

        # ── INCOME TAX ──
        self.income_tax_system = income_tax_system
        self.income_tax_flat_rate = income_tax_flat_rate
        # default brackets if none provided:
        self.income_tax_brackets = income_tax_brackets or [
            (4, 0.02),  # up to 4 units → 2%
            (7, 0.05),  # up to 7 → 5%
            (10, 0.10),  # up to 10 → 10%
            (float("inf"), 0.15),  # above 10 → 15%
        ]

        # Initiate taxes
        self.government_treasury_sugar = 0
        self.government_treasury_spice = 0
        self.government_treasury_wealth = 0

        # -- WEALTH TAX ──
        self.wealth_tax_system = wealth_tax_system
        self.flat_rate = flat_rate
        self.wealth_tax_period = wealth_tax_period

        self.wealth_tax_brackets = [
            (50, 0.02),
            (100, 0.04),
            (200, 0.06),
            (float("inf"), 0.08),
        ]

        # Stag hunt game
        self.enable_staghunt = enable_staghunt
        self.staghunt_payoffs = {
            ("stag", "stag"): (12, 12),
            ("stag", "hare"): (0, 7),
            ("hare", "stag"): (7, 0),
            ("hare", "hare"): (7, 7),
        }

        # initiate mesa grid class
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )
        self.space = self.grid
        # initiate datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "#Traders": lambda m: len(m.agents),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents),
                "Price": lambda m: geometric_mean(
                    flatten([a.prices for a in m.agents])
                ),
                "Sugar Treasury": lambda m: m.government_treasury_sugar,
                "Spice Treasury": lambda m: m.government_treasury_spice,
                "Wealth Treasury": lambda m: m.government_treasury_wealth,
            },
            agent_reporters={"Trade Network": lambda a: get_trade(a)},
        )
        raw = pkgutil.get_data(
            "mesa.examples.advanced.sugarscape_g1mt", "sugar-map.txt"
        )
        # read in landscape file from supplementary material
        # self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        # self.spice_distribution = np.flip(self.sugar_distribution, 1)

        sd = np.genfromtxt(StringIO(raw.decode("utf-8")), dtype=int)
        scale = 3
        sd = (sd * scale).astype(int)
        self.sugar_distribution = sd

        self.spice_distribution = np.flip(sd, axis=1)

        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
        )
        self.grid.add_property_layer(
            PropertyLayer.from_data("spice", self.spice_distribution)
        )

        Trader.create_agents(
            self,
            initial_population,
            self.random.choices(self.grid.all_cells.cells, k=initial_population),
            sugar=self.rng.integers(
                endowment_min, endowment_max, (initial_population,), endpoint=True
            ),
            spice=self.rng.integers(
                endowment_min, endowment_max, (initial_population,), endpoint=True
            ),
            metabolism_sugar=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population,), endpoint=True
            ),
            metabolism_spice=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population,), endpoint=True
            ),
            vision=self.rng.integers(
                vision_min, vision_max, (initial_population,), endpoint=True
            ),
        )
        self.datacollector.collect(self)

    def step(self):
        """
        Unique step function that does staged activation of sugar and spice
        and then randomly activates traders
        """
        # step Resource agents
        growth = self.rng.integers(1, 5, size=self.grid.sugar.data.shape)
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + growth, self.sugar_distribution
        )
        self.grid.spice.data = np.minimum(
            self.grid.spice.data + growth, self.spice_distribution
        )

        # step trader agents
        # to account for agent death and removal we need a separate data structure to
        # iterate
        trader_shuffle = self.agents_by_type[Trader].shuffle()

        for agent in trader_shuffle:
            agent.prices = []
            agent.trade_partners = []
            agent.move()
            agent.eat()
            agent.maybe_die()

        if not self.enable_trade:
            # If trade is not enabled, return early
            self.datacollector.collect(self)
            return

        trader_shuffle = self.agents_by_type[Trader].shuffle()

        for agent in trader_shuffle:
            agent.trade_with_neighbors()

        # 4) WEALTH TAX
        # Every N steps, skim a bit off each agent’s wealth
        if self.steps % self.wealth_tax_period == 0:
            for agent in self.agents_by_type[Trader]:
                agent.pay_wealth_tax()
        # collect model level data
        # fixme we can already collect agent class data
        # fixme, we don't have resource agents anymore so this can be done simpler
        self.datacollector.collect(self)
        """
        Mesa is working on updating datacollector agent reporter
        so it can collect information on specific agents from
        mesa.time.RandomActivationByType.

        Please see issue #1419 at
        https://github.com/projectmesa/mesa/issues/1419
        (contributions welcome)

        Below is one way to update agent_records to get specific Trader agent data
        """
        # Need to remove excess data
        # Create local variable to store trade data
        agent_trades = self.datacollector._agent_records[self.steps]
        # Get rid of all None to reduce data storage needs
        agent_trades = [agent for agent in agent_trades if agent[2] is not None]
        # Reassign the dictionary value with lean trade data
        self.datacollector._agent_records[self.steps] = agent_trades
        if self.steps == 1:
            print(self.datacollector.get_model_vars_dataframe().columns)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
