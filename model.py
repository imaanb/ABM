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


# Measurement functions
# Gini coefficient
def compute_gini(model):
    # wealths = [a.calculate_welfare(a.sugar, a.spice) for a in model.agents]
    wealths = [a.wealth for a in model.agents]
    if not wealths:
        return 0
    sorted_wealths = np.sort(np.array(wealths))
    n = len(wealths)
    cum_wealth = np.cumsum(sorted_wealths)
    gini = 1 - 2 * np.sum(cum_wealth) / (n * cum_wealth[-1]) + (1 / n)
    return gini


# Lorenz
def compute_lorenz(model):
    # wealths = [a.calculate_welfare(a.sugar, a.spice) for a in model.agents]
    wealths = [a.wealth for a in model.agents]
    if not wealths:
        return []

    sorted_wealths = np.sort(np.array(wealths))
    cum_wealth = np.cumsum(sorted_wealths)
    total_wealth = cum_wealth[-1]

    lorenz_curve = np.insert(cum_wealth / total_wealth, 0, 0)
    x = np.linspace(0, 1, len(lorenz_curve))

    return list(zip(x.tolist(), lorenz_curve.tolist()))


class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with Traders
    """

    def __init__(
        self,
        # general
        width=50,
        height=50,
        initial_population=200,
        # agents settings
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        # taxes
        wealth_tax_system="progressive",
        flat_rate=0.03,
        wealth_tax_period=10,
        income_tax_system: str = "proportional",  # "none" | "proportional" | "progressive" | "degressive"
        income_tax_flat_rate: float = 0.05,  # used only when system=="proportional"
        income_tax_brackets: list = None,
        vat_rate_sugar=0.2,
        vat_rate_spice=0.2,
        # redistribution
        redistribution_regime="proportional",  # "social" or "proportional" or "none"
        enable_trade=True,
        seed=None,
        enable_staghunt: bool = False,
    ):
        super().__init__(seed=seed)

        # Initiate width and height of sugarscape

        self.width = width
        self.height = height

        # -- TREASURY -- #
        self.treasury = {"sugar": 0, "spice": 0}  # Treasury where tax will be collected
        # self.vat_rate_sugar = .2 # VAT rate
        # self.vat_rate_spice = .2 # VAT rate
        # self.redistribution_regime = "proportional" # Choose from "geographic", "proportional", "social"
        self.resources = ["sugar", "spice"]
        """
        # Initiate taxes
        self.government_treasury_sugar = 0
        self.government_treasury_spice = 0
        self.government_treasury_wealth = 0
        """

        # Initiate population attributes
        self.enable_trade = enable_trade
        self.running = True

        # ── INCOME TAX ──
        self.income_tax_system = income_tax_system
        self.income_tax_flat_rate = income_tax_flat_rate
        # default brackets if none provided:
        self.income_tax_brackets = income_tax_brackets or [
            (1, 0.025),  # up to 4 units → 2%
            (2, 0.05),  # up to 7 → 5%
            (3, 0.075),  # up to 10 → 10%
            (float("inf"), 0.1),  # above 10 → 15%
        ]

        # -- WEALTH TAX ──
        self.wealth_tax_system = wealth_tax_system
        self.flat_rate = flat_rate
        self.wealth_tax_period = wealth_tax_period

        self.wealth_tax_brackets = [
            (50, 0.05),
            (100, 0.1),
            (200, 0.2),
            (float("inf"), 0.3),
        ]

        # -- VAT --
        self.vat_rate_sugar, self.vat_rate_spice = vat_rate_sugar, vat_rate_spice
        self.VAT_collected = 0  # cummualtive

        ## -- TRADING --
        self.trades_made = 0  # cummulative

        # -- REDISTRIBUTION --
        self.redistribution_regime = redistribution_regime
        self.redistributed = 0  # cummulative
        self.redistribution_regime = redistribution_regime

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
                "Treasury Sugar": lambda m: m.treasury["sugar"],
                "Treasury Spice": lambda m: m.treasury["spice"],
                "Treasury Total": lambda m: m.treasury["sugar"] + m.treasury["spice"],
                "redistributed cummulative": lambda m: m.redistributed,
                "trading": lambda m: m.trades_made,
                # "Sugar Treasury": lambda m: m.government_treasury_sugar,
                # "Spice Treasury": lambda m: m.government_treasury_spice,
                # "Wealth Treasury": lambda m: m.government_treasury_wealth,
                "Gini": compute_gini,
                "Lorenz": compute_lorenz,
                "Average Wealth": lambda m: np.mean([a.sugar + a.spice for a in m.agents]) if m.agents else 0,
            
            },
            agent_reporters={
                "Trade Network": lambda a: get_trade(a),
                "sugar": lambda a: a.sugar,
                "spice": lambda a: a.spice,
                "Wealth": lambda a: a.sugar + a.spice,
            },
        )
        raw = pkgutil.get_data(
            "mesa.examples.advanced.sugarscape_g1mt", "sugar-map.txt"
        )
        # read in landscape file from supplementary material
        # self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        # self.spice_distribution = np.flip(self.sugar_distribution, 1)

        sd = np.genfromtxt(StringIO(raw.decode("utf-8")), dtype=int)
        # scale = 3
        # sd = (sd * scale).astype(int)
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

    def redistribute_tax(self):
        """
        Redistribute the tax collected in the treasury according to the selected regime.
        - 'proportional': Give all agents a percentage of the treasury.
        - 'social': Give poorer agents (with less total wealth) a larger share.
        """
        # print("hi")
        # skip if none
        if self.redistribution_regime == "none":
            return

        # print(
        #     f"[step {self.steps}] Redistribution regime: {self.redistribution_regime}"
        # )
        # print(
        #     f"  Treasury sugar: {self.treasury['sugar']:.2f}, spice: {self.treasury['spice']:.2f}"
        # )

        # proportinal: each agent gets equal amount
        if self.redistribution_regime == "proportional":
            for resource in self.resources:
                total = self.treasury[resource]
                if len(self.agents) > 0 and total > 0:
                    share = total / len(self.agents)
                    for agent in self.agents:
                        if self.treasury[resource] >= share:
                            setattr(
                                agent,
                                resource.lower(),
                                getattr(agent, resource.lower()) + share,
                            )
                            self.treasury[resource] -= share
                            self.redistributed += share
                        else:
                            # Not enough left in treasury for a full share, give what's left and break
                            setattr(
                                agent,
                                resource.lower(),
                                getattr(agent, resource.lower())
                                + self.treasury[resource],
                            )
                            self.redistributed += self.treasury[resource]
                            self.treasury[resource] = 0
                            break

        # Social: poor (assest based) get more
        elif self.redistribution_regime == "social":
            for resource in self.resources:
                total = self.treasury[resource]
                if total > 0 and len(self.agents) > 0:
                    # calculate total wealth for each agent
                    agent_wealth = [
                        (agent, getattr(agent, "sugar") + getattr(agent, "spice"))
                        for agent in self.agents
                    ]
                    agent_wealth.sort(key=lambda x: x[1])  # Sort by wealth ascending

                    # Assign weights inversely proportional to wealth
                    ranks = np.arange(1, len(agent_wealth) + 1)
                    weights = len(agent_wealth) + 1 - ranks
                    weights = weights / weights.sum()
                    distributed = 0
                    for (agent, _), w in zip(agent_wealth, weights):
                        give = w * total
                        give = min(
                            give, self.treasury[resource]
                        )  # Ensure treasury has enough
                        if give <= 0:
                            break
                        setattr(
                            agent,
                            resource.lower(),
                            getattr(agent, resource.lower()) + give,
                        )
                        distributed += give
                        self.treasury[resource] -= give
                        if self.treasury[resource] <= 0:
                            break
                    self.redistributed += distributed

        else:
            print("redistribution regime not known")

    def step(self):
        """
        Unique step function that does staged activation of sugar and spice
        and then randomly activates traders

        NOTE: MAX GROWTH CAPPED AT 4, see growth = self.rng.integers(1, 5..
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

        # if not self.enable_trade:
        #     # If trade is not enabled, return early
        #     self.datacollector.collect(self)
        #     return

        trader_shuffle = self.agents_by_type[Trader].shuffle()

        for agent in trader_shuffle:
            agent.trade_with_neighbors()

        # 4) WEALTH TAX
        # Every N steps, skim a bit off each agent’s wealth
        if self.steps % self.wealth_tax_period == 0:
            for agent in self.agents_by_type[Trader]:
                agent.pay_wealth_tax()
            # outside or inside if statement?
            self.redistribute_tax()
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
