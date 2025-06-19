from pathlib import Path

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


class SugarscapeG1mtggtt(mesa.Model):
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
        enable_trade=True,
        seed=None,
        
    ):
        super().__init__(seed=seed)
        # Initiate width and height of sugarscape
        self.width = width
        self.height = height


        self.treasury = {"Sugar": 0 , "Spice": 0 } # Treasury where tax will be collected 
        self.vat_rate_sugar = .05 # VAT rate 
        self.vat_rate_spice = .5 # VAT rate 
        self.redistribution_regime = "geographic" # Choose from "geographic", "proportional", "social" 



        # Initiate population attributes
        self.enable_trade = enable_trade
        self.running = True

        # initiate mesa grid class
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )
        # initiate datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "#Traders": lambda m: len(m.agents),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents),
                "Price": lambda m: geometric_mean(flatten([a.prices for a in m.agents])),
                "VAT Sugar": lambda m: m.treasury["Sugar"],      # <-- add this
                "VAT Spice": lambda m: m.treasury["Spice"],      # <-- add this
                "VAT Total": lambda m: m.treasury["Sugar"] + m.treasury["Spice"],  # <-- add this
            },
            agent_reporters={"Trade Network": lambda a: get_trade(a)},
        )

        # read in landscape file from supplementary material
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.spice_distribution = np.flip(self.sugar_distribution, 1)

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

        def redistribute_tax(self):
            """
            Redistribute the tax collected in the treasury according to the selected regime.
            - 'geographic': Place resources back on random tiles.
            - 'proportional': Give all agents a percentage of the treasury.
            - 'social': Give poorer agents (with less total wealth) a larger share.
            """
            if self.redistribution_regime == "geographic":
                # Place sugar and spice back on all tiles equally
                for resource in ["sugar", "spice"]:
                    amount = int(self.treasury[resource.capitalize()])
                    if amount > 0:
                        per_tile = amount // (self.width * self.height)
                        if per_tile > 0:
                            layer = getattr(self.grid, resource)
                            max_vals = getattr(self, f"{resource}_distribution")
                            for x in range(self.width):
                                for y in range(self.height):
                                    layer.data[x, y] = min(layer.data[x, y] + per_tile, max_vals[x, y])
                            self.treasury[resource.capitalize()] -= per_tile * self.width * self.height

            elif self.redistribution_regime == "proportional":
                # Distribute treasury equally among all agents
                for resource in ["Sugar", "Spice"]:
                    total = self.treasury[resource]
                    if len(self.agents) > 0 and total > 0:
                        share = total // len(self.agents)
                        for agent in self.agents:
                            setattr(agent, resource.lower(), getattr(agent, resource.lower()) + share)
                        self.treasury[resource] -= share * len(self.agents)

            elif self.redistribution_regime == "social":  # low assets get more
                for resource in ["Sugar", "Spice"]:
                    total = self.treasury[resource]
                    if total > 0 and len(self.agents) > 0:
                        # Calculate total wealth for each agent
                        agent_wealth = [(agent, getattr(agent, "sugar") + getattr(agent, "spice")) for agent in self.agents]
                        agent_wealth.sort(key=lambda x: x[1])  # Sort by wealth ascending
                        # Assign weights inversely proportional to wealth (poorer get more)
                        ranks = np.arange(1, len(agent_wealth) + 1)
                        weights = (len(agent_wealth) + 1 - ranks)
                        weights = weights / weights.sum()
                        distributed = 0
                        for (agent, _), w in zip(agent_wealth, weights):
                            give = int(w * total)
                            setattr(agent, resource.lower(), getattr(agent, resource.lower()) + give)
                            distributed += give
                        self.treasury[resource] -= distributed




    

    def step(self):
        """
        Unique step function that does staged activation of sugar and spice
        and then randomly activates traders
        """
        # step Resource agents
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + 1, self.sugar_distribution
        )
        self.grid.spice.data = np.minimum(
            self.grid.spice.data + 1, self.spice_distribution
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

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
