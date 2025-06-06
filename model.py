from pathlib import Path
import numpy as np
import mesa
from mesa.space import MultiGrid
from mesa.examples.advanced.sugarscape_g1mt.agents import Trader

# Helper Functions
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def geometric_mean(list_of_prices):
    return np.exp(np.log(list_of_prices).mean()) if list_of_prices else 0

def get_trade(agent):
    if isinstance(agent, Trader):
        return agent.trade_partners
    else:
        return None

class SugarscapeG1mt(mesa.Model):
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

        self.width = width
        self.height = height
        self.enable_trade = enable_trade
        self.running = True

        self.grid = MultiGrid(self.width, self.height, torus=False)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "#Traders": lambda m: len(m.agents_by_type[Trader]),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents_by_type[Trader]),
                "Price": lambda m: geometric_mean(
                    flatten([a.prices for a in m.agents_by_type[Trader]])
                ),
            },
            agent_reporters={"Trade Network": lambda a: get_trade(a)},
        )

        # Load and store resource distributions
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.spice_distribution = np.flip(self.sugar_distribution, 1)

        self.sugar_map = {}
        self.spice_map = {}

        for x in range(self.width):
            for y in range(self.height):
                self.sugar_map[(x, y)] = self.sugar_distribution[y, x]
                self.spice_map[(x, y)] = self.spice_distribution[y, x]

        # Create Trader agents
        Trader.create_agents(
            self,
            initial_population,
            self.random.choices(list(self.grid.coord_iter()), k=initial_population),
            sugar=self.rng.integers(
                endowment_min, endowment_max + 1, (initial_population,)
            ),
            spice=self.rng.integers(
                endowment_min, endowment_max + 1, (initial_population,)
            ),
            metabolism_sugar=self.rng.integers(
                metabolism_min, metabolism_max + 1, (initial_population,)
            ),
            metabolism_spice=self.rng.integers(
                metabolism_min, metabolism_max + 1, (initial_population,)
            ),
            vision=self.rng.integers(
                vision_min, vision_max + 1, (initial_population,)
            ),
        )

    def step(self):
        # Regrow resources up to maximum
        for x in range(self.width):
            for y in range(self.height):
                self.sugar_map[(x, y)] = min(self.sugar_map[(x, y)] + 1, self.sugar_distribution[y, x])
                self.spice_map[(x, y)] = min(self.spice_map[(x, y)] + 1, self.spice_distribution[y, x])

        # Step trader agents (move/eat/die)
        trader_shuffle = self.agents_by_type[Trader].shuffle()

        for agent in trader_shuffle:
            agent.prices = []
            agent.trade_partners = []
            agent.move()
            agent.eat()
            agent.maybe_die()

        if not self.enable_trade:
            self.datacollector.collect(self)
            return

        trader_shuffle = self.agents_by_type[Trader].shuffle()

        for agent in trader_shuffle:
            agent.trade_with_neighbors()

        self.datacollector.collect(self)

        # Clean agent records
        agent_trades = self.datacollector._agent_records[self.steps]
        agent_trades = [agent for agent in agent_trades if agent[2] is not None]
        self.datacollector._agent_records[self.steps] = agent_trades

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
    @classmethod
    def create_agents(cls, model, n, positions, **kwargs):
        for i in range(n):
            trader = cls(
                unique_id=i,
                model=model,
                pos=positions[i],
                sugar=kwargs["sugar"][i],
                spice=kwargs["spice"][i],
                metabolism_sugar=kwargs["metabolism_sugar"][i],
                metabolism_spice=kwargs["metabolism_spice"][i],
                vision=kwargs["vision"][i],
            )
            model.grid.place_agent(trader, positions[i])
            model.schedule.add(trader)