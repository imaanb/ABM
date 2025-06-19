import solara

# from mesa.examples.advanced.sugarscape_g1mt.model import SugarscapeG1mt
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure

from model import SugarscapeG1mt

print("🐝 Loading APP.PY at", __file__)


def agent_portrayal(agent):
    return AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        color="black",
        marker="o",
        size=10,
        zorder=1,
    )


def propertylayer_portrayal(layer):
    if layer.name == "sugar":
        return PropertyLayerStyle(
            color="blue", alpha=0.8, colorbar=True, vmin=0, vmax=10
        )
    return PropertyLayerStyle(color="red", alpha=0.8, colorbar=True, vmin=0, vmax=10)


@solara.component
def LorenzPlot(model):
    # Holy line of code to make the custom plot also update: https://mesa.readthedocs.io/stable/tutorials/visualization_tutorial.html
    update_counter.get()

    fig = Figure()
    ax = fig.add_subplot(111)

    # Lorenz curve 
    lorenz = model.datacollector.get_model_vars_dataframe().iloc[-1]["Lorenz"]
    if lorenz:
        x, y = zip(*lorenz)
        ax.plot(x, y, label="Lorenz Curve", color="blue")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Equality")
        ax.set_title("Lorenz Curve")
        ax.set_xlabel("Cumulative Population Share")
        ax.set_ylabel("Cumulative Wealth Share")
        ax.legend()

    return solara.FigureMatplotlib(fig)


@solara.component
def TreasuryDisplay(model):
    with solara.Card("Government Treasuries"):
        solara.Text(f"Income‐tax sugar:  {model.government_treasury_sugar:.2f}")
        solara.Text(f"Income‐tax spice:  {model.government_treasury_spice:.2f}")
        solara.Text(f"Wealth‐tax sugar: {model.government_treasury_wealth:.2f}")


sugarscape_space = make_mpl_space_component(
    agent_portrayal=agent_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=None,
    draw_grid=False,
)

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": 50,
    "height": 50,
    # Population parameters
    "initial_population": Slider(
        "Initial Population", value=200, min=50, max=500, step=10
    ),
    # Agent endowment parameters
    "endowment_min": Slider("Min Initial Endowment", value=25, min=5, max=30, step=1),
    "endowment_max": Slider("Max Initial Endowment", value=50, min=30, max=100, step=1),
    # Metabolism parameters
    "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=3, step=1),
    "metabolism_max": Slider("Max Metabolism", value=5, min=3, max=15, step=1),
    # Vision parameters
    "vision_min": Slider("Min Vision", value=1, min=1, max=3, step=1),
    "vision_max": Slider("Max Vision", value=5, min=3, max=8, step=1),
    # Trade parameter
    "enable_trade": {"type": "Checkbox", "value": True, "label": "Enable Trading"},
    "enable_staghunt": {
        "type": "Checkbox",
        "value": False,
        "label": "Enable Stag Hunt",
    },
}

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
    enable_staghunt=False,
)

# model = SugarscapeG1mt()

Page = SolaraViz(
    model,
    components=[
        sugarscape_space,
        make_plot_component("#Traders"),
        make_plot_component("Price"),
        make_plot_component("Gini"),
        LorenzPlot,
        # TreasuryDisplay,
        # make_plot_component("Wealth Treasury"),
        # make_plot_component("Spice Treasury"),
    ],
    model_params=model_params,
    name="Sugarscape {G1, M, T}",
    play_interval=150,
)
Page  # noqa
