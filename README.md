# Sugarscape G1 + Growback + Trade

An **agent‑based recreation** of the *Sugarscape* model described by Epstein & Axtell (1996), written in **Python 3.12**, powered by the bleeding‑edge **Mesa 3** core and exposed through a **Solara** front‑end.

[![Python 3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

\## Quick start

```bash
# 1 – clone
$ git clone https://github.com/imaanb/ABM.git
$ cd ABM

# 2 – virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3 – install runtime deps
$ pip install -r requirements.txt

# 4 – launch the interactive UI
$ solara run src/sugarscape/app.py    # opens http://localhost:8765
```

The browser window lets you **start / pause / reset** the simulation and watch population‑level metrics in real time.

---

\## Repository layout (2025‑06)

| Path                          | What it is                                | Why you might open it                       |
| ----------------------------- | ----------------------------------------- | ------------------------------------------- |
| `src/sugarscape/agents.py`    | Agent classes (`Citizen`, `Patch`, …)     | Modify individual‑level rules or attributes |
| `src/sugarscape/model.py`     | Core Mesa `Model` implementation          | Tweak global parameters, add collectors     |
| `src/sugarscape/app.py`       | **Solara** front‑end                      | Change the UI or embed extra plots          |
| `scripts/stag.py`             | Headless run → saves figures              | Quick batch experiments without UI          |
| `scripts/ofat.py`             | One‑factor‑at‑a‑time sweep                | Single‑parameter sensitivity analysis       |
| `notebooks/experiments.ipynb` | Free‑form exploration                     | Prototype ideas & visuals in Jupyter        |
| `notebooks/final_gsa.ipynb`   | Global‑sensitivity notebook (SALib/Sobol) | Reproduce report figures                    |
| `SA_results/`                 | CSV/pickle outputs from sweeps            | Consumed by notebooks & visual scripts      |
| `visualizations/`             | Matplotlib / Plotly figures               | Ready‑to‑use images for talks & papers      |
| `.gitignore`                  | Git housekeeping                          | Keeps caches & OS junk out of the repo      |
| `requirements.txt`            | Runtime dependencies                      | Reproduce the environment                   |

*(Development helpers such as `black`, `ruff`, or `ipykernel` live in `requirements-dev.txt`, excluded here for brevity.)*

---

\##  References

* Epstein, J. M. & Axtell, R. L. ***Growing Artificial Societies: Social Science from the Bottom Up***, Brookings Press (1996).
* Mesa Development Team. *Mesa: Agent‑based modeling in Python* (2025). [https://github.com/projectmesa/mesa](https://github.com/projectmesa/mesa)

---

\##  License

Released under the **MIT License** – see [LICENSE](LICENSE) for details.
