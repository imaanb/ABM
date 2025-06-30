# Sugarscape G1 + Growback + Trade + Taxing

An **agent‑based recreation** of the *Sugarscape* model described by Epstein & Axtell (1996), written in **Python 3.12**, powered by the bleeding‑edge **Mesa 3** core and exposed through a **Solara** front‑end.

[![Python 3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
## Quick start

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
## Repository layout

| Path                          | Function                                | 
| ----------------------------- | ----------------------------------------- | 
| `src/sugarscape/agents.py`    | Agent classes (`Citizen`, `Patch`, …)     | 
| `src/sugarscape/model.py`     | Core Mesa `Model` implementation          | 
| `src/sugarscape/app.py`       | **Solara** front‑end                      | 
| `scripts/stag.py`             | Visualizations related to stag hunt game  | 
| `scripts/ofat.py`             | Script for performing OFAT                | 
| `notebooks/experiments.ipynb` | Experiments related to Gini Coefficient and Lorenz Curve| 
| `notebooks/final_gsa.ipynb`   | Global‑sensitivity notebook (SALib/Sobol) | 
| `SA_results/`                 | pickle outputs from sweeps                | 
| `visualizations/`             | OFAT and Sobol's method plot              | 
| `.gitignore`                  | Git housekeeping                          | 
| `requirements.txt`            | Runtime dependencies                      | 
---

##  References

* Epstein, J. M. & Axtell, R. L. ***Growing Artificial Societies: Social Science from the Bottom Up***, Brookings Press (1996).
* Mesa Development Team. *Mesa: Agent‑based modeling in Python* (2025). [https://github.com/projectmesa/mesa](https://github.com/projectmesa/mesa)

---
##  License

Released under the **MIT License** – see [LICENSE](LICENSE) for details.
