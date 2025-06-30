# Sugarscape G1 + Growback + Trade

An **agent‑based recreation** of the *Sugarscape* model described by Epstein & Axtell (1996), written in Python 3.12 with the latest development version of **Mesa 3** and a **Solara** front‑end.

[![Python 3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Quick start

```bash
# 1 – clone
$ git clone https://github.com/imaanb/ABM
$ cd <repo>

# 2 – virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

# 3 – install dependencies
$ pip install -r requirements.txt

# 4 – launch the interactive UI
$ solara run app.py            # open http://localhost:8765 in your browser
```

The browser window lets you **start / pause / reset** the simulation and track
population‑level metrics in real time.

---

## 🗺️ Repository layout

| Path                | What it is                                     | Why you might open it                              |
| ------------------- | ---------------------------------------------- | -------------------------------------------------- |
| `agents.py`         | Agent classes (Citizen, SugarPatch, etc.)      | Change individual‑level rules or attributes        |
| `model.py`          | Core Mesa `Model` implementation               | Tweak global parameters or add new data collectors |
| `app.py`            | Solara front‑end glue code                     | Modify the UI or embed additional plots            |
| `stag.py`           | CLI script that runs a headless simulation     | Quick batch runs without the UI                    |
| `stag_p.py`         | Parallelised variant using Mesa’s batch runner | Speed‑up for heavy experiments                     |
| `ofat.py`           | **One‑factor‑at‑a‑time** sensitivity analysis  | Inspect effect of a single parameter               |
| `experiments.ipynb` | Notebook with exploratory experiments          | Prototype ideas & visuals in Jupyter               |
| `final_gsa.ipynb`   | Global Sensitivity Analysis (SALib / Sobol)    | Reproduce figures for the report                   |
| `SA_results/`       | Pickled or CSV outputs from sensitivity runs   | Used by notebooks & visualisation scripts          |
| `visualizations/`   | Matplotlib / Plotly figures                    | Ready‑to‑use images for papers or slides           |
| `.gitignore`        | Git housekeeping                               | Keeps junk out of the repo                         |
| `requirements.txt`  | Exact Python dependencies                      | Freeze to reproduce environments                   |

---

## 📚 References

* Epstein, J. M. & Axtell, R. L. ***Growing Artificial Societies: Social Science from the Bottom Up***, Brookings Press (1996).
* Mesa Development Team. *Mesa: Agent‑based modeling in Python* (2025).  [https://github.com/projectmesa/mesa](https://github.com/projectmesa/mesa)

---

## ⚖️ License

Released under the **MIT License** – see [LICENSE](LICENSE) for details.
