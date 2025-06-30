# Sugarscape G1 + Growback + Trade + Taxing

An **agent‑based recreation** of the *Sugarscape* model described by Epstein & Axtell (1996). 

[![Python 3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
## Quick start Solara UI

```bash
# 1 – Clone the repo
git clone https://github.com/imaanb/ABM.git
cd ABM

# 2 – Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate

# 3 – Install exact, pinned dependencies
pip install -r requirements.txt

# 4 – Install your package in “editable” mode
pip install -e .

# 5 – (Optional) Verify the install
python -c "import sugarscape; print('sugarscape version:', sugarscape.__version__)"

# 6 – Launch the Solara UI (opens http://localhost:8765)
solara run src/sugarscape/app.py (or run sugarscape.app)
```
---
## Repository layout

| Path                          | Function                                  |
| ----------------------------- | ----------------------------------------- |
| `src/sugarscape/agents.py`    | Agent classes (`Trader`)                  |
| `src/sugarscape/model.py`     | Core Mesa `Model` implementation          |
| `src/sugarscape/app.py`       | **Solara** front-end                      |
| `scripts/stag.py`             | Visualizations related to stag hunt game  |
| `scripts/ofat.py`             | Script for performing OFAT                |
| `notebooks/experiments.ipynb` | Experiments related to Gini Coefficient and Lorenz Curve |
| `notebooks/final_gsa.ipynb`   | Global-sensitivity notebook (SALib/Sobol) |
| `data/`                       | Pickle outputs from GSA                   |
| `visualizations/`             | OFAT and Sobol’s method plot              |
| `.gitignore`                  | Gitignore                                 |
| `requirements.txt`            | Module requirements                       |
| `setup.py`                    | Build and install script                   |




##  References

* Epstein, J. M. & Axtell, R. L. ***Growing Artificial Societies: Social Science from the Bottom Up***, Brookings Press (1996).
* Mesa Development Team. *Mesa: Agent‑based modeling in Python* (2025). [https://github.com/projectmesa/mesa](https://github.com/projectmesa/mesa)

---
##  License

Released under the **MIT License** – see [LICENSE](LICENSE) for details.
