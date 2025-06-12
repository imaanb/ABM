# Sugarscape G1 + Growback + Trade  
_Agent-Based Model in Mesa 3 (dev) with a Solara Web UI_

[![Python 3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An interactive re-implementation of the **Sugarscape Growback + Trade**
> model (Epstein & Axtell, 1996).  
> Written in Python 3.12, rendered via **[Solara](https://solara.dev)**,
> and powered by the latest _Mesa_ development branch.

---

## ✨ Demo

```bash
solara run app.py
# ▶︎ Serving at http://localhost:8765

# 1 — clone
git clone https://github.com/<your-org>/<repo>.git
cd <repo>

# 2 — create & activate virtual-env
python3.12 -m venv .venv
source .venv/bin/activate

# 3 — install deps
pip install -r requirements.txt

# 4 — launch web UI
solara run app.py          # open http://localhost:8765
