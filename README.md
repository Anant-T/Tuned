# tuned

Local code, run on Kaggle's free GPU. GitHub is the bridge.

```
edit locally  ->  git push  ->  re-run Kaggle cell 1  ->  train
```

The bridge is **one-way**. Kaggle pulls from GitHub and never pushes back.
Model weights and datasets stay out of git (see `.gitignore`) - they belong in
`/kaggle/working`, and you retrieve them from the notebook's **Output** tab.

## Layout

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package. Kaggle puts `src/` on `sys.path`. |
| `notebooks/kaggle_bootstrap.ipynb` | Import this into Kaggle once. |

## Local setup (once)

Matches Kaggle's interpreter exactly, so imports that work here work there.

```powershell
uv venv                      # reads .python-version -> CPython 3.12.13
.venv\Scripts\Activate.ps1
uv pip install -e .          # makes `import tuned` work without sys.path hacks
```

The version lives in one place: `__version__` in `src/tuned/__init__.py`.
Hatchling reads it from there, so bumping that line is the whole release process.

## Kaggle setup (once)

1. [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook**
2. **File -> Import Notebook** -> upload `notebooks/kaggle_bootstrap.ipynb`
3. Sidebar **Settings -> Internet: On** (needs a phone-verified account)
4. Sidebar **Session options -> Accelerator** -> `GPU T4 x2` or `GPU P100`
5. Run both cells. If cell 2 prints a GPU name, the bridge is live.

## Verifying a push reached Kaggle

Bump `__version__` in `src/tuned/__init__.py`, push, then re-run cells 1 and 2.
The printed `tuned_version` should match what you pushed.

## Two constraints that bite

**Target Python 3.12.** Kaggle runs 3.12.13 (verified). Code in `src/` that uses
newer syntax will import locally and fail there.

**Only `src/` crosses the bridge.** The Kaggle notebook is a detached copy made
at import time - edits to `notebooks/kaggle_bootstrap.ipynb` here do *not* reach
it. Keep the bootstrap cell thin; put real logic in `src/`, which is pulled fresh
every run.

## Free-tier limits worth knowing

- ~30 GPU-hours/week, reset weekly
- Verified accelerator: `GPU T4 x2` = 2x Tesla T4, 15360 MiB each
- 12h max per session (9h on TPU); an idle notebook is killed after 20 min
- `/kaggle/working` persists during the session and caps at 20 GB
- `/kaggle/temp` is scratch and is discarded when the session ends
