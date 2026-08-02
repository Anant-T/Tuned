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

## Kaggle setup (once)

1. [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook**
2. **File -> Import Notebook** -> upload `notebooks/kaggle_bootstrap.ipynb`
3. Sidebar **Settings -> Internet: On** (needs a phone-verified account)
4. Sidebar **Session options -> Accelerator** -> `GPU T4 x2` or `GPU P100`
5. Run both cells. If cell 2 prints a GPU name, the bridge is live.

## Verifying a push reached Kaggle

Bump `__version__` in `src/tuned/__init__.py`, push, then re-run cells 1 and 2.
The printed `tuned_version` should match what you pushed.

## Free-tier limits worth knowing

- ~30 GPU-hours/week, reset weekly
- 12h max per session (9h on TPU); an idle notebook is killed after 20 min
- `/kaggle/working` persists during the session and caps at 20 GB
- `/kaggle/temp` is scratch and is discarded when the session ends
