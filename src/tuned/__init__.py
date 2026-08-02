"""tuned - code that runs locally and on Kaggle's free GPU.

The bridge is one-way: you edit here, push to GitHub, and the Kaggle
notebook pulls. Model weights never live in this package.
"""

__version__ = "0.1.1"


def where_am_i():
    """Report the runtime environment.

    Run this in the notebook right after the bootstrap cell. If the printed
    ``tuned_version`` matches what you just pushed, the bridge is live.
    """
    import os
    import platform
    import shutil
    import subprocess
    import sys

    gpu = "none detected"
    if shutil.which("nvidia-smi"):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                gpu = "; ".join(l.strip() for l in probe.stdout.strip().splitlines())
            else:
                gpu = "nvidia-smi present but returned nothing"
        except Exception as exc:
            gpu = f"nvidia-smi failed: {exc}"

    info = {
        "tuned_version": __version__,
        "host": "kaggle" if os.path.isdir("/kaggle") else platform.node(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "package_path": os.path.dirname(__file__),
    }

    pad = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{pad}} : {value}")
    return info
