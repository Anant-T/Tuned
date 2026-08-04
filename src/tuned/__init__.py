"""tuned - multi-adapter fine-tuning of Ministral-3-14B-Reasoning on Kaggle free-tier GPUs.

Model weights never live in this package.
"""

__version__ = "0.1.1"


def where_am_i():
    """Report the runtime environment.

    Run this in a fresh Kaggle session to verify the environment.
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

    run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    info = {
        "tuned_version": __version__,
        "platform": f"kaggle ({run_type})" if run_type else "local/other",
        "host": platform.node(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "package_path": os.path.dirname(__file__),
    }

    pad = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{pad}} : {value}")
    return info
