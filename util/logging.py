from __future__ import annotations

import logging
import time
from pathlib import Path


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    """Create a root logger with console output and optional file logging."""
    if log_file:
        log_path = Path(log_file)
        if log_path.exists():
            log_path.unlink()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


class ProgressBar(object):
    """
    Simple terminal progress bar.

    Example:
        >>> pbar = ProgressBar(n_total=30, desc="Training")
        >>> pbar(step=0)
    """

    def __init__(self, n_total, width=30, desc="Training"):
        """Initialize display width, total steps, and description."""
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info=None):
        """Render the progress bar for one step with optional scalar metrics."""
        info = info or {}
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f"[{self.desc}] {current}/{self.n_total} ["
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += "=" * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += "="
        bar += "." * (self.width - prog_width)
        bar += "]"
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = "%d:%02d:%02d" % (
                    eta // 3600,
                    (eta % 3600) // 60,
                    eta % 60,
                )
            elif eta > 60:
                eta_format = "%d:%02d" % (eta // 60, eta % 60)
            else:
                eta_format = "%ds" % eta
            time_info = f" - ETA: {eta_format}"
        else:
            if time_per_unit >= 1:
                time_info = f" {time_per_unit:.1f}s/step"
            elif time_per_unit >= 1e-3:
                time_info = f" {time_per_unit * 1e3:.1f}ms/step"
            else:
                time_info = f" {time_per_unit * 1e6:.1f}us/step"

        show_bar += time_info
        if len(info) != 0:
            show_info = f"{show_bar} " + "-".join(
                [f" {key}: {value:.4f} " for key, value in info.items()]
            )
            print(show_info, end="")
        else:
            print(show_bar, end="")
