"""
===========================================================================
Streamlit Log Capture
Description: Streams Python logging output into a Streamlit st.status box,
so slow first-call operations (cold model/embedding caches, AlphaFold/P2Rank
fetches) give the user real-time feedback instead of a silent spinner.
===========================================================================

Workflow:
1. run_with_log_feedback opens a collapsible st.status box and attaches a
   logging.Handler to the root logger for the duration of the call,
   streaming every emitted log record (thermokp.py and every module it
   calls into already log via the `logging` module) into the box as it
   happens.
2. The handler is removed once the call finishes, successfully or not, so
   repeated calls across Streamlit reruns never accumulate duplicate
   handlers on the root logger.

Known Caveats:
- Captures every record that propagates to the root logger, not just
  thermokp.py's own - acceptable here since the dashboard process runs no
  other unrelated logging.

Author: ThermoKP Team
License: MIT
"""

import logging
from typing import Any, Callable, List, TypeVar

import streamlit as st

_MAX_LOG_LINES = 200
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

T = TypeVar("T")


class _StatusLogHandler(logging.Handler):
    """Appends formatted log records into a Streamlit placeholder as they arrive."""

    def __init__(self, placeholder: Any) -> None:
        super().__init__()
        self.placeholder = placeholder
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))
        self.placeholder.code("\n".join(self.lines[-_MAX_LOG_LINES:]), language="log")


def run_with_log_feedback(label: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run `fn`, streaming its log output into an expandable status box.

    Parameters
    ----------
    label : str
        Status box title (e.g. "Predicting kinetics"). Streamlit auto-marks
        it complete when `fn` returns, or failed if it raises.
    fn : callable
        Callable to run; its return value is passed through unchanged.
    *args, **kwargs
        Forwarded to `fn`.

    Returns
    -------
    Any
        `fn`'s return value.

    Raises
    ------
    Exception
        Whatever `fn` raises, after marking the status box as failed - the
        caller is still responsible for handling it (e.g. thermokp.py's
        ThermoKPError).
    """
    root_logger = logging.getLogger()
    with st.status(label, expanded=True) as status:
        log_box = st.empty()
        handler = _StatusLogHandler(log_box)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        root_logger.addHandler(handler)
        try:
            result = fn(*args, **kwargs)
        except Exception:
            status.update(state="error")
            raise
        finally:
            root_logger.removeHandler(handler)
    return result
