def run_gui(*args, **kwargs):
    # Lazy import avoids runpy double-import warning for `python -m gui.app`.
    from gui.app import run_gui as _run_gui

    return _run_gui(*args, **kwargs)


__all__ = ["run_gui"]
