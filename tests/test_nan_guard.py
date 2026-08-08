"""The divergence guard's decision logic (pure, no ML deps).

Design constraints it encodes (research-verified 2026-08-08):
- keyed on grad_norm, NOT loss: logging_nan_inf_filter defaults True and
  rewrites nan losses in logs, so loss never shows divergence
- DDP GradScaler calibration logs grad_norm=nan on steps 1-2 of a HEALTHY
  run (observed in both green gates) - a grace window must cover it
- a single isolated non-finite later is normal GradScaler backoff; only a
  consecutive streak means divergence
"""

from tuned.train.sft import _NonFiniteWindow


def test_grace_window_documents_that_the_step_is_absolute():
    # observe() keys on state.global_step, so a run resumed at step 61 gets NO
    # fresh grace window - which is correct, a restored GradScaler must not
    # recalibrate. A refactor to a step-relative counter would silently open a
    # 2-step blind spot on every resume, so the reason has to be written down.
    doc = _NonFiniteWindow.__doc__
    assert "global_step" in doc
    assert "resumed" in doc


def test_calibration_nans_within_grace_do_not_abort():
    w = _NonFiniteWindow(grace_steps=2, window=3)
    assert w.observe(1, "nan") is False
    assert w.observe(2, "nan") is False


def test_streak_after_grace_aborts_and_needs_full_window():
    w = _NonFiniteWindow(grace_steps=2, window=3)
    w.observe(1, "nan")
    w.observe(2, "nan")
    assert w.observe(3, "nan") is False
    assert w.observe(4, "nan") is False
    assert w.observe(5, "nan") is True


def test_finite_value_resets_the_streak():
    w = _NonFiniteWindow(grace_steps=2, window=3)
    assert w.observe(3, "nan") is False
    assert w.observe(4, 0.31) is False
    assert w.observe(5, "nan") is False
    assert w.observe(6, "nan") is False
    assert w.observe(7, "nan") is True


def test_accepts_floats_and_strings_like_trainer_logs():
    w = _NonFiniteWindow(grace_steps=0, window=2)
    assert w.observe(1, float("inf")) is False
    assert w.observe(2, "nan") is True


def test_unparseable_values_neither_advance_nor_reset():
    w = _NonFiniteWindow(grace_steps=0, window=2)
    assert w.observe(1, "nan") is False
    assert w.observe(2, None) is False   # junk: streak neither advanced...
    assert w.observe(3, "n/a") is False  # ...nor reset
    assert w.observe(4, "nan") is True   # second real nan completes the window
