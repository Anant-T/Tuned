"""The divergence guard's decision logic (pure, no ML deps).

Design constraints it encodes (research-verified 2026-08-08):
- keyed on grad_norm, NOT loss: logging_nan_inf_filter defaults True and
  rewrites nan losses in logs, so loss never shows divergence
- DDP GradScaler calibration logs grad_norm=nan on steps 1-2 of a HEALTHY
  run (observed in both green gates) - a grace window must cover it
- a single isolated non-finite later is normal GradScaler backoff; only a
  consecutive streak means divergence
"""

from tuned.train.sft import _NonFiniteWindow, clip_binding_rate


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


# P1.6: clip_binding_rate is the pure half of the max_grad_norm=0.3 binding
# instrument (the impure half lives in _NonFiniteGuard.on_log, which is
# nested inside main() and untestable without the GPU stack - see
# test_sft_args.py for the source-inspection tests of that wiring). This
# module-level import with no torch installed in this environment is itself
# the torch-free-importability proof, same as _NonFiniteWindow above.


def test_clip_binding_rate_counts_norms_at_or_above_the_limit():
    # >= , not >: a norm sitting exactly ON the limit is still "binding" by
    # the spec this instrument implements, so excluding the boundary value
    # would undercount.
    assert clip_binding_rate([0.1, 0.2, 0.3, 0.4], 0.3) == 0.5


def test_clip_binding_rate_empty_input_is_zero_not_a_raise():
    # No steps observed and no steps bound are different questions, but a
    # pure function with no caller context can't tell them apart - and must
    # not raise on an empty window just because the caller hasn't gated it.
    assert clip_binding_rate([], 0.3) == 0.0


def test_clip_binding_rate_excludes_non_finite_from_the_count_but_not_the_denominator():
    # A non-finite grad_norm is _NonFiniteWindow/_NonFiniteGuard's failure
    # mode (fp16 divergence), not evidence the clip bound - so it must not
    # count as binding. But it also must not silently shrink n: if it did,
    # the printed rate would look like it was computed over cleaner data than
    # it actually was. Keeping it in the denominator is the conservative
    # choice - it can only pull the reported rate DOWN, never inflate it.
    assert clip_binding_rate([float("nan"), 0.5, 0.5], 0.3) == 2 / 3
    assert clip_binding_rate([float("inf"), float("-inf")], 0.3) == 0.0


def test_clip_binding_rate_unparseable_entries_get_the_same_treatment():
    assert clip_binding_rate([None, "n/a", 0.5], 0.3) == 1 / 3


def test_clip_binding_rate_all_at_or_above_the_limit_is_one():
    # The scenario _NonFiniteGuard WARNs on: a run whose grad_norm never
    # drops below max_grad_norm is training on a fully clipped schedule.
    assert clip_binding_rate([0.3, 0.4, 0.5, 0.9], 0.3) == 1.0


def test_clip_binding_rate_well_under_the_limit_is_low():
    # The healthy baseline this instrument exists to distinguish from the
    # case above: build_sft_config's own comment records 0.08-0.19 as the
    # smoke-lane band max_grad_norm=0.3 was fitted against, well clear of it.
    assert clip_binding_rate([0.08, 0.1, 0.12, 0.19], 0.3) == 0.0


def test_the_clip_report_is_one_line_when_the_clip_is_not_binding():
    from tuned.train.sft import clip_report

    lines = clip_report([0.08, 0.1, 0.12, 0.19], 0.3)
    assert lines == ["clip_binding_rate=0.000 max_grad_norm=0.3 n=4"]


def test_the_clip_report_warns_when_the_schedule_has_become_a_clip():
    """The reading this instrument exists to produce. A run whose grad_norm
    never drops below max_grad_norm is not following the configured LR curve -
    it is following a normalised-gradient schedule - and the number alone,
    printed once among thousands of log lines, would not say so."""
    from tuned.train.sft import CLIP_BINDING_WARN, clip_report

    lines = clip_report([0.3, 0.4, 0.5, 0.9], 0.3)
    assert len(lines) == 2
    assert lines[0].startswith("clip_binding_rate=1.000 ")
    assert lines[1].startswith("WARN ")
    assert "normalised-gradient schedule" in lines[1]
    # The boundary itself does NOT warn - the threshold is "above", so a run
    # sitting exactly on it reads as the last healthy value rather than the
    # first unhealthy one.
    assert len(clip_report([1.0] * 3 + [0.0] * 7, 1.0)) == 1
    assert clip_binding_rate([1.0] * 3 + [0.0] * 7, 1.0) == CLIP_BINDING_WARN
