from tuned import where_am_i


def test_detects_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    info = where_am_i()
    assert info["platform"] == "kaggle (Interactive)"


def test_detects_non_kaggle(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    info = where_am_i()
    assert info["platform"] == "local/other"
