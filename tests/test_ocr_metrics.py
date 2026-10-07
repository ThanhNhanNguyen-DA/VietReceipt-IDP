import unicodedata

import pytest

from src.evaluation.bootstrap import paired_ratio_diff_ci, ratio_ci
from src.ocr.metrics import Edit, char_edit, rate, score_pair, strip_diacritics, word_edit


def test_nfc_makes_two_encodings_equal():
    nfd = unicodedata.normalize("NFD", "Tổng tiền thanh toán")
    assert nfd != "Tổng tiền thanh toán"
    assert char_edit("Tổng tiền thanh toán", nfd).errors == 0


def test_cer_and_wer_basic():
    assert char_edit("abcd", "abxd") == Edit(1, 4)
    assert word_edit("tong tien 100", "tong tien 1000") == Edit(1, 3)
    assert char_edit("", "abc") == Edit(3, 0)           # tham chiếu rỗng: chỉ cộng lỗi, không chia cho 0


def test_variants_separate_case_and_diacritics():
    p = score_pair("Tổng Tiền", "tong tien")
    assert p["char_err_exact"] > 0 and p["char_err_nocase"] > 0 and p["char_err_nodiac"] == 0
    assert strip_diacritics("Đường Lê Lợi") == "Duong Le Loi"


def test_corpus_rate_is_weighted_not_mean_of_rates():
    edits = [Edit(1, 1), Edit(0, 99)]
    assert rate(edits) == pytest.approx(0.01)          # trung bình từng mẫu sẽ là 0.5


def test_whitespace_is_normalised():
    assert char_edit("a  b\n c", "a b c").errors == 0


def test_bootstrap_resamples_groups_and_is_reproducible():
    groups = ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5
    err = [0] * 15 + [5] * 5
    ln = [10] * 20
    r1, r2 = ratio_ci(groups, err, ln, seed=1), ratio_ci(groups, err, ln, seed=1)
    assert r1 == r2 and r1["n_groups"] == 4
    assert r1["value"] == pytest.approx(25 / 200)
    assert r1["low"] <= r1["value"] <= r1["high"] and r1["high"] > r1["low"]


def test_paired_diff_detects_consistent_improvement():
    groups = [f"g{i}" for i in range(30) for _ in range(3)]
    den = [10] * 90
    worse, better = [2] * 90, [1] * 90
    d = paired_ratio_diff_ci(groups, worse, den, better, den)
    assert d["diff"] == pytest.approx(0.1) and d["low"] > 0 and d["p_a_better"] == 0.0


@pytest.mark.parametrize("groups,num,den", [([], [], []), (["a"], [1], [0]), (["a"], [1, 2], [10])])
def test_bootstrap_rejects_undefined_inputs(groups, num, den):
    with pytest.raises(ValueError):
        ratio_ci(groups, num, den)
