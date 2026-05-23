from src.data_efficiency import VARIANTS, FRACTIONS


def test_fractions_present():
    assert 1.0 in FRACTIONS
    assert 0.1 in FRACTIONS
    assert all(0 < f <= 1.0 for f in FRACTIONS)


def test_variants_subset_of_shipped_configs():
    import glob
    shipped = {p.split("/")[-1].rstrip(".yaml") for p in glob.glob("configs/*.yaml")}
    assert set(VARIANTS) <= shipped
