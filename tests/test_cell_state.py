import pandas as pd


def test_assign_cell_state_proliferation_and_emt():
    df = pd.DataFrame(
        {
            "Ki67": [100.0, 1.0, 1.0],
            "PCNA": [1.0, 1.0, 1.0],
            "Vimentin": [1.0, 100.0, 1.0],
            "Ecadherin": [10.0, 1.0, 10.0],
        }
    )

    from cell_state import compute_marker_thresholds, assign_cell_state  # noqa: WPS433

    thr = compute_marker_thresholds(df, markers=["Ki67", "PCNA", "Vimentin", "Ecadherin"], pct=90)
    states = assign_cell_state(df, thr)

    # row0: Ki67 high -> proliferating
    assert states.iloc[0] == "proliferating"
    # row1: Vimentin high + Ecadherin low -> emt
    assert states.iloc[1] == "emt"
    # row2: neither -> other
    assert states.iloc[2] == "other"

