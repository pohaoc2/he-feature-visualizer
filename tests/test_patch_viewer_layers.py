from pathlib import Path


def test_server_patches_allows_new_layers():
    from server_patches import ALLOWED_LAYERS  # noqa: WPS433

    expected = {
        "he",
        "mux_rgb",
        "overlay_cells",
        "cell_mask",
        "cellvit_mask",
        "cell_state",
        "vasculature",
        "immune",
        "oxygen",
        "glucose",
    }
    assert expected.issubset(ALLOWED_LAYERS)


def test_viewer_html_has_new_layer_elements():
    html = Path("viewer_patches.html").read_text()
    # controls
    assert 'id="mux_rgb"' in html
    assert 'id="cellvit_mask"' in html
    assert 'id="cell_state"' in html
    assert 'id="oxygen"' in html
    assert 'id="glucose"' in html
    # stacked images
    assert 'id="img-mux_rgb"' in html
    assert 'id="img-cellvit_mask"' in html
    assert 'id="img-cell_state"' in html
    assert 'id="img-oxygen"' in html
    assert 'id="img-glucose"' in html

