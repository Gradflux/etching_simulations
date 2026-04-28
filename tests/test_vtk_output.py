import numpy as np

from jaxps.io import write_vtk_structured_grid


def test_write_vtk_structured_grid_writes_legacy_file(tmp_path):
    path = tmp_path / "field.vtk"
    phi = np.zeros((4, 3), dtype=np.float32)

    write_vtk_structured_grid(path, phi, spacing=(0.1, 0.2))

    text = path.read_text()
    assert "DATASET STRUCTURED_POINTS" in text
    assert "DIMENSIONS 4 3 1" in text
    assert "SCALARS phi float 1" in text
