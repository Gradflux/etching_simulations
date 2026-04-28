import importlib


def test_examples_import_without_running_main():
    for module_name in [
        "examples.isotropic_etch_2d",
        "examples.isotropic_deposition_2d",
        "examples.masked_trench_etch_2d",
        "examples.directional_etch_2d",
        "examples.directional_etch_3d",
        "examples.flux_deposition_2d",
        "examples.flux_deposition_3d",
    ]:
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")
