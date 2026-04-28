import pytest

from jaxps.materials import Material, MaterialRegistry


def test_material_creation_and_registry_lookup():
    silicon = Material(name="Si", id=1, properties={"density": 2.33})
    oxide = Material(name="SiO2", id=2)
    registry = MaterialRegistry().register(silicon).register(oxide)

    assert registry.get("Si") is silicon
    assert registry.get_by_id(2) is oxide
    assert silicon.properties["density"] == 2.33


def test_material_registry_rejects_duplicates():
    registry = MaterialRegistry().register(Material("Si", id=1))

    with pytest.raises(ValueError):
        registry.register(Material("Si", id=2))
    with pytest.raises(ValueError):
        registry.register(Material("Other", id=1))
    with pytest.raises(KeyError):
        registry.get("missing")
