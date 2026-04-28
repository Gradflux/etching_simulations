import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxps.geometry import (
    extract_surface_mesh_3d,
    extract_zero_contour_2d,
    extrude_2d_to_3d,
    make_grid_2d,
    make_grid_3d,
    sdf_circle,
    sdf_plane,
    sdf_sphere,
    slice_3d,
)
from jaxps.io import DomainSetup, SimulationState, load_simulation, save_simulation
from jaxps.materials import (
    MaterialValueMap,
    default_material_registry,
    deserialize_material_registry,
    get_material,
    is_builtin_material,
    list_builtin_materials,
    serialize_material_registry,
)
from jaxps.models import (
    IsotropicEtch,
    MaterialMaskedEtch,
    MultiParticleProcess,
    ParticleSpecies,
    PolynomialCosineYield,
    SimplifiedFluorocarbonEtch,
    SimplePlasmaProcess,
    combine_species_rates,
    fluorocarbon_velocity,
    incidence_cosine,
    polynomial_cosine_yield,
    simple_plasma_velocity,
    update_passivation_coverage,
)
from jaxps.rays import normalize_flux_backend, select_flux_backend
from jaxps.utils import (
    AdvectionParameters,
    CoverageParameters,
    ProcessParameters,
    RayTracingParameters,
    device_report,
)


def test_builtin_and_custom_material_registry_round_trip():
    registry = default_material_registry()

    assert is_builtin_material("Si")
    assert "SiO2" in {material.name for material in list_builtin_materials()}
    assert get_material("Vacuum", registry).name == "Void"

    custom = registry.register("LowK", category="dielectric", density=1.4, aliases=("LK",), metadata={"k": 2.7})
    low_k = custom.get("LK")

    assert low_k.id >= 1000
    assert low_k.metadata["k"] == 2.7

    restored = deserialize_material_registry(serialize_material_registry(custom))
    assert restored.get("LowK").id == low_k.id
    assert restored.get_by_id(low_k.id).name == "LowK"

    with pytest.raises(ValueError):
        custom.register("Si")


def test_material_value_map_scalar_and_vector_values_are_jax_compatible():
    registry = default_material_registry()
    material_ids = jnp.asarray([registry.get("Si").id, registry.get("SiO2").id, registry.get("Mask").id])

    rates = MaterialValueMap(default=0.1).set("Si", 1.0).set("Mask", 0.0)
    assert np.allclose(np.asarray(rates.evaluate(material_ids, registry)), [1.0, 0.1, 0.0])

    vectors = MaterialValueMap(default=(0.0, 0.0)).set("Si", (1.0, 2.0))
    evaluated = vectors.evaluate(material_ids, registry)

    assert evaluated.shape == (3, 2)
    assert np.allclose(np.asarray(evaluated[0]), [1.0, 2.0])
    assert np.allclose(np.asarray(evaluated[1]), [0.0, 0.0])


def test_domain_and_simulation_serialization_round_trip(tmp_path):
    domain = DomainSetup(
        dim=2,
        bounds=((-1.0, 1.0), (-2.0, 2.0)),
        shape=(11, 21),
        boundary_conditions=("open", "zero_gradient"),
        initial_geometry={"type": "plane"},
    )
    restored_domain = DomainSetup.from_dict(domain.to_dict())

    assert restored_domain.dim == 2
    assert restored_domain.boundary_conditions == ("open", "zero_gradient")
    assert np.allclose(restored_domain.spacing, (0.2, 0.2))

    registry = default_material_registry().register("CustomMask")
    grid = domain.make_grid()
    phi = sdf_plane(grid, point=(0.0, 0.0), normal=(1.0, 0.0))
    material_ids = jnp.full(grid.shape, registry.get("CustomMask").id)
    state = SimulationState(phi=phi, grid=grid, material_ids=material_ids, metadata={"case": "round_trip"})

    save_simulation(tmp_path / "case", state, registry)
    loaded_state, loaded_registry = load_simulation(tmp_path / "case")

    assert loaded_state.phi.shape == grid.shape
    assert loaded_state.metadata["case"] == "round_trip"
    assert loaded_registry.get("CustomMask").id == registry.get("CustomMask").id
    assert np.all(np.asarray(loaded_state.material_ids) == registry.get("CustomMask").id)

    with pytest.raises(ValueError):
        DomainSetup(dim=4, bounds=((0.0, 1.0),), shape=(5,))


def test_polynomial_cosine_yield_convention_and_jit():
    mu = jnp.asarray([0.0, 0.5, 1.0])
    coefficients = (1.0, 2.0, 3.0)

    expected = jnp.asarray([1.0, 2.75, 6.0])
    assert np.allclose(np.asarray(polynomial_cosine_yield(mu, coefficients)), np.asarray(expected))
    jitted_yield = jax.jit(polynomial_cosine_yield, static_argnames=("clamp",))
    assert np.allclose(np.asarray(jitted_yield(mu, jnp.asarray(coefficients))), np.asarray(expected))
    assert np.allclose(np.asarray(polynomial_cosine_yield(mu, (-1.0,), clamp=True)), 0.0)

    normals = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    directions = jnp.asarray([-1.0, 0.0])
    assert np.allclose(np.asarray(incidence_cosine(directions, normals)), [1.0, 1.0, 0.0])
    assert np.isclose(float(incidence_cosine(jnp.asarray([1.0, 0.0]), normals[:1])[0]), 0.0)
    assert np.isclose(float(PolynomialCosineYield((0.0, 1.0)).from_direction(directions, normals)[0]), 1.0)


def test_target_material_and_mask_material_velocities():
    registry = default_material_registry()
    material_ids = jnp.asarray([registry.get("Si").id, registry.get("SiO2").id, registry.get("Mask").id])
    phi = jnp.zeros_like(material_ids, dtype=jnp.float32)

    target_velocity = IsotropicEtch(0.2, target_materials=("Si",)).velocity(phi, material_ids=material_ids, registry=registry)
    assert np.allclose(np.asarray(target_velocity), [-0.2, 0.0, 0.0])

    masked_velocity = MaterialMaskedEtch(base_rate=0.2, mask_materials=("Mask",), mask_rate=0.05).velocity(
        material_ids,
        registry,
    )
    assert np.allclose(np.asarray(masked_velocity), [-0.2, -0.2, -0.05])


def test_mesh_extraction_extrusion_and_slicing():
    grid2d = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(51, 51))
    circle = sdf_circle(grid2d, center=(0.0, 0.0), radius=0.5)
    contour = extract_zero_contour_2d(circle, grid2d)

    assert contour.points.shape[0] > 20
    radii = np.linalg.norm(np.asarray(contour.points), axis=1)
    assert np.isclose(np.mean(radii), 0.5, atol=2.0 * grid2d.spacing[0])

    phi3d, grid3d = extrude_2d_to_3d(circle, grid2d, z_bounds=(-0.2, 0.2), nz=5)
    sliced, sliced_grid = slice_3d(phi3d, grid3d, axis=2, index_or_coordinate=0.0)

    assert phi3d.shape == (51, 51, 5)
    assert sliced_grid.shape == grid2d.shape
    assert np.allclose(np.asarray(sliced), np.asarray(circle))

    sphere_grid = make_grid_3d(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), shape=(21, 21, 21))
    sphere = sdf_sphere(sphere_grid, center=(0.0, 0.0, 0.0), radius=0.4)
    surface = extract_surface_mesh_3d(sphere, sphere_grid)

    assert surface.vertices.shape[0] > 20
    assert np.isclose(np.mean(np.linalg.norm(np.asarray(surface.vertices), axis=1)), 0.4, atol=0.15)


def test_flux_backend_selection_and_parameter_validation():
    selection = select_flux_backend("AUTO")
    assert selection.actual_backend in {"JAX_GRID", "JAX_RAYS"}
    assert selection.to_dict()["requested_backend"] == "AUTO"
    assert normalize_flux_backend("jax_grid") == "JAX_GRID"
    assert isinstance(device_report()["devices"], list)

    assert AdvectionParameters(cfl=0.25).to_dict()["cfl"] == 0.25
    assert RayTracingParameters(num_rays=8).to_dict()["num_rays"] == 8
    assert CoverageParameters(initial=0.1).to_dict()["initial"] == 0.1
    assert ProcessParameters(name="etch").to_dict()["name"] == "etch"

    with pytest.raises(ValueError):
        normalize_flux_backend("unknown")
    with pytest.raises(ValueError):
        AdvectionParameters(cfl=0.0)
    with pytest.raises(NotImplementedError):
        select_flux_backend("EXTERNAL_OPTIX")


def test_plasma_fluorocarbon_and_multi_particle_models():
    registry = default_material_registry()
    material_ids = jnp.asarray([registry.get("Si").id, registry.get("SiO2").id])
    normals = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])

    plasma_velocity = simple_plasma_velocity(
        ion_flux=jnp.asarray([1.0, 1.0]),
        neutral_flux=jnp.asarray([2.0, 0.0]),
        passivation_flux=jnp.asarray([1.0, 1.0]),
        ion_yield=0.5,
        neutral_coefficient=0.1,
        passivation_sticking=0.2,
    )
    assert np.allclose(np.asarray(plasma_velocity), [-0.5, -0.3])

    plasma = SimplePlasmaProcess(
        ion_yield=MaterialValueMap(default=0.0).set("Si", 1.0),
        neutral_coefficient=MaterialValueMap(default=0.0),
        passivation_sticking=MaterialValueMap(default=0.0),
    )
    assert np.allclose(np.asarray(plasma.velocity(material_ids, 1.0, 0.0, 0.0, registry)), [-1.0, 0.0])

    base = fluorocarbon_velocity(1.0, 1.0, 0.0, 1.0, 1.0, 0.5)
    passivated = fluorocarbon_velocity(1.0, 1.0, 2.0, 1.0, 1.0, 0.5)
    covered = fluorocarbon_velocity(1.0, 1.0, 0.0, 1.0, 1.0, 0.5, coverage=1.0)
    assert float(passivated) > float(base)
    assert float(covered) > float(base)

    fluorocarbon = SimplifiedFluorocarbonEtch(
        chemical_coefficients=MaterialValueMap(default=0.0).set("Si", 1.0),
        ion_yields=MaterialValueMap(default=0.0).set("Si", 0.5),
        polymer_sticking=MaterialValueMap(default=0.0),
    )
    assert np.allclose(np.asarray(fluorocarbon.velocity(material_ids, 1.0, 1.0, 0.0, registry=registry)), [-1.5, 0.0])
    assert np.allclose(np.asarray(update_passivation_coverage(jnp.asarray([0.5]), 1.0, 1.0, 0.1, 1.0, 1.0)), [0.55])

    process = MultiParticleProcess(
        species=(
            ParticleSpecies(name="ion", kind="etch", flux=1.0, coefficient=2.0),
            ParticleSpecies(name="polymer", kind="deposition", flux=0.5, coefficient=1.0),
        ),
        return_diagnostics=True,
    )
    total, diagnostics = process.velocity(normals)
    assert diagnostics.shape == (2, 2)
    assert np.allclose(np.asarray(total), [-1.5, -1.5])

    material_species = MultiParticleProcess(
        species=(
            ParticleSpecies(
                name="material-selective",
                kind="etch",
                flux=1.0,
                coefficient=1.0,
                material_response=MaterialValueMap(default=0.0).set("Si", 3.0),
            ),
        )
    )
    assert np.allclose(np.asarray(material_species.velocity(normals, material_ids, registry)), [-3.0, 0.0])
    assert np.allclose(
        np.asarray(jax.jit(combine_species_rates)(jnp.asarray([[-1.0, -2.0], [0.25, 0.5]]))),
        [-0.75, -1.5],
    )
