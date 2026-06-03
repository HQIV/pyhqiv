"""
pyhqiv: Python implementation of the Horizon-Quantized Informational Vacuum framework.

.. warning::
   Experimental status. All features are experimental. APIs and numerical results may change.
   Public contribution and feedback are greatly appreciated.

   The CMB pipeline in particular has known issues (analytic transfer vs full Boltzmann
   hierarchy, phenomenological map vs first-principles projection, peak positions/shape).
   See docs/HQIV_CMB_Pipeline.md for details.

If you use this package in research, please cite:
https://doi.org/10.5281/zenodo.18794889
"""

__citation__ = """
@misc{ettinger2026hqiv,
    title        = {Horizon-Quantized Informational Vacuum (HQIV) Framework},
    author       = {Steven Ettinger},
    year         = 2026,
    doi          = {10.5281/zenodo.18794889},
    url          = {https://doi.org/10.5281/zenodo.18794889}
}
"""

__doc__ = """
pyhqiv: Python implementation of the Horizon-Quantized Informational Vacuum framework.

If you use this package in research, please cite:
https://doi.org/10.5281/zenodo.18794889
"""

from pyhqiv import defects, molecular, semiconductors, waveguide
from pyhqiv.algebra import OctonionHQIVAlgebra
from pyhqiv.energy_field import (
    HQIVEnergyField,
    confined_energy_from_composite,
    effective_horizon_from_energy_mev,
    merge_constituents,
    total_energy_from_state_matrix,
)
from pyhqiv.ase_interface import (
    HQIVCalculator,
    hqiv_energy_at_positions,
    hqiv_forces_analytic,
    hqiv_stress_virial,
)
from pyhqiv.atom import HQIVAtom
from pyhqiv.bulk_seed import BULK_SEED_AVAILABLE, get_bulk_seed
from pyhqiv.cmb_pipeline import HQIVCMBPipeline, cmb_pipeline_status
from pyhqiv.constants import (
    A_LOC_ANG,
    AGE_APPARENT_GYR_PAPER,
    AGE_WALL_GYR_PAPER,
    ALPHA,
    COMBINATORIAL_INVARIANT,
    GAMMA,
    HBAR_C_EV_ANG,
    LAPSE_COMPRESSION_PAPER,
    M_TRANS,
    OMEGA_TRUE_K_PAPER,
    T_CMB_K,
    T_LOCK_GEV,
    T_LOCK_NOW_GEV,
    T_PL_GEV,
)
from pyhqiv.cosmology import HQIVCosmology, HQIVUniverseEvolver
from pyhqiv.crystal import HQIVCrystal, high_symmetry_k_path, hqiv_potential_shift
from pyhqiv.defects import charged_defect_supercell, formation_energy
from pyhqiv.export import (
    export_charge_density_ovito,
    export_charge_density_vesta,
    pyscf_hqiv_shift,
)
from pyhqiv.fields import PhaseHorizonFDTD
from pyhqiv.fluid import eddy_viscosity, f_inertia, g_vac_vector, modified_momentum_rhs
from pyhqiv.lattice import DiscreteNullLattice, omega_k_from_distance
from pyhqiv.orbit import HQIVOrbit, parker_perihelion_lapse
from pyhqiv.perturbations import HQIVPerturbations, PerturbationMode
from pyhqiv.phase import HQIVPhaseLift, default_phase_lift
from pyhqiv.protocols import (
    NullLatticeBase,
    NullLatticeProtocol,
    PhaseLiftBase,
    PhaseLiftProtocol,
)
from pyhqiv.redshift import HQIVRedshift, z_expansion_from_scale_factor, z_total_apparent
from pyhqiv.utils import theta_ref_ang_from_curvature
from pyhqiv.polarization import RedshiftDecomposition, decompose_redshift
from pyhqiv.hqiv_scalings import get_hqiv_nuclear_constants
from pyhqiv.horizon_network import HorizonNetwork, relax_nucleon_positions, relax_quark_positions

# LEAN subatomic module: algebraic structure and effective mode counts only.
from pyhqiv.subatomic import (
    _sphere_touching_mu,
    composite_invariants,
    composite_state_matrix,
    confined_effective_modes_for_flavor,
    effective_modes_from_composite,
    nucleon_effective_modes,
    quark_flavors_from_flavor_content,
    quark_flavors_for_nucleon,
    quark_state_matrix,
    quark_state_matrices_for_flavor,
)

# Legacy subatomic layer retains PDG/QCD-based horizons and energies for the
# public API that depends on MeV/fm units. New code should prefer the LEAN
# algebraic helpers above, but we re-export the legacy numerics here so that
# existing users and tests keep working.
from pyhqiv import subatomic_legacy as _subatomic_legacy

color_singlet_projector = _subatomic_legacy.color_singlet_projector
make_proton_from_quark_states = _subatomic_legacy.make_proton_from_quark_states
SUBATOMIC_PDG_MEV = _subatomic_legacy.SUBATOMIC_PDG_MEV
confined_effective_theta_m = _subatomic_legacy.confined_effective_theta_m
confined_energy_mev = _subatomic_legacy.confined_energy_mev
confined_pdg_energy_mev = _subatomic_legacy.confined_pdg_energy_mev
nucleon_charge_unwrapped_folded_measures = _subatomic_legacy.nucleon_charge_unwrapped_folded_measures
nucleon_energy_mev = _subatomic_legacy.nucleon_energy_mev
nucleon_effective_theta_m_for_flavor = _subatomic_legacy.nucleon_effective_theta_m_for_flavor
proton_energy_mev = _subatomic_legacy.proton_energy_mev
neutron_energy_mev = _subatomic_legacy.neutron_energy_mev
nucleon_energies_mev = _subatomic_legacy.nucleon_energies_mev
proton_effective_theta_m = _subatomic_legacy.proton_effective_theta_m
neutron_effective_theta_m = _subatomic_legacy.neutron_effective_theta_m
nucleon_effective_theta_m = _subatomic_legacy.nucleon_effective_theta_m
t_qcd_gev_at_epoch = _subatomic_legacy.t_qcd_gev_at_epoch
quark_binding_angles = _subatomic_legacy.quark_binding_angles
from pyhqiv.nuclear import (
    ELEMENT_SYMBOL_TO_Z,
    ELEMENT_Z_TO_SYMBOL,
    Nuclide,
    NuclearConfig,
    nuclide_from_symbol,
    binding_energy_isotope,
    half_life_nuclide_hqiv,
    decay_chain_nuclide_hqiv,
)
from pyhqiv.response import compute_conductivity, response_tensor_diagonal
from pyhqiv.semiconductors import (
    compute_band_gap,
    compute_conductivity_tensor,
    dielectric_function_epsilon,
    dos,
    effective_mass,
)
from pyhqiv.solar_core import HQIVSolarCore, phi_solar_radial_profile
from pyhqiv.system import HQIVSystem
from pyhqiv.thermo import (
    TESTABLE_PREDICTIONS,
    HQIVEquationOfState,
    HQIVHydrogen,
    HQIVIdealGas,
    HQIVRealGas,
    HQIVThermoSystem,
    PhaseDiagramGenerator,
    compute_free_energy,
    hqiv_answer_thermo,
    lapse_compression_thermo,
    phi_from_rho_T,
    plot_phase_diagram_standard_vs_hqiv,
    shell_fraction_energy_shift,
    thermo_ase_phase_stability,
    thermo_crystal_phi,
    thermo_fluid_lapse,
    theta_local_from_density,
)

__all__ = [
    "A_LOC_ANG",
    "ALPHA",
    "GAMMA",
    "HBAR_C_EV_ANG",
    "molecular",
    "T_PL_GEV",
    "T_LOCK_GEV",
    "T_LOCK_NOW_GEV",
    "T_CMB_K",
    "M_TRANS",
    "COMBINATORIAL_INVARIANT",
    "OMEGA_TRUE_K_PAPER",
    "LAPSE_COMPRESSION_PAPER",
    "AGE_WALL_GYR_PAPER",
    "AGE_APPARENT_GYR_PAPER",
    "OctonionHQIVAlgebra",
    "DiscreteNullLattice",
    "omega_k_from_distance",
    "HQIVCosmology",
    "HQIVUniverseEvolver",
    "get_bulk_seed",
    "BULK_SEED_AVAILABLE",
    "HQIVPhaseLift",
    "default_phase_lift",
    "HQIVAtom",
    "HQIVSystem",
    "PhaseHorizonFDTD",
    "f_inertia",
    "g_vac_vector",
    "eddy_viscosity",
    "modified_momentum_rhs",
    "waveguide",
    "HQIVCrystal",
    "hqiv_potential_shift",
    "high_symmetry_k_path",
    "semiconductors",
    "compute_band_gap",
    "dos",
    "effective_mass",
    "compute_conductivity_tensor",
    "dielectric_function_epsilon",
    "defects",
    "formation_energy",
    "charged_defect_supercell",
    "export_charge_density_vesta",
    "export_charge_density_ovito",
    "pyscf_hqiv_shift",
    "compute_conductivity",
    "response_tensor_diagonal",
    "HQIVCalculator",
    "hqiv_energy_at_positions",
    "hqiv_forces_analytic",
    "hqiv_stress_virial",
    "NullLatticeProtocol",
    "NullLatticeBase",
    "PhaseLiftProtocol",
    "PhaseLiftBase",
    "HQIVSolarCore",
    "phi_solar_radial_profile",
    "HQIVRedshift",
    "z_total_apparent",
    "z_expansion_from_scale_factor",
    "RedshiftDecomposition",
    "decompose_redshift",
    "get_hqiv_nuclear_constants",
    "HorizonNetwork",
    "relax_nucleon_positions",
    "relax_quark_positions",
    "HQIVEnergyField",
    "effective_horizon_from_energy_mev",
    "merge_constituents",
    "total_energy_from_state_matrix",
    "confined_energy_from_composite",
    "color_singlet_projector",
    "make_proton_from_quark_states",
    "SUBATOMIC_PDG_MEV",
    "confined_effective_theta_m",
    "confined_energy_mev",
    "confined_pdg_energy_mev",
    "nucleon_charge_unwrapped_folded_measures",
    "nucleon_energy_mev",
    "nucleon_effective_theta_m_for_flavor",
    "proton_energy_mev",
    "neutron_energy_mev",
    "nucleon_energies_mev",
    "quark_flavors_from_flavor_content",
    "quark_state_matrices_for_flavor",
    "proton_effective_theta_m",
    "neutron_effective_theta_m",
    "nucleon_effective_theta_m",
    "t_qcd_gev_at_epoch",
    "quark_binding_angles",
    "quark_state_matrix",
    "Nuclide",
    "NuclearConfig",
    "ELEMENT_SYMBOL_TO_Z",
    "ELEMENT_Z_TO_SYMBOL",
    "nuclide_from_symbol",
    "binding_energy_isotope",
    "half_life_nuclide_hqiv",
    "decay_chain_nuclide_hqiv",
    "HQIVOrbit",
    "parker_perihelion_lapse",
    "HQIVPerturbations",
    "PerturbationMode",
    "HQIVCMBPipeline",
    "cmb_pipeline_status",
    "HQIVThermoSystem",
    "HQIVEquationOfState",
    "HQIVIdealGas",
    "HQIVRealGas",
    "HQIVHydrogen",
    "PhaseDiagramGenerator",
    "compute_free_energy",
    "hqiv_answer_thermo",
    "phi_from_rho_T",
    "theta_local_from_density",
    "theta_ref_ang_from_curvature",
    "shell_fraction_energy_shift",
    "lapse_compression_thermo",
    "thermo_fluid_lapse",
    "thermo_crystal_phi",
    "thermo_ase_phase_stability",
    "TESTABLE_PREDICTIONS",
    "plot_phase_diagram_standard_vs_hqiv",
]

try:
    from pyhqiv._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
