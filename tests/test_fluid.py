"""Tests for HQIV modified fluid closure (F1 on FLUID_OMAXWELL_ROADMAP).

Run: ``PYTHONPATH=src python3 -m pytest tests/test_fluid.py`` (pytest) or
``PYTHONPATH=src python3 -m unittest tests.test_fluid`` (stdlib).
"""

from __future__ import annotations

import unittest

import numpy as np

from pyhqiv.fluid import (
    PlasmaFluidClosureHypothesis,
    eddy_viscosity,
    f_inertia,
    g_vac_vector,
    modified_momentum_rhs,
)
from pyhqiv.metric import gamma_hqiv


class TestFluid(unittest.TestCase):
    def test_gamma_hqiv_matches_fluid_defaults(self) -> None:
        g = gamma_hqiv()
        self.assertAlmostEqual(g, 2.0 / 5.0)
        # term = φ ∇δ̇θ′ + δ̇θ′ ∇φ; pick nonzero δ̇θ′ and ∇φ so first component is 1
        gv = g_vac_vector(
            phi=1.0,
            dot_delta_theta=1.0,
            grad_phi=np.array([1.0, 0.0, 0.0]),
            grad_dot_delta_theta=np.zeros(3),
        )
        self.assertAlmostEqual(float(gv[0]), -g / 6.0)

    def test_f_inertia_phi_zero_is_one(self) -> None:
        f = f_inertia(10.0, 0.0)
        self.assertAlmostEqual(float(f), 1.0)

    def test_f_inertia_laminar_large_a(self) -> None:
        phi = 1.0
        a_large = 1e6 * phi
        f = f_inertia(a_large, phi)
        self.assertAlmostEqual(float(f), 1.0, delta=1e-5)

    def test_f_inertia_bounded(self) -> None:
        f = f_inertia(0.001, 1000.0, f_min=0.01)
        self.assertGreaterEqual(float(f), 0.01)
        self.assertLessEqual(float(f), 1.0)

    def test_g_vac_zero_when_grads_zero(self) -> None:
        g = g_vac_vector(
            phi=3.0,
            dot_delta_theta=0.5,
            grad_phi=np.zeros(3),
            grad_dot_delta_theta=np.zeros(3),
        )
        np.testing.assert_allclose(g, 0.0)

    def test_eddy_viscosity_positive(self) -> None:
        nu = eddy_viscosity(1.0, 0.1, 2.0, coherence_factor=1.0)
        self.assertGreater(nu, 0)
        self.assertAlmostEqual(nu, gamma_hqiv() * 1.0 * 0.1 * 4.0)

    def test_modified_momentum_rhs_shape(self) -> None:
        gp = np.array([1.0, 2.0, 3.0])
        rhs = modified_momentum_rhs(gp, gp * 0, gp * 0, gp * 0, rho=1.0)
        np.testing.assert_allclose(rhs, -gp)

    def test_plasma_fluid_closure_hypothesis_holds(self) -> None:
        g = gamma_hqiv()
        theta, dot, lc, c = 1.0, 0.2, 2.0, 0.8
        nu_e = g * theta * abs(dot) * (lc**2) * c
        nu_m = 0.01
        h = PlasmaFluidClosureHypothesis(
            nu_mol=nu_m,
            nu_eddy=nu_e,
            nu_total=nu_m + nu_e,
            theta_local=theta,
            dot_delta_theta=dot,
            l_coh=lc,
            coherence=c,
        )
        self.assertTrue(h.holds())

    def test_plasma_fluid_closure_hypothesis_fails_bad_coherence(self) -> None:
        gamma_hqiv()
        h = PlasmaFluidClosureHypothesis(
            nu_mol=0.0,
            nu_eddy=1.0,
            nu_total=1.0,
            theta_local=1.0,
            dot_delta_theta=1.0,
            l_coh=1.0,
            coherence=1.5,
        )
        self.assertFalse(h.holds())


if __name__ == "__main__":
    unittest.main()
