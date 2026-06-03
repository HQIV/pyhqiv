from pyhqiv.auxiliary_field import phi_of_shell
from pyhqiv.lean_witnesses import load_lean_witnesses
from pyhqiv.now_setters import active_slice, m_now, nowSetter, now_geometry, now_set_from_electron_horizon


def test_now_defaults_to_lean_reference_shell() -> None:
    w = load_lean_witnesses()
    assert m_now == w.get_int("m_now_electron_shell")


def test_now_set_from_electron_horizon_updates_geometry() -> None:
    geom = now_set_from_electron_horizon(4)
    assert geom.m_now == 4
    assert geom.phi_now == phi_of_shell(4)
    assert now_geometry().m_now == 4


def test_now_setter_selects_range_slice() -> None:
    selected = nowSetter(
        4.2,
        slices=[
            {"id": 3, "lower": 3.0, "upper": 4.0},
            {"id": 4, "lower": 4.0, "upper": 5.0, "reference_m": 9},
        ],
    )
    assert int(selected["id"]) == 4
    assert int(active_slice()["reference_m"]) == 9

