from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from test_metal.core.models import OLSResult


class ColumnName(str, Enum):
    C = "C"
    MN = "Mn"
    SI = "Si"
    S = "S"
    P = "P"
    CR = "Cr"
    NI = "Ni"
    CU = "Cu"
    AL = "Al"
    N = "N"
    TI = "Ti"
    ZR = "Zr"
    V = "V"
    MO = "Mo"
    NB = "Nb"
    SN = "Sn"
    AS = "As"
    CA = "Ca"


def find_models_for_element(results: list[OLSResult], element: ColumnName) -> list[OLSResult]:
    x_name = f"steel_{element.value}_before"
    y_name = f"steel_{element.value}_after"
    return [r for r in results if r.x_col == x_name and r.y_col == y_name]


def build_optimization_targets(
    results: list[OLSResult], df: pd.DataFrame
) -> dict[str, tuple[str, float]]:
    targets: dict[str, tuple[str, float]] = {}
    for label, element, _cls in OPTIMIZATION_ELEMENTS:
        element_models = find_models_for_element(results, element)
        if not element_models:
            continue
        y_col = f"steel_{element.value}_after"
        if y_col not in df.columns:
            continue
        model = element_models[0]
        x_min = float(model.x.min())
        x_max = float(model.x.max())
        achievable_min = min(
            model.intercept + model.slope * x_min,
            model.intercept + model.slope * x_max,
        )
        targets[label] = (model.x_col, float(achievable_min))
    return targets


TARGET_AFTER = "steel_S_after"
TARGET_BEFORE = "steel_S_before"

# Domain class of each element, used to choose the per-element change-metric
# sign convention. "reduction" elements are impurities the refining step is
# intended to lower (positive metric = more removed = better). "additive"
# elements are intentionally added during the heat (positive metric = more
# added = worse, by name). A third value "neutral" would use the raw signed
# difference with no value judgment; not currently used.
ElementClass = str  # "reduction" | "additive" | "neutral"

# Elements driven through inverse regression to produce the Pareto front.
# Each entry pairs a human-readable label with the matching ColumnName enum
# value and the element's domain class so the before/after columns and the
# per-element metric sign convention derive from a single source of truth.
OPTIMIZATION_ELEMENTS: list[tuple[str, ColumnName, ElementClass]] = [
    ("Sulfur (S)", ColumnName.S, "reduction"),
    ("Silicon (Si)", ColumnName.SI, "additive"),
]


def optimization_model_columns() -> list[tuple[str, str, str, ElementClass]]:
    """Return ``(element_label, before_x_col, after_y_col, element_class)`` tuples."""
    return [
        (label, f"steel_{element.value}_before", f"steel_{element.value}_after", cls)
        for label, element, cls in OPTIMIZATION_ELEMENTS
    ]


COLUMN_NAMES: list[str] = [
    "sample_number",
    "steel_C_before",
    "steel_Mn_before",
    "steel_Si_before",
    "steel_S_before",
    "steel_P_before",
    "steel_Cr_before",
    "steel_Ni_before",
    "steel_Cu_before",
    "steel_Al_before",
    "steel_N_before",
    "steel_Ti_before",
    "steel_Zr_before",
    "steel_V_before",
    "steel_Mo_before",
    "steel_Nb_before",
    "steel_Sn_before",
    "steel_As_before",
    "steel_Ca_before",
    "slag_CaO_before",
    "slag_SiO2_before",
    "slag_MgO_before",
    "slag_FeO_before",
    "slag_Al2O3_before",
    "slag_P2O5_before",
    "slag_Fe2O3_before",
    "slag_MnO_before",
    "slag_S_before",
    "slag_basicity_before",
    "add_FeCa_before",
    "add_FeSi65_before",
    "add_FeMn_before",
    "add_FeSiMn_before",
    "add_lime_before",
    "add_smelting_before",
    "add_AlP_before",
    "add_SiCa_before",
    "add_Al12_before",
    "add_Al13_before",
    "add_FeB_before",
    "add_feldspar_before",
    "add_FeTi_before",
    "add_AlGr_before",
    "sulfur_reduction_ratio",
    "sample_number_after",
    "steel_C_after",
    "steel_Mn_after",
    "steel_Si_after",
    "steel_S_after",
    "steel_P_after",
    "steel_Cr_after",
    "steel_Ni_after",
    "steel_Cu_after",
    "steel_Al_after",
    "steel_N_after",
    "steel_Ti_after",
    "steel_Zr_after",
    "steel_V_after",
    "steel_Mo_after",
    "steel_Nb_after",
    "steel_Sn_after",
    "steel_As_after",
    "steel_Ca_after",
    "slag_CaO_after",
    "slag_SiO2_after",
    "slag_MgO_after",
    "slag_FeO_after",
    "slag_Al2O3_after",
    "slag_P2O5_after",
    "slag_Fe2O3_after",
    "slag_MnO_after",
    "slag_S_after",
    "slag_basicity_after",
    "total_weight",
    "heat_number",
    "upk_number",
    "processing_time",
    "bottom_blowing_time",
    "argon_flow_total",
    "blow_flow1",
    "blow_flow2",
    "slag_height_incoming",
    "slag_height_outgoing",
    "slag_color_incoming",
    "slag_color_outgoing",
    "metal_temp_first",
    "metal_temp_last",
    "heat_temp_first",
    "heat_temp_last",
    "heating_duration",
    "energy_consumption",
]

PREDICTORS_AFTER: list[str] = [
    "steel_C_after",
    "steel_Mn_after",
    "steel_Si_after",
    "steel_P_after",
    "steel_Cr_after",
    "steel_Ni_after",
    "steel_Cu_after",
    "steel_Al_after",
    "steel_N_after",
    "steel_Ti_after",
    "steel_Zr_after",
    "steel_V_after",
    "steel_Mo_after",
    "steel_Nb_after",
    "steel_Sn_after",
    "steel_As_after",
    "steel_Ca_after",
    "slag_CaO_after",
    "slag_SiO2_after",
    "slag_MgO_after",
    "slag_FeO_after",
    "slag_Al2O3_after",
    "slag_P2O5_after",
    "slag_Fe2O3_after",
    "slag_MnO_after",
    "slag_S_after",
    "slag_basicity_after",
    "add_FeCa_before",
    "add_FeSi65_before",
    "add_FeMn_before",
    "add_FeSiMn_before",
    "add_lime_before",
    "add_smelting_before",
    "add_AlP_before",
    "add_SiCa_before",
    "add_Al12_before",
    "add_Al13_before",
    "add_FeB_before",
    "add_feldspar_before",
    "add_FeTi_before",
    "add_AlGr_before",
    "sulfur_reduction_ratio",
]

PREDICTORS_BEFORE: list[str] = [
    "steel_C_before",
    "steel_Mn_before",
    "steel_Si_before",
    "steel_P_before",
    "steel_Cr_before",
    "steel_Ni_before",
    "steel_Cu_before",
    "steel_Al_before",
    "steel_N_before",
    "steel_Ti_before",
    "steel_Zr_before",
    "steel_V_before",
    "steel_Mo_before",
    "steel_Nb_before",
    "steel_Sn_before",
    "steel_As_before",
    "steel_Ca_before",
    "slag_CaO_before",
    "slag_SiO2_before",
    "slag_MgO_before",
    "slag_FeO_before",
    "slag_Al2O3_before",
    "slag_P2O5_before",
    "slag_Fe2O3_before",
    "slag_MnO_before",
    "slag_S_before",
    "slag_basicity_before",
    "add_FeCa_before",
    "add_FeSi65_before",
    "add_FeMn_before",
    "add_FeSiMn_before",
    "add_lime_before",
    "add_smelting_before",
    "add_AlP_before",
    "add_SiCa_before",
    "add_Al12_before",
    "add_Al13_before",
    "add_FeB_before",
    "add_feldspar_before",
    "add_FeTi_before",
    "add_AlGr_before",
    "sulfur_reduction_ratio",
]
