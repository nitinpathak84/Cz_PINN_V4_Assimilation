from dataclasses import dataclass
from physicsnemo.sym.geometry.primitives_2d import Rectangle

@dataclass
class CzGeomParams:
    R_cr: float
    h_m: float
    R_c: float
    H_s: float
    R_w: float
    z_top: float
    bc_thickness: float

@dataclass
class CzGeometries:
    melt: Rectangle
    crystal: Rectangle
    melt_free_band: Rectangle
    crystal_side_band: Rectangle
    interface_band: Rectangle
    axis_band: Rectangle

def build_geometries(p: CzGeomParams) -> CzGeometries:
    eps = p.bc_thickness

    melt = Rectangle((0.0, 0.0), (p.R_cr, p.h_m))
    crystal = Rectangle((0.0, p.h_m), (p.R_c, p.h_m + p.H_s))

    # melt free surface ~ z = h_m
    melt_free_band = Rectangle((0.0, p.h_m - eps), (p.R_cr, p.h_m + eps))

    # interface ~ z = h_m, r <= R_c
    interface_band = Rectangle((0.0, p.h_m - eps), (p.R_c, p.h_m + eps))

    # crystal side ~ r = R_c
    crystal_side_band = Rectangle((p.R_c - eps, p.h_m), (p.R_c + eps, p.h_m + p.H_s))

    # axis ~ r = 0
    axis_band = Rectangle((0.0, 0.0), (eps, p.z_top))

    return CzGeometries(
        melt=melt,
        crystal=crystal,
        melt_free_band=melt_free_band,
        crystal_side_band=crystal_side_band,
        interface_band=interface_band,
        axis_band=axis_band,
    )