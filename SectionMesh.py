#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#                                          
#####################################################################

import threading
from concurrent.futures import ThreadPoolExecutor, Future

from fipy import Gmsh2D


# gmsh (Geuzaine & Remacle 2009) input of a section: half a channel and half a rib, bounded by the symmetry lines of the channel and the rib
SECTION_GEO = '''
              cellSize = %(cell_size)g;
              r_c   = %(r_c)g;
              h_c   = %(h_c)g;
              t_w_i = %(t_w_i)g;
              t_w_o = %(t_w_o)g;
              psi_c = %(psi_c)g;
              psi_w = %(psi_w)g;

              Point(1) = {0, 0, 0};             
              
              Point(2) = {0, r_c, 0, cellSize};
              Point(9) = {0, r_c + t_w_i, 0, cellSize};
              Point(6) = {0, r_c + t_w_i + h_c, 0, cellSize};
              Point(5) = {0, r_c + t_w_i + h_c + t_w_o, 0, cellSize};
              
              Point(8) = {Sin(psi_c/2)*(r_c + t_w_i), Cos(psi_c/2)*(r_c + t_w_i), 0, cellSize};
              Point(7) = {Sin(psi_c/2)*(r_c + t_w_i + h_c), Cos(psi_c/2)*(r_c + t_w_i + h_c), 0, cellSize};
              
              Point(3) = {Sin(psi_c/2 + psi_w/2)*r_c, Cos(psi_c/2 + psi_w/2)*r_c, 0, cellSize};
              Point(4) = {Sin(psi_c/2 + psi_w/2)*(r_c + t_w_i + h_c + t_w_o), Cos(psi_c/2+ psi_w/2)*(r_c + t_w_i + h_c + t_w_o), 0, cellSize};
              
              Circle(11) = {2, 1, 3};
              Line(12)   = {3, 4};
              Circle(13) = {4, 1, 5};
              Line(14)   = {5, 6};
              Circle(15) = {6, 1, 7};
              Line(16)   = {7, 8};
              Circle(17) = {8, 1, 9};
              Line(18)   = {9, 2};

              Line Loop(20) = {11, 12, 13 , 14, 15, 16, 17, 18};
              Plane Surface(21) = {20};
              Physical Surface("Domain")         = {21};
              Physical Line("ChamberWall")       = {11};
              Physical Line("SideWall")          = {12};
              Physical Line("OuterWall")         = {13};
              Physical Line("TopWall")           = {14};
              Physical Line("CoolantTopWall")    = {15};
              Physical Line("CoolantSideWall")   = {16};
              Physical Line("CoolantBottomWall") = {17};
              Physical Line("BottomWall")        = {18};
              
              '''

# gmsh input of a two-pass section: from the centre of a forward (pass 1) channel over a full rib to the centre of the neighbouring
# return (pass 2) channel. The channels alternate around the circumference, so these are the lines of symmetry
TWO_PASS_GEO = '''
              cellSize = %(cell_size)g;
              r0 = %(r_c)g;
              r1 = %(r_c)g + %(t_w_i)g;
              r2 = %(r_c)g + %(t_w_i)g + %(h_c)g;
              r3 = %(r_c)g + %(t_w_i)g + %(h_c)g + %(t_w_o)g;
              ta = %(psi_c)g / 2;
              tb = %(psi_c)g / 2 + %(psi_w)g;
              tp = %(psi_c)g + %(psi_w)g;

              Point(1)  = {0, 0, 0};

              Point(2)  = {0, r0, 0, cellSize};
              Point(3)  = {0, r1, 0, cellSize};
              Point(4)  = {0, r2, 0, cellSize};
              Point(5)  = {0, r3, 0, cellSize};

              Point(6)  = {Sin(ta)*r1, Cos(ta)*r1, 0, cellSize};
              Point(7)  = {Sin(ta)*r2, Cos(ta)*r2, 0, cellSize};
              Point(8)  = {Sin(tb)*r1, Cos(tb)*r1, 0, cellSize};
              Point(9)  = {Sin(tb)*r2, Cos(tb)*r2, 0, cellSize};

              Point(10) = {Sin(tp)*r0, Cos(tp)*r0, 0, cellSize};
              Point(11) = {Sin(tp)*r1, Cos(tp)*r1, 0, cellSize};
              Point(12) = {Sin(tp)*r2, Cos(tp)*r2, 0, cellSize};
              Point(13) = {Sin(tp)*r3, Cos(tp)*r3, 0, cellSize};

              Circle(21) = {2, 1, 10};
              Line(22)   = {10, 11};
              Circle(23) = {11, 1, 8};
              Line(24)   = {8, 9};
              Circle(25) = {9, 1, 12};
              Line(26)   = {12, 13};
              Circle(27) = {13, 1, 5};
              Line(28)   = {5, 4};
              Circle(29) = {4, 1, 7};
              Line(30)   = {7, 6};
              Circle(31) = {6, 1, 3};
              Line(32)   = {3, 2};

              Line Loop(40) = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
              Plane Surface(41) = {40};
              Physical Surface("Domain")          = {41};
              Physical Line("ChamberWall")        = {21};
              Physical Line("OuterWall")          = {27};
              Physical Line("SymmetryWall")       = {22, 26, 28, 32};
              Physical Line("CoolantBottomWall")  = {31};
              Physical Line("CoolantSideWall")    = {30};
              Physical Line("CoolantTopWall")     = {29};
              Physical Line("Coolant2BottomWall") = {23};
              Physical Line("Coolant2SideWall")   = {24};
              Physical Line("Coolant2TopWall")    = {25};
              '''

# meshes are cached by their gmsh input, sections with the same geometry (e.g. along the cylindrical chamber) share one mesh
_cache = {}
_lock = threading.Lock()
_pool = None
PREFETCH_WORKERS = 4


def section_geo(cell_size, idx, cooling_geom):
    # gmsh input for the section at contour index idx
    r_c = cooling_geom.geom[idx,1]          # inner chamber radius
    h_c = cooling_geom.h_c[idx]      # height of the cooling channel
    t_w_i = cooling_geom.t_w_i[idx]  # inner wall thickness  
    t_w_o = cooling_geom.t_w_o[idx]  # outer wall thickness
    psi_c = cooling_geom.psi_c_n[idx]     # radial section of Single cooling channel, normal to the channel direction
    psi_w = cooling_geom.psi_w_n[idx]     # radial section of Single wall segment, normal to the channel direction
    template = TWO_PASS_GEO if getattr(cooling_geom, 'passes', 1) == 2 else SECTION_GEO
    return template % locals()


def prefetch_section_meshes(cell_size, cooling_geom, indices):
    # start generating the meshes of the given sections in the background, in the order they will be needed.
    # gmsh runs as a separate process, so several meshes are generated at the same time
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)
        for idx in indices:
            geo = section_geo(cell_size, idx, cooling_geom)
            if geo not in _cache:
                _cache[geo] = _pool.submit(Gmsh2D, geo)


def get_section_mesh(cell_size, idx, cooling_geom):
    geo = section_geo(cell_size, idx, cooling_geom)
    with _lock:
        entry = _cache.get(geo)
    if entry is None:
        entry = Gmsh2D(geo)
    elif isinstance(entry, Future):
        entry = entry.result()
    with _lock:
        _cache[geo] = entry
    return entry
