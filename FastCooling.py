#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Fast cooling model for design sweeps. The wall stress estimate is in Stress.py.
#
# The fast model is the regenerative cooling march of RegenCooling.HeatTransfer along the chamber (same hot gas and coolant heat transfer,
# film cooling, curvature, pressure drop and two-pass shooting), with the FiPy/gmsh section solve replaced by one of:
#   'grid'    a structured finite volume grid of the section, unrolled flat (default, within a few K of the 2D solver, ~30x faster)
#   'network' a thermal resistance network: hot gas film -> hot wall -> channel bottom + two half ribs as fins, the rib tips feeding the
#             outer wall over the channel as a second fin (fastest, but misses the sideways conduction to wide ribs of low conductivity walls)
# Coolant properties come from a table built once from thermo.

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from RegenCooling import HeatTransfer
from Output import Output1D
from PropLibrary import create_coolant


##################################################
# coolant property table
##################################################

class TableState:
    # liquid coolant state with the attributes RegenCooling uses from a thermo Mixture
    def __init__(self, IDs, ws, T, P, rho, mu, Cp, Pr):
        self.IDs, self.ws, self.T, self.P = IDs, ws, float(T), float(P)
        self.phase = 'l'
        self.rhol = self.rhog = self.rho = rho
        self.mul = self.mug = self.mu = mu
        self.Cpl = self.Cpg = self.Cp = Cp
        self.Prl = self.Prg = self.Pr = Pr


class CoolantTable:
    """[Summary]
    Liquid coolant properties on a temperature and pressure grid, interpolated bilinearly (viscosity in log space, as liquid viscosity
    is close to exponential in temperature, Andrade 1930). States outside the
    grid or not liquid fall back to thermo. Built once per case, as thermo takes milliseconds per state.
    """
    def __init__(self, IDs, ws, T_min, T_max, P_min, P_max, n_T=240, n_P=6):
        self.IDs, self.ws = list(IDs), list(ws)
        self.T = np.linspace(T_min, T_max, n_T)
        self.P = np.linspace(P_min, P_max, n_P)
        shape = (n_P, n_T)
        self.rho, self.log_mu, self.Cp, self.Pr = (np.full(shape, np.nan) for _ in range(4))
        for i, P in enumerate(self.P):
            for j, T in enumerate(self.T):
                try:
                    m = create_coolant(self.IDs, ws=self.ws, T=T, P=P)
                except Exception:
                    continue
                if m.phase == 'l' and m.Prl is not None:
                    self.rho[i, j], self.log_mu[i, j], self.Cp[i, j], self.Pr[i, j] = m.rhol, np.log(m.mul), m.Cpl, m.Prl

    def state(self, T, P):
        # interpolated state, None where the table does not cover it
        if not (self.T[0] <= T <= self.T[-1] and self.P[0] <= P <= self.P[-1]):
            return None
        j = min(int((T - self.T[0]) / (self.T[1] - self.T[0])), len(self.T) - 2)
        i = min(int((P - self.P[0]) / (self.P[1] - self.P[0])), len(self.P) - 2) if len(self.P) > 1 else 0
        fT = (T - self.T[j]) / (self.T[1] - self.T[0])
        fP = (P - self.P[i]) / (self.P[1] - self.P[0]) if len(self.P) > 1 else 0.0
        def interp(a):
            return ((1 - fP) * ((1 - fT) * a[i, j] + fT * a[i, j + 1]) + fP * ((1 - fT) * a[i + 1, j] + fT * a[i + 1, j + 1]))
        values = [interp(a) for a in (self.rho, self.log_mu, self.Cp, self.Pr)]
        if not np.all(np.isfinite(values)):
            return None
        return TableState(self.IDs, self.ws, T, P, values[0], np.exp(values[1]), values[2], values[3])


##################################################
# 1D station model
##################################################

class StationResult:
    # results of a 1D station solve, with the attributes RegenCooling reads from the 2D solver
    pass


def half_rib_conductance(h_side, h_top, k, rib, height, t_outer, width):
    # conductance [W/m/K per unit channel length] of half a rib (thickness rib/2, one wetted side, adiabatic mid plane) as a fin,
    # whose tip feeds the outer wall over half the channel width, itself a fin wetted on its channel side (the channel top wall).
    # Straight fin with a tip conductance: Incropera et al. (2007) Table 3.4 case A, with h_tip A_c replaced by G_top
    M = np.sqrt(h_side * k * rib / 2)
    m = np.sqrt(h_side / (k * rib / 2))
    G_top = np.sqrt(h_top * k * t_outer) * np.tanh(np.sqrt(h_top / (k * t_outer)) * width / 2)
    beta = G_top / M
    mH = m * height
    return M * (np.sinh(mH) + beta * np.cosh(mH)) / (np.cosh(mH) + beta * np.sinh(mH))


class GridSection:
    """[Summary]
    Structured finite volume grid of a cooling channel section, unrolled flat: x across the channels (normal to the flow), y from the hot 
    face outwards. Single pass: from the channel centre to the rib centre. Two-pass: from the centre of a first pass channel over a full rib 
    to the centre of a return channel. The hot face and the channel top wall areas are scaled with their radius relative to the channel 
    bottom. Heat exchange at the boundaries: heat transfer coefficient in series with the conduction over the half cell next to the wall.
    Interface conductances are the series (harmonic mean) conductances of the two half cells (Patankar 1980, Sec. 4.2).
    """
    def __init__(self, width, rib, t_wall, height, t_outer, r_gas, r_channel, two_pass, cells_across=4):
        self.two_pass = two_pass
        target = min(t_wall, height, width / 2, rib / 2) / cells_across

        def segments(lengths):
            edges = [0.0]
            for L in lengths:
                n = max(2, int(np.ceil(L / target)))
                edges += list(edges[-1] + np.linspace(0, L, n + 1)[1:])
            return np.array(edges)

        x_edges = segments([width / 2, rib, width / 2] if two_pass else [width / 2, rib / 2])
        y_edges = segments([t_wall, height, t_outer])
        self.dx, self.dy = np.diff(x_edges), np.diff(y_edges)
        xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
        X, Yc = x_edges[-1], yc
        in_band = (yc > t_wall) & (yc < t_wall + height)
        fluid1 = np.outer(xc < width / 2, in_band)                     # [i, j]
        fluid2 = np.outer(xc > X - width / 2, in_band) if two_pass else np.zeros_like(fluid1)
        solid = ~(fluid1 | fluid2)
        self.solid = solid
        self.ids = -np.ones(solid.shape, dtype=int)
        self.ids[solid] = np.arange(solid.sum())
        self.n = int(solid.sum())
        self.width_hot = X * r_gas / r_channel                         # hot face width [m] per section
        scale_top = (r_channel + height) / r_channel

        nx, ny = solid.shape
        # interior neighbour pairs of solid cells: ids a, b and the geometric factors for the conductance
        pairs = []
        for i in range(nx - 1):
            for j in range(ny):
                if solid[i, j] and solid[i + 1, j]:
                    pairs.append((self.ids[i, j], self.ids[i + 1, j], self.dy[j], self.dx[i] / 2, self.dx[i + 1] / 2))
        for i in range(nx):
            for j in range(ny - 1):
                if solid[i, j] and solid[i, j + 1]:
                    pairs.append((self.ids[i, j], self.ids[i, j + 1], self.dx[i], self.dy[j] / 2, self.dy[j + 1] / 2))
        p = np.array(pairs)
        self.pa, self.pb, self.p_len, self.p_da, self.p_db = p[:, 0].astype(int), p[:, 1].astype(int), p[:, 2], p[:, 3], p[:, 4]

        # boundary faces: (cell id, face length, distance centre to face, kind, pass)
        faces = []
        for i in range(nx):
            faces.append((self.ids[i, 0], self.dx[i] * r_gas / r_channel, self.dy[0] / 2, 'hot', 0))
        for pas, fluid in ((1, fluid1), (2, fluid2)):
            for i in range(nx):
                for j in range(ny):
                    if not solid[i, j]:
                        continue
                    if j + 1 < ny and fluid[i, j + 1]:
                        faces.append((self.ids[i, j], self.dx[i], self.dy[j] / 2, 'bottom', pas))
                    if j - 1 >= 0 and fluid[i, j - 1]:
                        faces.append((self.ids[i, j], self.dx[i] * scale_top, self.dy[j] / 2, 'top', pas))
                    if (i + 1 < nx and fluid[i + 1, j]) or (i - 1 >= 0 and fluid[i - 1, j]):
                        faces.append((self.ids[i, j], self.dy[j], self.dx[i] / 2, 'side', pas))
        self.f_cell = np.array([f[0] for f in faces], dtype=int)
        self.f_len = np.array([f[1] for f in faces])
        self.f_dist = np.array([f[2] for f in faces])
        self.f_kind = np.array([f[3] for f in faces])
        self.f_pass = np.array([f[4] for f in faces])

    def solve(self, k_cells, h_faces, T_faces_ref):
        # temperatures of the solid cells for cell conductivities k_cells [W/m/K], face heat transfer coefficients and reference temperatures
        ka, kb = k_cells[self.pa], k_cells[self.pb]
        G = self.p_len / (self.p_da / ka + self.p_db / kb)
        G_face = self.f_len / (1 / h_faces + self.f_dist / k_cells[self.f_cell])
        rows = np.concatenate([self.pa, self.pb, self.pa, self.pb, self.f_cell])
        cols = np.concatenate([self.pa, self.pb, self.pb, self.pa, self.f_cell])
        vals = np.concatenate([G, G, -G, -G, G_face])
        A = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        b = np.bincount(self.f_cell, weights=G_face * T_faces_ref, minlength=self.n)
        T = scipy.sparse.linalg.spsolve(A.tocsc(), b)
        q_faces = G_face * (T_faces_ref - T[self.f_cell]) / self.f_len            # heat flux into the wall per face [W/m^2]
        return T, q_faces


class QuietOutput(Output1D):
    # output arrays without writing sim_data.csv
    def write_csv(self):
        pass


class FastHeatTransfer(HeatTransfer):
    """[Summary]
    Regenerative cooling march with the 1D station model instead of the 2D section solver. The wall conductivity is taken at the local 
    wall temperatures (mean hot wall temperature for the hot wall, coolant side wall temperature for the ribs) within the iteration that 
    each station needs anyway for the temperature dependent heat transfer coefficients; k_fixed [W/m/K] uses one constant value instead. 
    table is an optional CoolantTable for fast coolant properties.
    """
    def __init__(self, *args, k_fixed=None, table=None, section_model='grid', **kwargs):
        super().__init__(*args, **kwargs)
        self.k_fixed = k_fixed
        self.section_model = section_model            # 'grid': structured finite volume section, 'network': thermal resistance network
        self.table = table
        self.verbose = False
        self.section_fields = {}

    def make_coolant(self, IDs, ws=None, T=298.15, P=101325.0):
        state = self.table.state(T, P) if self.table is not None else None
        return state if state is not None else create_coolant(IDs, ws=ws, T=T, P=P)

    def prefetch_meshes(self, order):
        pass

    def solve_section(self, idx, two_pass=False):
        if self.section_model == 'grid':
            return self.solve_grid(idx, two_pass)
        return self.solve_network(idx, two_pass)

    def solve_grid(self, idx, two_pass=False):
        cg = self.cooling_geometry
        r, r_i = self.geometry[idx, 1], cg.r_i[idx]
        width, rib = cg.psi_c_n[idx] * r_i, cg.psi_w_n[idx] * r_i
        # the grid of a station stays the same during a run; later two-pass iterations start from the previous temperatures
        grids = self.__dict__.setdefault('_grids', {})
        if (idx, two_pass) not in grids:
            grids[idx, two_pass] = GridSection(width, rib, cg.t_w_i[idx], cg.h_c[idx], cg.t_w_o[idx], r, r_i, two_pass)
        grid = grids[idx, two_pass]
        passes = (1, 2) if two_pass else (1,)
        T_c = {1: float(self.coolant.T), 2: float(self.coolant2.T) if two_pass else None}
        hot = grid.f_kind == 'hot'
        coolant = {(p, kind): (grid.f_pass == p) & (grid.f_kind == kind) for p in passes for kind in ('bottom', 'side', 'top')}

        previous = self.section_fields.get(idx)
        if previous is not None:
            T_cells, T_face_hot, T_face_c = previous
        else:
            T_cells = np.full(grid.n, T_c[1] + 150.0)
            T_face_hot, T_face_c = T_c[1] + 300.0, {p: T_c[p] + 100.0 for p in passes}
        h = np.zeros(len(grid.f_cell))
        T_ref = np.zeros(len(grid.f_cell))
        for iteration in range(50):
            k_cells = np.full(grid.n, self.k_fixed) if self.k_fixed is not None else self.material.k(T_cells)
            h_g, T_hg = self.heat_trans_coeff_gas(T_face_hot, idx)
            h[hot], T_ref[hot] = h_g, T_hg + self.q_rad / h_g               # radiation from the gas as an increase of the gas temperature
            h_c = {}
            for p in passes:
                for kind in ('bottom', 'side', 'top'):
                    h_c[p, kind] = self.heat_trans_coeff_coolant(T_face_c[p], idx, wall=kind, coolant_pass=p)
                    h[coolant[p, kind]], T_ref[coolant[p, kind]] = h_c[p, kind], T_c[p]
            T_new, q_faces = grid.solve(k_cells, h, T_ref)
            T_faces = T_ref - q_faces / h                                   # wall surface temperatures
            T_face_hot_new = float(np.average(T_faces[hot], weights=grid.f_len[hot]))
            wet = {p: (grid.f_pass == p) for p in passes}
            T_face_c_new = {p: float(np.average(T_faces[wet[p]], weights=grid.f_len[wet[p]])) for p in passes}
            change = max([abs(T_face_hot_new - T_face_hot), float(np.max(np.abs(T_new - T_cells)))] + [abs(T_face_c_new[p] - T_face_c[p]) for p in passes])
            T_cells, T_face_hot, T_face_c = T_new, T_face_hot_new, T_face_c_new
            if change < 0.01:
                break
        self.section_fields[idx] = (T_cells, T_face_hot, T_face_c)

        res = StationResult()
        res.inner_wall_temp = float(np.max(T_faces[hot]))
        res.inner_wall_temp_mean = T_face_hot
        res.coolant_wall_temp = float(np.max(T_faces[grid.f_pass == 1]))
        res.halpha, res.T_hg, res.halpha_c_bottom = h_g, T_hg, h_c[1, 'bottom']
        heat_in = float(np.sum(q_faces[hot] * grid.f_len[hot]))
        res.q_chamber, res.q_outer = np.array([heat_in / grid.width_hot]), np.array([0.0])
        # heat per channel of each pass [W/m]: the section holds half a channel of each pass
        res.Q_c = -2 * float(np.sum(q_faces[grid.f_pass == 1] * grid.f_len[grid.f_pass == 1]))
        if two_pass:
            res.Q_c2 = -2 * float(np.sum(q_faces[grid.f_pass == 2] * grid.f_len[grid.f_pass == 2]))
            res.coolant_wall_temp_2 = float(np.max(T_faces[grid.f_pass == 2]))
            res.halpha_c2_bottom = h_c[2, 'bottom']
        res.solution = None
        return res

    def solve_network(self, idx, two_pass=False):
        cg = self.cooling_geometry
        r, r_i = self.geometry[idx, 1], cg.r_i[idx]
        width = cg.psi_c_n[idx] * r_i                    # channel width normal to the flow [m]
        rib = cg.psi_w_n[idx] * r_i                      # rib width normal to the flow [m]
        H, t, t_o = cg.h_c[idx], cg.t_w_i[idx], cg.t_w_o[idx]
        passes = (1, 2) if two_pass else (1,)
        T_c = {1: float(self.coolant.T), 2: float(self.coolant2.T) if two_pass else None}
        A = r * (cg.psi_c_n[idx] + cg.psi_w_n[idx]) * len(passes)       # hot wall width per channel (per channel pair for two-pass) [m]

        T_wg, T_wl = T_c[1] + 300.0, T_c[1] + 100.0
        for iteration in range(100):
            k_wall = self.k_fixed if self.k_fixed is not None else float(self.material.k(0.5 * (T_wg + T_wl)))
            k_rib = self.k_fixed if self.k_fixed is not None else float(self.material.k(T_wl))
            h_g, T_hg = self.heat_trans_coeff_gas(T_wg, idx)
            T_ref = T_hg + self.q_rad / h_g                             # radiation from the gas as an increase of the gas temperature
            G, h_bottom = {}, {}
            for p in passes:
                h_b = self.heat_trans_coeff_coolant(T_wl, idx, wall='bottom', coolant_pass=p)
                h_s = self.heat_trans_coeff_coolant(T_wl, idx, wall='side', coolant_pass=p)
                h_t = self.heat_trans_coeff_coolant(T_wl, idx, wall='top', coolant_pass=p)
                G[p] = h_b * width + 2 * half_rib_conductance(h_s, h_t, k_rib, rib, H, t_o, width)
                h_bottom[p] = h_b
            G_total = sum(G.values())
            T_eff = sum(G[p] * T_c[p] for p in passes) / G_total
            Q = (T_ref - T_eff) / (1 / (h_g * A) + t / (k_wall * A) + 1 / G_total)
            T_wg_new = T_ref - Q / (h_g * A)
            T_wl_new = T_wg_new - Q * t / (k_wall * A)
            converged = abs(T_wg_new - T_wg) < 0.01 and abs(T_wl_new - T_wl) < 0.01
            T_wg, T_wl = T_wg_new, T_wl_new
            if converged:
                break

        # the network gives the pitch-averaged hot wall temperature. The wall strip over the channel is a heated fin held at the rib
        # temperature at its edges (Incropera et al. 2007, Sec. 3.6), so its centre is hotter by
        # (q/h - (T_wl - T_c)) * (1 - sech(m w/2)), m = sqrt(h/(k t))
        q_avg = Q / A
        excess = 0.0
        for p in passes:
            m_strip = np.sqrt(h_bottom[p] / (k_wall * t))
            excess = max(excess, max(q_avg / h_bottom[p] - (T_wl - T_c[p]), 0.0) * (1 - 1 / np.cosh(m_strip * width / 2)))

        res = StationResult()
        res.inner_wall_temp, res.coolant_wall_temp = T_wg + excess, T_wl + excess       # peak values over the channel centre
        res.inner_wall_temp_mean = T_wg
        res.halpha, res.T_hg, res.halpha_c_bottom = h_g, T_hg, h_bottom[1]
        res.q_chamber, res.q_outer = np.array([Q / A]), np.array([0.0])
        res.Q_c = G[1] * (T_wl - T_c[1])                                # per channel of the pass [W/m]
        if two_pass:
            res.Q_c2, res.coolant_wall_temp_2, res.halpha_c2_bottom = G[2] * (T_wl - T_c[2]), T_wl, h_bottom[2]
        res.solution = None
        return res
