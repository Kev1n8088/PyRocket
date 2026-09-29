#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#                                          
#####################################################################

from fipy import CellVariable, FaceVariable, Variable, Gmsh2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm, Viewer
from fipy.boundaryConditions.constraint import Constraint
import numpy as np
from matplotlib import pyplot as plt

from Output import Settings2D
from SectionMesh import get_section_mesh
import Units


# boundaries of the section mesh where heat is exchanged, see SectionMesh.py
HEATED_WALLS  = ("ChamberWall",)
COOLANT_WALLS = ("CoolantBottomWall", "CoolantSideWall", "CoolantTopWall")
COOLANT2_WALLS = ("Coolant2BottomWall", "Coolant2SideWall", "Coolant2TopWall")        # return pass of two-pass cooling
OUTER_WALLS   = ("OuterWall",)


class HeatEquationSolver():
    """[Summary]
    2D heat conduction in the wall section, finite volumes with FiPy (Guyer et al. 2009) on a gmsh mesh, symmetry planes adiabatic.
    'direct' steady state: convective boundaries as face conductances (film coefficient in series with the half cell, Patankar
    1980, Ch. 4), Picard iteration on the temperature dependent h and k; wall temperatures at the wall surfaces.
    'pseudo-transient' and transient runs march the heat equation in time with constant diffusivity; wall temperatures of the
    cells next to the walls.
    """
    half_cell_resistance = True                          # direct solver: include the conduction over the half cell next to the walls
    last_mean_temperature = None                         # direct solver: mean temperature of the previous section, initial guess for the next one
    pseudo_transient_first_step = 50                     # pseudo-transient steady state: first time step as multiple of settings.time_step

    def __init__(self, idx, gas, material, cooling_geometry, halpha_func, halpha_c_func, q_rad, coolant_temperature, ambient_temperature, path, settings,
                 coolant_temperature_2=None, initial_field=None):
        self.idx              = idx                      # location along the chamber contour
        self.gas              = gas                      # isentropic gas object
        self.material         = material                 # material object
        self.cooling_geometry = cooling_geometry         # cooling channel geometry object                    
        self.halpha_func      = halpha_func              # hot gas side heat transfer coefficient function pointer
        self.halpha_c_func    = halpha_c_func            # coolant side heat transfer coefficient function pointer
        self.q_rad            = q_rad                    # radiative heat flux [W/m^2]
        self.T_c              = coolant_temperature      # coolant temperature [K]
        self.T_amb            = ambient_temperature      # ambient temperature around the thruster [K]
        self.settings         = settings
        self.path             = path                     # folder path for figures
        self.boltzmann        = 5.67e-8			         # stefan boltzmann constant 
        self.T_c2             = coolant_temperature_2    # coolant temperature of the return pass of two-pass cooling [K]
        self.initial_field    = initial_field            # direct solver: temperatures of an earlier solution of this section as initial guess


    def heat_transfer_coefficients(self, T_face):
        # heat transfer coefficients of the hot gas and the coolant for the current wall temperatures, T_face per mesh face
        self.halpha, self.T_hg = self.halpha_func(np.mean(T_face[self.faces["ChamberWall"]]), self.idx)
        self.halpha_c_bottom   = self.halpha_c_func(np.mean(T_face[self.faces["CoolantBottomWall"]]), self.idx, wall='bottom')
        self.halpha_c_top      = self.halpha_c_func(np.mean(T_face[self.faces["CoolantTopWall"]]), self.idx, wall='top')
        self.halpha_c_side     = self.halpha_c_func(np.mean(T_face[self.faces["CoolantSideWall"]]), self.idx, wall='side')
        if self.two_pass:
            self.halpha_c2_bottom = self.halpha_c_func(np.mean(T_face[self.faces["Coolant2BottomWall"]]), self.idx, wall='bottom', coolant_pass=2)
            self.halpha_c2_top    = self.halpha_c_func(np.mean(T_face[self.faces["Coolant2TopWall"]]), self.idx, wall='top', coolant_pass=2)
            self.halpha_c2_side   = self.halpha_c_func(np.mean(T_face[self.faces["Coolant2SideWall"]]), self.idx, wall='side', coolant_pass=2)


    ##################################################
    # transient heat equation with flux boundary conditions
    ##################################################

    def setup_boundary_conditions(self, mesh, phi):
        # flux boundary conditions, built once per section. Their coefficients are FiPy variables updated before every time step
        self.halpha_v   = Variable(value=0.0)
        self.T_hg_v     = Variable(value=0.0)
        self.halpha_c_v = {wall: Variable(value=0.0) for wall in ('bottom', 'top', 'side')}
        self.k_v        = FaceVariable(mesh=mesh, value=1.0)
        k, n = self.k_v, mesh.faceNormals

        # q = -k delta T
        constraints = [
            Constraint([(-1 / k * self.halpha_c_v['top'] * (phi.faceValue - self.T_c)) * n], mesh.physicalFaces["CoolantTopWall"]),
            Constraint([(-1 / k * self.halpha_c_v['side'] * (phi.faceValue - self.T_c)) * n], mesh.physicalFaces["CoolantSideWall"]),
            Constraint([(-1 / k * self.halpha_c_v['bottom'] * (phi.faceValue - self.T_c)) * n], mesh.physicalFaces["CoolantBottomWall"]),

            # hot gas side heat transfer on the inner chamber wall, including contribution from radiation
            Constraint([(1 / k * (self.halpha_v * (self.T_hg_v - phi.faceValue) + self.q_rad)) * n], mesh.physicalFaces["ChamberWall"]),

            # Radiation boundary condition on the outside wall
            Constraint([(-self.boltzmann * self.material.eps / k * (phi.faceValue ** 4 - self.T_amb ** 4)) * n], mesh.physicalFaces["OuterWall"]),
        ]
        if self.two_pass:
            self.halpha_c2_v = {wall: Variable(value=0.0) for wall in ('bottom', 'top', 'side')}
            constraints += [
                Constraint([(-1 / k * self.halpha_c2_v['top'] * (phi.faceValue - self.T_c2)) * n], mesh.physicalFaces["Coolant2TopWall"]),
                Constraint([(-1 / k * self.halpha_c2_v['side'] * (phi.faceValue - self.T_c2)) * n], mesh.physicalFaces["Coolant2SideWall"]),
                Constraint([(-1 / k * self.halpha_c2_v['bottom'] * (phi.faceValue - self.T_c2)) * n], mesh.physicalFaces["Coolant2BottomWall"]),
            ]
        for constraint in constraints:
            phi.faceGrad.constrain(constraint)


    def boundary_conditions(self, mesh, phi, x, y):
        # update the coefficients of the flux boundary conditions with the current solution
        T_face = np.asarray(phi.faceValue)
        self.heat_transfer_coefficients(T_face)
        self.halpha_v.setValue(self.halpha)
        self.T_hg_v.setValue(self.T_hg)
        self.halpha_c_v['bottom'].setValue(self.halpha_c_bottom)
        self.halpha_c_v['top'].setValue(self.halpha_c_top)
        self.halpha_c_v['side'].setValue(self.halpha_c_side)
        if self.two_pass:
            self.halpha_c2_v['bottom'].setValue(self.halpha_c2_bottom)
            self.halpha_c2_v['top'].setValue(self.halpha_c2_top)
            self.halpha_c2_v['side'].setValue(self.halpha_c2_side)
        self.k_v.setValue(self.material.k(T_face))       # thermal conductivity of the material (dependent on temperature)


    def adaptive_time_step(self, T_max, low_val=50, max_val=400):
        # adaptive time step based on temperature development of the last two time steps
        delta_T = T_max[-1] - T_max[-2]
        step_down = 2

        if delta_T < 0:
            # temperature decreasing — past peak, step down to home in on steady state
            self.time_step /= step_down

        elif delta_T < low_val:
            # temperature actively rising — increase time step to reach steady state faster
            self.time_step *= self.settings.adaptive_up

        elif delta_T > max_val:
            # unusually large jump — step down for stability
            self.time_step /= step_down


    def run_transient(self, mesh, phi, x, y):
        self.setup_boundary_conditions(mesh, phi)

        # Equation to solve
        eq = TransientTerm() == DiffusionTerm(coeff=self.material.alpha)

        running = True
        T_max = [self.T_amb, self.T_amb]
        time = 0
        step = 0

        # initalise first time step with value from settings. Reaching steady state this way only needs a stable time step,
        # larger than the one needed to resolve a real transient
        self.time_step = self.settings.time_step
        if self.settings.run_time == 'steady_state':
            self.time_step *= self.pseudo_transient_first_step

        while running:
            # update boundary conditions
            self.boundary_conditions(mesh, phi, x, y)

            # update time step
            self.adaptive_time_step(T_max)

            # solve equation and update time step
            eq.solve(var=phi, dt=self.time_step)
            time += self.time_step

            # termination critera when steady state is approximately reached
            diff = abs(T_max[-1] - max(phi))
            if diff < self.settings.tolerance:
                running = False 
				
			# terminate at specific run time if it is set
            if self.settings.run_time != 'steady_state':
                if time >= self.settings.run_time:
                    running = False
			
			# terminate if maximum number of iterations is reached 
            if step >= self.settings.max_iter:
                print('No convergence reached after ', self.settings.max_iter, ' iterations. Error of last time step is ', diff)
                print('Increase mesh resolution or decrease time step for better convergence behaviour')
                running = False
				
            if self.settings.print_result:
                print('time: ', round(time,3), ' [s]	T max: ', round(Units.temperature(max(phi)), 3), ' [' + Units.temperature_unit() + ']')

            T_max.append(max(phi))
            step += 1 
            self.diff = diff

        # evaluate the solution once as plain arrays, indexing the FiPy variables re-evaluates them on every access
        T_face = np.asarray(phi.faceValue)
        grad   = np.asarray(phi.faceGrad)
        q_face = self.material.k(T_face) * np.sqrt(grad[0]**2 + grad[1]**2)      # q = k * dT_dn, boundary fluxes along normal vectors
        return T_face, q_face


    ##################################################
    # direct steady state solution
    ##################################################

    def run_steady(self, mesh, phi):
        # geometry of the boundary faces: owning cell, face length and distance from the cell centre to the face
        cell_centres = np.asarray(mesh.cellCenters)
        face_centres = np.asarray(mesh.faceCenters)
        normals      = np.asarray(mesh.faceNormals)
        face_length  = np.asarray(mesh._faceAreas)
        cell_volume  = np.asarray(mesh.cellVolumes)
        owner        = np.asarray(mesh.faceCellIDs[0])
        walls = HEATED_WALLS + COOLANT_WALLS + OUTER_WALLS + (COOLANT2_WALLS if self.two_pass else ())
        faces = {w: np.flatnonzero(self.faces[w]) for w in walls}
        cells = {w: owner[faces[w]] for w in walls}
        dist  = {w: np.abs(np.sum((face_centres[:, faces[w]] - cell_centres[:, cells[w]]) * normals[:, faces[w]], axis=0)) for w in walls}
        if not self.half_cell_resistance:
            dist = {w: 0.0 * dist[w] for w in walls}

        # linear system: div(k grad T) - C T + S = 0, C and S carry the heat exchange of the boundary faces
        k_face = FaceVariable(mesh=mesh, value=1.0)
        C = CellVariable(mesh=mesh, value=0.0)
        S = CellVariable(mesh=mesh, value=0.0)
        eq = DiffusionTerm(coeff=k_face) - ImplicitSourceTerm(coeff=C) + S == 0

        tolerance = min(self.settings.tolerance, 1e-3)   # temperature change between iterations [K]
        if self.initial_field is not None and len(self.initial_field) == mesh.numberOfCells:
            phi.setValue(self.initial_field)
        elif HeatEquationSolver.last_mean_temperature is not None:
            phi.setValue(HeatEquationSolver.last_mean_temperature)
        T_wall = {w: np.asarray(phi)[cells[w]] for w in walls}
        for iteration in range(1, 101):
            T_cell = np.asarray(phi).copy()

            # heat transfer coefficients and conductivity at the current wall temperatures
            T_face = np.asarray(phi.faceValue).copy()
            for w in walls:
                T_face[faces[w]] = T_wall[w]
            self.heat_transfer_coefficients(T_face)
            h = {"ChamberWall": self.halpha, "CoolantBottomWall": self.halpha_c_bottom, "CoolantSideWall": self.halpha_c_side, "CoolantTopWall": self.halpha_c_top}
            T_ref = {"ChamberWall": self.T_hg + self.q_rad / self.halpha}                # radiation from the gas as an increase of the gas temperature
            for w in COOLANT_WALLS:
                T_ref[w] = self.T_c
            if self.two_pass:
                h.update({"Coolant2BottomWall": self.halpha_c2_bottom, "Coolant2SideWall": self.halpha_c2_side, "Coolant2TopWall": self.halpha_c2_top})
                for w in COOLANT2_WALLS:
                    T_ref[w] = self.T_c2
            # radiation to the ambient, linearised: h_rad = sigma eps (T^2 + T_amb^2)(T + T_amb) (Incropera et al. 2007, Eq. 1.9)
            T_o = T_wall["OuterWall"]
            h["OuterWall"] = self.boltzmann * self.material.eps * (T_o**2 + self.T_amb**2) * (T_o + self.T_amb)
            T_ref["OuterWall"] = self.T_amb

            k_face.setValue(self.material.k(T_face))
            C_val, S_val, h_eff = np.zeros(len(cell_volume)), np.zeros(len(cell_volume)), {}
            for w in walls:
                k_w = self.material.k(T_cell[cells[w]])
                h_eff[w] = 1.0 / (1.0 / h[w] + dist[w] / k_w)                           # film coefficient in series with the half cell
                np.add.at(C_val, cells[w], h_eff[w] * face_length[faces[w]] / cell_volume[cells[w]])
                np.add.at(S_val, cells[w], h_eff[w] * face_length[faces[w]] * T_ref[w] / cell_volume[cells[w]])
            C.setValue(C_val)
            S.setValue(S_val)

            eq.solve(var=phi)
            T_new = np.asarray(phi)

            # heat flux into the wall per boundary face [W/m^2] and wall surface temperatures from continuity of the flux
            q = {w: h_eff[w] * (T_ref[w] - T_new[cells[w]]) for w in walls}
            T_wall = {w: T_ref[w] - q[w] / h[w] for w in walls}
            heat = {w: np.sum(q[w] * face_length[faces[w]]) for w in walls}
            q_in = heat["ChamberWall"]
            imbalance = abs(sum(heat.values())) / max(abs(q_in), 1e-12)

            self.diff = np.max(np.abs(T_new - T_cell))
            if self.settings.print_result:
                print('iteration: ', iteration, '	T max: ', round(Units.temperature(np.max(T_new)), 3), ' [' + Units.temperature_unit() + ']', '	change: %.2e K, heat balance error %.2e' % (self.diff, imbalance))
            # converged when the temperatures stop changing; the heat balance holds to the accuracy of the linear solver
            if self.diff < tolerance and imbalance < 1e-4:
                break
        else:
            print('Direct steady state solution not converged after 100 iterations: temperature change %.3g K, heat balance error %.3g' % (self.diff, imbalance))

        self.iterations = iteration
        self.heat_balance_error = imbalance
        HeatEquationSolver.last_mean_temperature = float(np.mean(T_new))

        # face temperatures and heat fluxes, positive out of the wall for the coolant and outer walls as in the transient solution
        T_face = np.asarray(phi.faceValue).copy()
        q_face = np.zeros(len(face_length))
        for w in walls:
            T_face[faces[w]] = T_wall[w]
            q_face[faces[w]] = q[w] if w in HEATED_WALLS else -q[w]
        return T_face, q_face


    def run_sim(self):
        # generate mesh
        mesh = get_section_mesh(self.settings.cell_size, self.idx, self.cooling_geometry)
        # boundary face masks as plain arrays
        self.faces = {name: np.asarray(faces, dtype=bool) for name, faces in mesh.physicalFaces.items()}
        self.two_pass = "Coolant2BottomWall" in self.faces
        if self.two_pass and self.T_c2 is None:
            raise ValueError('two-pass section needs the coolant temperature of the return pass')
        # set solution variable and intial guess
        phi = CellVariable(name="Temperature [K]", mesh=mesh, value=float(self.T_amb))
        x, y = mesh.faceCenters

        if self.settings.run_time == 'steady_state' and getattr(self.settings, 'steady_solver', 'direct') == 'direct':
            T_face, q_face = self.run_steady(mesh, phi)
        else:
            T_face, q_face = self.run_transient(mesh, phi, x, y)

        self.outer_wall_temp = np.max(T_face[self.faces["OuterWall"]])
        self.inner_wall_temp = np.max(T_face[self.faces["ChamberWall"]])
        self.coolant_wall_temp = np.max(np.concatenate([T_face[self.faces[w]] for w in COOLANT_WALLS]))
        self.solution = np.asarray(phi).copy()

        self.q_c_top     = q_face[self.faces["CoolantTopWall"]]
        self.q_c_bottom  = q_face[self.faces["CoolantBottomWall"]]
        self.q_c_side    = q_face[self.faces["CoolantSideWall"]]
        self.q_chamber   = q_face[self.faces["ChamberWall"]]
        self.q_outer     = q_face[self.faces["OuterWall"]]

		# total coolant side heat transferred [W/m]
        face_length = np.asarray(mesh._faceAreas)                                 # length of the boundary faces of the 2D mesh [m]
        self.Q_c = 2 * (np.sum(self.q_c_top * face_length[self.faces["CoolantTopWall"]])
                        + np.sum(self.q_c_bottom * face_length[self.faces["CoolantBottomWall"]])
                        + np.sum(self.q_c_side * face_length[self.faces["CoolantSideWall"]]))

        if self.two_pass:
            # return pass: the section holds half a channel of each pass, so Q_c and Q_c2 are per channel of their pass [W/m]
            self.coolant_wall_temp_2 = np.max(np.concatenate([T_face[self.faces[w]] for w in COOLANT2_WALLS]))
            self.Q_c2 = 2 * sum(np.sum(q_face[self.faces[w]] * face_length[self.faces[w]]) for w in COOLANT2_WALLS)

        if self.settings.save_fig:
            # save section temperature profile
            T_plot = CellVariable(name='Temperature [' + Units.temperature_unit() + ']', mesh=mesh, value=Units.temperature(np.asarray(phi)))
            viewer = Viewer(vars=T_plot, datamin=min(T_plot), datamax=max(T_plot))
            viewer.plot(filename=(self.path + "/" + str(self.idx)))
            plt.close()

        if self.settings.log_thermocouples and self.two_pass:
            print('Thermocouple logging is not available for two-pass sections, they have no rib symmetry line')
        elif self.settings.log_thermocouples:
            # log temperature value at thermocouple locations for validation purposes
            if self.idx in self.settings.thermocouples[:,0]:

                # find which thermocouple is nearest
                TC = np.where(self.idx == self.settings.thermocouples[:,0])[0][0]

                def closest_node(node, nodes):
                    #determine link between TC position and mesh nodes
                    dist = (nodes - node)**2
                    return np.argmin(dist)
                
                # get x position in this section mesh
                x_mesh = closest_node(self.settings.thermocouples[TC, 1], x[mesh.physicalFaces["SideWall"]])
                print('Thermocouple location:    ', self.idx)
                self.T_TC = phi.faceValue[mesh.physicalFaces["SideWall"]][x_mesh]
                print('Thermocouple temperature: ', self.T_TC)
		
		
		
if __name__ == "__main__":

    import os
    from CaseSetup import load_config

    config = load_config('config.py')
	
    try:
        os.mkdir('TestSectionThermalSim')
    except:
        pass
		
    def halpha_func(T, idx):
        return 1600, 2400

    def halpha_c_func(T, idx, wall=None):
        return 1113

    q_rad = 0
    T_c = 288
    T_amb = 288
    idx = 1
	
    solver = HeatEquationSolver(
        idx,
        config.gas,
        config.material,
        config.cooling_geom,
        halpha_func,
        halpha_c_func,
        q_rad,
        T_c,
        T_amb,
        path='TestSectionThermalSim',
        settings=config.settings
    )

    solver.run_sim()

