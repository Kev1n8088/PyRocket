#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#                                          
#####################################################################

import numpy as np
import thermo
import scipy.optimize 

from SectionThermalSim import HeatEquationSolver
from SectionMesh import prefetch_section_meshes
from PropLibrary import create_coolant



class HeatTransfer():
    def __init__(self, cea, gas, geometry, material, coolant, cooling_geometry, m_dot, m_dot_coolant, T_amb, output, settings2D, model='standard-bartz', cool_model='gnielinski', eta_c_star=0.92, film=False, bartz_knockdown=1.0, turnaround_loss=1.5):
        """[summary]
        Class to calculate heat tranfer coefficients and radiative heat transfer at arbitrary locations along the chamber contour.
		These functions are passed to the 2D thermal simulation, after which the program advances to the next chamber section. Several options are 
		available for coolant and heat transfer models. Coolant flow properties stored as total conditions, and use a 'thermo' 
		Mixture object to update coolant properties. Hot gas properties from CEA and taken at total chamber conditions. 
		Temperature and pressure of the gas are determined using isentropic expansion.        
        Material thermal condutivity can be temperature dependent.
        USE SI UNITS (sorry freedom lovers)
        """
        self.cea = cea                       			# NASA CEA object 
        self.gas = gas                      			# isentropic gas object 
        self.geometry = geometry             			# engine contour
        self.material = material             			# material object
        self.coolant = coolant               			# themro Mixture object
        self.m_dot = m_dot                   			# total mass flow
        self.m_dot_coolant = m_dot_coolant   			# coolant mass flow [kg/s]
        self.model = model                   			# hot gas side heat transfer model
        self.cool_model = cool_model		 			# coolant side heat transfer model
        self.eta_c_star = eta_c_star       			# combustion efficiency
        self.cooling_geometry = cooling_geometry	 	# cooling channel geometry class
        self.bartz_knockdown = bartz_knockdown        # bartz knockdown factor
        self.turnaround_loss = turnaround_loss        # two-pass cooling: pressure loss coefficient of the turnaround manifold, times the dynamic pressure in the channels
        self.passes = getattr(cooling_geometry, 'passes', 1)
        self.A_c = cooling_geometry.A_c					# cooling channel area [m^2]
        self.D_h = cooling_geometry.D_h				    # cooling channel hydraulic diameter [m]

        self.film = film

        self.T_amb = T_amb								# ambient temperature [K], must be float!!!
        self.boltzmann = 5.67e-8						# stefan boltzmann constant

        self.out = output								# output class
        self.settings2D = settings2D					# settings for 2D thermal sim



    def heat_trans_coeff_gas(self, T_wall, idx):
        # get hot gas properties for this chamber location. CEA properties assumed constant, with isentropic gas properties being taken at local point
        gamma = self.cea.gamma
        mu    = self.cea.mu
        cp    = self.cea.Cp
        Pr    = self.cea.Pr
        T_s   = self.gas.T_s[idx]
        p_s   = self.gas.p_s[idx]
        M     = self.gas.M[idx]
        T_aw  = self.gas.T_aw[idx]

        # local geometrical properties 
        local_area  = np.pi * self.geometry[idx,1]**2
        throat_area = np.pi * np.amin(self.geometry[:,1])**2
        D_t         = 2 * np.amin(self.geometry[:,1])
        
        cstar = self.cea.chamber_pressure * throat_area / self.m_dot
        
        # driving gas temperature for all models: the adiabatic wall temperature (as in RPA's Bartz method, Ponomarenko 2012),
        # corrected for c* efficiency as eta_c*^2 since c* ~ sqrt(T). Replaces the original PyRocket T_s + 0.9 (eta^2 T_aw - T_s),
        # which applied a recovery factor of 0.9 a second time
        T_hg       = T_aw * self.eta_c_star**2

        # film cooling: Stanton number reduction St/St0 from FilmCooling (Shine et al. 2012), 1.0 outside the liquid film
        St_St0 = 1.0
        if self.film != False:
            if idx in self.film.cooled_idx:
                St_St0 = self.film.St_St0
            else:
                pass
        else:
            pass

        # Nusselt number correlation for the combustion gases
        if self.model == 'standard-bartz':
            # Bartz (1957), stagnation properties from CEA. sigma with viscosity exponent omega = 0.6 (0.68 = 0.8 - omega/5, 0.12 = omega/5).
            # Throat curvature factor (D_t/R_throat)^0.1 taken as 1. Tends to overpredict near the throat of small engines
            sigma  = ((0.5 * T_wall/self.cea.Tc * (1+(gamma-1)/2 * M**2) + 0.5)**(0.68) * (1+(gamma-1)/2 * M**2)**(0.12))**(-1)
            halpha = 0.026/(D_t**0.2) * (mu**0.2)*cp/(Pr**0.6) * (self.gas.p_t/cstar)**0.8 * (throat_area/local_area)**0.9 * sigma
            halpha *= self.bartz_knockdown

        elif self.model == 'modified-bartz':
            # Bartz form with the local mass flux G and properties at the Eckert (1955) reference temperature T_f. This variant is from 
            # the original PyRocket, no source given
            T_f    = 0.5 * T_wall + 0.28 * T_s + 0.22 * T_aw
            G      = self.m_dot/local_area
            halpha = 0.026 * G**0.8/(D_t)**0.2 * mu**0.2*cp/Pr**0.6 * (self.cea.Tc/T_f)**0.68
            halpha *= self.bartz_knockdown

        elif self.model == 'cinjarev':
            # Cinjarev: St = 0.0162 Re^-0.18 Pr^-0.18 (T_hot/T_w)^0.35 with the local diameter (Schmidt 1999; Kirchberger et al. 2009;
            # Betti et al. 2015, Table 5). With G = 4 m_dot/(pi D^2) the viscosity cancels: 0.0162 (4/pi)^0.82 = 0.01975. T_hot is T_hg
            # = eta_c*^2 T_aw here, the references use T_aw
            halpha = 0.01975 * self.cea.k**0.18 * (self.m_dot*cp)**0.82 / (2*self.geometry[idx,1])**1.82 * (T_hg/T_wall)**0.35

        else:
            raise ValueError('Invalid heat transfer method. Select: "standard-bartz", "modified-bartz" or "cinjarev"')

        return halpha * St_St0, T_hg
        
        
    def pressure_drop(self, idx, coolant_pass=1):
        # Darcy-Weisbach with the Colebrook (1939) friction factor, roughness from MaterialLib
        surface_roughness = self.material.roughness
        coolant, Re, v_coolant = (self.coolant, self.Re, self.v_coolant) if coolant_pass == 1 else (self.coolant2, self.Re2, self.v_coolant2)
        # fd = scipy.optimize.fsolve(lambda f: -2 * np.log10(surface_roughness / (3.7 * self.D_h[idx]) + 2.51 / (self.Re * np.sqrt(f))) -1 / np.sqrt(f), 0.001)
        # dp = fd * self.section_length[idx] / self.D_h[idx] * 0.5 * self.coolant.rho * self.v_coolant**2 

        # cached: depends only on Re and D_h, which stay the same while a section is solved
        key = (idx, float(Re), surface_roughness)
        cache = self.__dict__.setdefault('_friction_cache', {})
        if cache.get(coolant_pass, (None,))[0] != key:
            cache[coolant_pass] = (key, scipy.optimize.fsolve(lambda f: -2 * np.log10(surface_roughness / (3.7 * self.D_h[idx]) + 2.51 / (Re * np.sqrt(f))) -1 / np.sqrt(f), 0.001)[0])
        fd = cache[coolant_pass][1]
        rho = coolant.rhol if coolant.phase == 'l' else coolant.rhog

        # helical path length: section length / cos(alpha)
        L_eff = self.section_length[idx] * self.cooling_geometry.L_factor[idx]
        dp = fd * L_eff / self.D_h[idx] * 0.5 * rho * v_coolant**2

        # curved channel, turbulent: Ito (1959) f_curved/f_straight = [Re (r/R)^2]^(1/20), valid for Re (r/R)^2 > 6.
        # The returned fd stays the straight-channel value used by the Gnielinski correlation
        dp *= self.curvature_parameter(idx, coolant_pass)**0.05

        return dp, fd


    def curvature_parameter(self, idx, coolant_pass=1):
        # Ito (1959) curvature parameter Re (r/R)^2, r = D_h/2, R = centreline radius of curvature. Floored at 1 so the corrections
        # vanish for straight or weakly curved channels
        R_curve = self.cooling_geometry.R_curve[idx]
        if not np.isfinite(R_curve):
            return 1.0
        Re = self.Re if coolant_pass == 1 else self.Re2
        return max(1.0, float(Re) * (self.D_h[idx] / (2.0 * R_curve))**2)

        
    def radiation(self, idx):
        # empirical H2O and CO2 gas radiation, q [W/m^2] with partial pressure [bar], chamber radius [m] and static temperature.
        # From the original PyRocket, no source given
        R_c = self.geometry[idx,1]					# chamber radius

        p_h2o = self.cea.mole_fractions[1]['H2O'][0] * self.gas.p_s[idx]
        # radiative heat flux of water molecules 
        q_rad = 5.74 * (p_h2o/1e5*R_c)**0.3 * (self.gas.T_s[idx]/100)**3.5

        # check if CO2 is present in the exhaust
        if '*CO2' in self.cea.mole_fractions[1]:
            p_co2 = self.cea.mole_fractions[1]['*CO2'][0] * self.gas.p_s[idx]
            # radiative heat flux of co2 molecules 
            q_rad += 4 * (p_co2/1e5*R_c)**0.3 * (self.gas.T_s[idx]/100)**3.5
    
        return q_rad


    def heat_trans_coeff_coolant(self, T_wall_coolant, idx, wall='bottom', coolant_pass=1):
        # coolant flow velocity based on bulk properteis. With two-pass cooling each pass has half of the channels, the coolant mass 
        # flow passes through both passes one after the other. coolant_pass 2 is the return pass
        coolant = self.coolant if coolant_pass == 1 else self.coolant2
        A_c = self.A_c[idx] if self.passes == 1 else self.A_c[idx] / 2

        rho = coolant.rhol if coolant.phase == 'l' else coolant.rhog
        mu = coolant.mul if coolant.phase == 'l' else coolant.mug
        v_coolant = self.m_dot_coolant / (rho * A_c)
        Re = rho * v_coolant * self.D_h[idx] / mu
        if coolant_pass == 1:
            self.v_coolant, self.Re = v_coolant, Re
        else:
            self.v_coolant2, self.Re2 = v_coolant, Re
		
		# near-wall film temperature as the logarithmic mean of wall and bulk temperature (original PyRocket assumption)
        def get_near_wall_fluid():
            T_avg = (T_wall_coolant - coolant.T) / np.log(T_wall_coolant / coolant.T)
            fluid = self.make_coolant(coolant.IDs, ws=coolant.ws, T=T_avg, P=coolant.P)
            return fluid

		# thermodynamic properties of the near wall fluid, Pr implementation of thermo does not work reliably for mixtures near their critical point. Uses gaseouse Pr if liquid Pr returns 'None'
        if coolant.phase == 'l':
            Pr = coolant.Prl
            Cp = coolant.Cpl
            mu = coolant.mul

        elif coolant.phase == 'g':
            near_wall_coolant = get_near_wall_fluid()
            Pr = near_wall_coolant.Prg
            Cp = near_wall_coolant.Cpg
            mu = near_wall_coolant.mug

        else:
            raise ValueError('Coolant is neither gaseous, nor liquid. Thermo implementation not suitable for supercritical fluids')
        
        k  = Cp * mu / Pr

        # Nusselt number correlations for the coolant side 
        if self.cool_model == 'dittus-boelter':
            # Sieder & Tate (1936) form with the (mu_bulk/mu_wall)^0.14 viscosity correction (option keeps its historical name).
            # Valid for Re > 1e4 and 0.7 < Pr < 16700 (Incropera et al. 2007)
            near_wall_coolant = get_near_wall_fluid()
            Nu = 0.027 * Re**0.8 * Pr**0.33 * (coolant.mu / near_wall_coolant.mu)**(0.14)		

        elif self.cool_model == 'dittus-boelter-simple':
            # Dittus & Boelter (1930) in the McAdams form for heating (Winterton 1998). Valid for Re > 1e4 and 0.6 < Pr < 160 (Incropera et al. 2007)
            Nu = 0.023 * Re**0.8 * Pr**0.4

        elif self.cool_model == 'gnielinski':
            # Gnielinski (1976), valid for 3000 < Re < 5e6 and 0.5 < Pr < 2000. Uses the Colebrook friction factor, so roughness
            # raises Nu (Gnielinski used a smooth-pipe friction factor). No entrance or property-ratio correction
            _, fd = self.pressure_drop(idx, coolant_pass)
            Nu = (fd/8) * (Re - 1000)*Pr / (1 + 12.7*(fd/8)**0.5 * (Pr**(2/3) - 1))

        else:
            raise ValueError('Invalid heat transfer method. Select: "dittus-boelter", "dittus-boelter-simple" or "gnielinski"')

        # helical channel curvature, turbulent: Niino et al. (1982) Nu_curved/Nu_straight = [Re (r/R)^2]^(+0.05) on the concave
        # (outer, top) wall and ^(-0.05) on the convex (hot, bottom) wall. Side walls uncorrected
        if wall == 'top':
            Nu *= self.curvature_parameter(idx, coolant_pass)**0.05
        elif wall == 'bottom':
            Nu *= self.curvature_parameter(idx, coolant_pass)**(-0.05)

        # convert Nusselt number to heat transfer coefficient
        halpha = Nu * k / self.D_h[idx]

        if halpha < 0:
            raise ValueError('Negative heat transfer coefficient, check applicability of cooling model to Reynolds number range')

        # check if halpha is an array or a float (artifact of the pressure drop dependence or fluid model)
        if isinstance(halpha, float):
            return halpha
        else:
            return halpha[0]


    def section_heat_flux(self, idx):
        self.q_rad = self.radiation(idx)
        
        # solve 2D section temperature profile, passes functions for heat transfer coefficients for temperature dependence of boundary conditions
                
        self.log('SOLVING Section Number	', self.section_number, ' / ', len(self.geometry[:,1]))
        
        solver = self.solve_section(idx)

        self.T_wall_i  = solver.inner_wall_temp	    # inner wall temperature taken as maximum temperature of the 2D profile
        self.T_wall_c  = solver.coolant_wall_temp	# maximum temperature of the coolant wetted walls, for the boiling margin
        self.halpha    = solver.halpha			    # heat transfer coefficients taken as the of the boundary conditions of the 2D solver at the last timestep
        self.halpha_c  = solver.halpha_c_bottom		# average heat transfer coefficient at bottom wall
        self.T_hg      = solver.T_hg

        # total heat flux into the inner chamber wall
        self.q 		   = np.mean(solver.q_chamber)

        # radiation leaving to ambient 
        self.q_rad_out = np.mean(solver.q_outer)
		
        # Q_c is per unit channel length [W/m]; the channel runs L_factor = 1/cos(alpha) times the section length
        Q = float(solver.Q_c) * self.section_length[idx] * self.cooling_geometry.L_factor[idx] * self.cooling_geometry.n_channels

 
        # coolant energy balance, dT = Q / (m_dot Cp)
        Cp_bulk        = self.coolant.Cpl if self.coolant.phase == 'l' else self.coolant.Cpg
        dT             = Q / (self.m_dot_coolant * Cp_bulk) 
        dp, _          = self.pressure_drop(idx)
        
        # recalcualte cooling fluid properties, such as density, Pr, Cp etc.
        self.check_pressure(self.coolant.P - dp, idx)
        self.coolant   = self.make_coolant(self.coolant.IDs, ws=self.coolant.ws, T=(self.coolant.T+dT), P=(self.coolant.P-dp))
        
        
    def solve_section(self, idx, two_pass=False):
        # temperature field of the section with the 2D solver; returns the solver with wall temperatures, heat transfer coefficients and heat fluxes
        if not two_pass:
            solver = HeatEquationSolver(idx, self.gas, self.material, self.cooling_geometry, self.heat_trans_coeff_gas, self.heat_trans_coeff_coolant, self.q_rad, self.coolant.T, self.T_amb, self.out.folder_path, self.settings2D)
            solver.run_sim()
            return solver
        solver = HeatEquationSolver(idx, self.gas, self.material, self.cooling_geometry, self.heat_trans_coeff_gas, self.heat_trans_coeff_coolant, self.q_rad,
                                    self.coolant.T, self.T_amb, self.out.folder_path, self.settings2D,
                                    coolant_temperature_2=self.coolant2.T, initial_field=self.section_fields.get(idx))
        solver.run_sim()
        self.section_fields[idx] = solver.solution
        return solver


    def make_coolant(self, IDs, ws=None, T=298.15, P=101325.0):
        # coolant state at temperature T [K] and pressure P [Pa]
        return create_coolant(IDs, ws=ws, T=T, P=P)


    def prefetch_meshes(self, order):
        # generate the section meshes in the background, in the order the coolant passes the sections
        prefetch_section_meshes(self.settings2D.cell_size, self.cooling_geometry, order)


    def log(self, *args):
        # progress messages, off for quiet runs
        if getattr(self, 'verbose', True):
            print(*args)


    def section_heat_flux_two_pass(self, idx):
        # two-pass section: pass 1 (self.coolant) flows in the direction of the march, the return pass (self.coolant2) against it
        self.q_rad = self.radiation(idx)
        self.log('SOLVING Section Number	', self.section_number, ' / ', len(self.geometry[:,1]))

        solver = self.solve_section(idx, two_pass=True)

        self.T_wall_i  = solver.inner_wall_temp
        self.T_wall_c  = solver.coolant_wall_temp
        self.T_wall_c2 = solver.coolant_wall_temp_2
        self.halpha    = solver.halpha
        self.halpha_c  = solver.halpha_c_bottom
        self.halpha_c2 = solver.halpha_c2_bottom
        self.T_hg      = solver.T_hg
        self.q 		   = np.mean(solver.q_chamber)
        self.q_rad_out = np.mean(solver.q_outer)

        # heat taken up by each pass in this section, half of the channels belong to each pass
        per_pass = self.section_length[idx] * self.cooling_geometry.L_factor[idx] * self.cooling_geometry.n_channels / 2
        Q1, Q2 = float(solver.Q_c) * per_pass, float(solver.Q_c2) * per_pass

        Cp1 = self.coolant.Cpl if self.coolant.phase == 'l' else self.coolant.Cpg
        Cp2 = self.coolant2.Cpl if self.coolant2.phase == 'l' else self.coolant2.Cpg
        dp1, _ = self.pressure_drop(idx, 1)
        dp2, _ = self.pressure_drop(idx, 2)

        # pass 1 heats up and loses pressure in the march direction; the return pass comes from the other side, so marching 
        # against its flow it was colder and at higher pressure
        self.check_pressure(self.coolant.P - dp1, idx)
        self.coolant  =self.make_coolant(self.coolant.IDs, ws=self.coolant.ws, T=self.coolant.T + Q1 / (self.m_dot_coolant * Cp1), P=self.coolant.P - dp1)
        self.coolant2 = self.make_coolant(self.coolant2.IDs, ws=self.coolant2.ws, T=self.coolant2.T - Q2 / (self.m_dot_coolant * Cp2), P=self.coolant2.P + dp2)


    def check_pressure(self, P, idx):
        # the coolant has to arrive with a positive pressure, else the property calculation fails with an obscure error
        if P <= 0:
            raise ValueError('Coolant pressure drops to %.1f bar at x = %.1f mm: the pressure drop exceeds the inlet pressure. Use wider or taller '
                             'channels, a smaller helix angle or a higher inlet_pressure' % (P / 1e5, self.geometry[idx, 0] * 1e3))


    def march_order(self):
        # section indices in the order the coolant (of the first pass) passes them, and the contour length from the previous one
        n_sections = len(self.geometry[:,1])
        if self.settings2D.start_idx == -1:
            order = list(range(n_sections - 1, -1, -1))
        elif self.settings2D.start_idx == 0:
            order = list(range(n_sections))
        else:
            raise ValueError('Invalid starting point for cooling fluid, select -1 or 0 for nozzle or injector side, respectively')
        self.section_length = np.zeros(n_sections)
        for prev, idx in zip(order[:-1], order[1:]):
            self.section_length[idx] = np.hypot(self.geometry[prev,0] - self.geometry[idx,0], self.geometry[prev,1] - self.geometry[idx,1])
        return order


    def estimated_heat_load(self, T_wall=600.0):
        # rough total heat load [W] for a first guess of the return pass outlet temperature, from the hot gas side at a nominal wall temperature
        r, x = self.geometry[:,1], self.geometry[:,0]
        ds = np.concatenate(([0.0], np.hypot(np.diff(x), np.diff(r))))
        Q = 0.0
        for idx in range(len(r)):
            h, T_hg = self.heat_trans_coeff_gas(T_wall, idx)
            Q += max(h * (T_hg - T_wall), 0.0) * 2 * np.pi * r[idx] * ds[idx]
        return Q


    def run_two_pass(self, tolerance_T=0.02, tolerance_P=50.0, max_iterations=15):
        """[Summary]
        Two-pass cooling with alternating channels: half of the channels carry the coolant from the inlet end (start_idx) to the other end, 
        where a manifold turns it into the other half of the channels, which carry it back to the inlet end. Both passes exchange heat with 
        the wall in every section, and with each other through the ribs. The return pass state at the inlet end (its outlet) is unknown when 
        marching from there, so it is guessed and corrected until the return pass leaves the manifold at the temperature of the first pass 
        and at its pressure less the turnaround loss.
        """
        order = self.march_order()
        self.prefetch_meshes(order)
        inlet = self.coolant
        self.section_fields = {}

        Cp_in = inlet.Cpl if inlet.phase == 'l' else inlet.Cpg
        T2_out = inlet.T + self.estimated_heat_load() / (self.m_dot_coolant * Cp_in)
        P2_out = inlet.P - 1e5
        history = []

        for iteration in range(1, max_iterations + 1):
            self.log('TWO-PASS iteration %d: return pass outlet guess %.2f K, %.4f bar' % (iteration, T2_out, P2_out / 1e5))
            self.coolant  = self.make_coolant(inlet.IDs, ws=inlet.ws, T=inlet.T, P=inlet.P)
            self.coolant2 = self.make_coolant(inlet.IDs, ws=inlet.ws, T=T2_out, P=P2_out)

            for i, idx in enumerate(order):
                self.section_number = i + 1
                self.section_heat_flux_two_pass(idx)
                self.record_output(idx)

            # turnaround manifold: dp = K * rho v^2 / 2, K = turnaround_loss (default 1.5, of the order of sharp 180 degree
            # return bends in Idelchik 1994; the manifold geometry is not modelled)
            rho1 = self.coolant.rhol if self.coolant.phase == 'l' else self.coolant.rhog
            self.dp_turnaround = self.turnaround_loss * 0.5 * rho1 * self.v_coolant**2
            r_T = self.coolant2.T - self.coolant.T
            r_P = self.coolant2.P - (self.coolant.P - self.dp_turnaround)
            self.log('TWO-PASS iteration %d: return pass at the turnaround off by %.3f K and %.1f Pa' % (iteration, r_T, r_P))
            if abs(r_T) < tolerance_T and abs(r_P) < tolerance_P:
                break

            # shooting method: the pressure error shifts one to one (pressure drops barely depend on the pressure level), secant for T
            P2_out -= r_P
            if P2_out <= 0:
                raise ValueError('Coolant pressure drop exceeds the inlet pressure: both passes and the turnaround need about %.1f bar, the inlet '
                                 'pressure is %.1f bar. Use wider or taller channels, a smaller helix angle or a higher inlet_pressure'
                                 % ((inlet.P - P2_out) / 1e5, inlet.P / 1e5))
            history.append((T2_out, r_T))
            if len(history) >= 2 and abs(history[-1][0] - history[-2][0]) > 1e-9:
                slope = (history[-1][1] - history[-2][1]) / (history[-1][0] - history[-2][0])
                T2_out -= r_T / slope if abs(slope) > 0.05 else r_T
            else:
                T2_out -= r_T
        else:
            self.log('TWO-PASS not converged after %d iterations: return pass off by %.3f K and %.1f Pa at the turnaround' % (max_iterations, r_T, r_P))

        self.two_pass_iterations = iteration
        self.out.dp_turnaround = self.dp_turnaround
        self.out.two_pass_iterations = iteration
        self.out.two_pass_residual = (r_T, r_P)
        self.out.write_csv()


    def record_output(self, idx):
        # write the state of the current section to the output class
        self.out.halpha[idx]    = self.halpha
        self.out.halpha_c[idx]  = self.halpha_c
        self.out.q_rad[idx]     = self.q_rad
        self.out.q[idx]         = self.q
        self.out.T_wall_i[idx]  = self.T_wall_i

        self.out.T_c[idx]       = np.asarray(self.coolant.T).item() if hasattr(self.coolant.T, '__iter__') else self.coolant.T
        self.out.P_c[idx]       = np.asarray(self.coolant.P).item() if hasattr(self.coolant.P, '__iter__') else self.coolant.P

        self.out.Re[idx]	    = self.Re
        self.out.T_hg[idx]	    = self.T_hg 
        self.out.T_wall_c[idx]  = self.T_wall_c
        self.out.v_coolant[idx] = self.v_coolant

        if self.passes == 2:
            self.out.T_c2[idx]       = float(self.coolant2.T)
            self.out.P_c2[idx]       = float(self.coolant2.P)
            self.out.Re2[idx]        = self.Re2
            self.out.v_coolant2[idx] = self.v_coolant2
            self.out.halpha_c2[idx]  = self.halpha_c2
            self.out.T_wall_c2[idx]  = self.T_wall_c2


    def run(self):
        if self.passes == 2:
            return self.run_two_pass()

        # arrays for section length and area in between the 2D sections solved in the thermal sim 
        self.section_length     = np.ndarray(len(self.geometry[:,1]))

        # generate the section meshes in the background, in the order the coolant passes the sections
        n_sections = len(self.geometry[:,1])
        order = range(n_sections - 1, -1, -1) if self.settings2D.start_idx == -1 else range(n_sections)
        self.prefetch_meshes(order)

        # determine the section length and inner chamber surface area in each section
        for i in range(len(self.geometry[:,1])):
        # section number 0 at the injector!!!
			
            if self.settings2D.start_idx == -1:
				# for coolant entering at the bottom of the cooling channels, use idx as inverse of i 
                idx = len(self.geometry[:,1]) - i - 1
			
            elif self.settings2D.start_idx == 0:
				# in case coolant starts at injector side
                idx = i
			
            else:
                raise ValueError('Invalid starting point for cooling fluid, select -1 or 0 for nozzle or injector side, respectively')

            if i == 0:
                self.section_length[idx] = 0
                
            else:
                # contour length from the previously solved section, which lies downstream (idx+1) for coolant entering at
                # the nozzle and upstream (idx-1) for coolant entering at the injector
                prev  = idx + 1 if self.settings2D.start_idx == -1 else idx - 1
                x_len = (self.geometry[prev,0] - self.geometry[idx,0])**2
                y_len = (self.geometry[prev,1] - self.geometry[idx,1])**2
                
                self.section_length[idx] = np.sqrt(x_len + y_len)

            # solve section heat transfer and temperature field and update coolant properties
            self.section_number = i + 1                   # position in the order the coolant passes the sections
            self.section_heat_flux(idx)

            # write to output class
            self.record_output(idx)

        # write to file in output folder
        self.out.write_csv()


