###############################################################################
#                         Liquid Film Cooling Model					          #
# Basic 0D analytical film cooling model based energy balance by Shine et al. #
#                                                                   	      #
# Creator:  Joanthan Neeser                                        		      #
# Date:     06.02.2022                                             		      #
# Version:  1.1 														      #                                      
###############################################################################


import numpy as np
from scipy import optimize


class FilmCooling():
	def __init__(self, cea, coolant, injection_velocity, injector_diameter, chamber_diameter, m_dot, m_dot_coolant):
		self.cea       = cea
		self.coolant   = coolant
		self.v_c       = injection_velocity
		self.d_inj     = injector_diameter
		self.d_ch      = chamber_diameter
		self.m_dot     = m_dot
		self.m_dot_c   = m_dot_coolant
		
		# coolant flow Reynolds number
		self.Re_c    = self.v_c * self.d_inj * self.coolant.rho / self.coolant.mu
		
		# mean combustion gas velocity, Reynolds number and mass flux
		self.v_g     = self.m_dot / self.cea.rho / (np.pi * self.d_ch**2 / 4)
		self.Re_g    = self.v_g * self.d_ch * self.cea.rho / self.cea.mu
		self.G_m     = self.cea.rho * (self.v_g - self.v_c)
		
		# coolant boiling temperature and heat of evaporation taken from the dominant specie
		idx = np.argmax(self.coolant.ws)						# index of dominant specie by mass fraction
		self.T_sat   = self.coolant.Tbs[idx]								# boiling temperature of dominant specie
		
		# mass averaged entalpy of evaporation [J/kg]
		self.dH_evap = np.sum([self.coolant.Hvap_Tbs[i] * self.coolant.ws[i] for i in range(len(self.coolant.ws))])
		
		
	def coolant_flow(self):
		# liquid mass flow per circumference 
		self.gamma_c = self.m_dot_c / (np.pi * self.d_ch)
		
		# entrained liquid fraction E, Sawant et al. (2008) correlation as used by Shine et al. (2012)
		We = self.cea.rho * self.v_g**2 * self.d_ch / self.coolant.sigma * ((self.coolant.rho - self.cea.rho)/self.cea.rho)**0.25
		a  = 2.31e-4 * self.Re_c**(-0.35)
		
		Re_c_film = 250 * np.log(self.Re_c) - 1265
		Em = 1 - Re_c_film / self.Re_c
		E = Em * np.tanh(a * We**1.25)	
		
		# liquid mass flow per circumference accounting for loss of entrained flwo
		self.gamma = self.gamma_c * (1 - E)
	
	def radiation(self):
		# same empirical H2O/CO2 radiation as RegenCooling.radiation, at chamber conditions (no source given in the original PyRocket)

		p_h2o = self.cea.mole_fractions[1]['H2O'][0] * self.cea.chamber_pressure
		# radiative heat flux of water molecules 
		self.q_rad = 5.74 * (p_h2o / 1e5 * 0.5 * self.d_ch)**0.3 * (self.cea.Tc/100)**3.5

		# check if CO2 is present in the exhaust
		if '*CO2' in self.cea.mole_fractions[1]:
			p_co2 = self.cea.mole_fractions[1]['*CO2'][0] * self.cea.chamber_pressure
			# radiative heat flux of co2 molecules 
			self.q_rad += 4 * (p_co2/1e5 * 0.5 * self.d_ch)**0.3 * (self.cea.Tc/100)**3.5
		
		
		
	def stanton_number(self):
		# Darcy friction factor, McKeon et al. (2005) Eq. 3.1 (log10, fitted for 3e5 <= Re <= 1.8e7), and the Fanning factor f = lambda/4
		# as in Shine (2013), Eq. 5.6
		func = lambda lam: 1.930 * np.log10(self.Re_g * np.sqrt(lam)) - 0.537 - 1 / np.sqrt(lam)
		self.lam = optimize.brentq(func, 1e-4, 1.0)                    # func increases monotonically with lambda
		fr = self.lam / 4
		
		# free-stream turbulence correction K_t = 1 + 4 e_t (Pletcher 1988, as used by Shine et al. 2012)
		e_t = 0.1		# highest value reported for small GOx/H2 engines
		K_t = 1 + 4 * e_t
		
		self.dH_fg_star = self.dH_evap + self.coolant.Cp * (self.T_sat - self.coolant.T)
	
		# Stanton number without film, Friend & Metzner (1958) analogy with the Fanning factor f (Shine 2013, Eq. 5.7)
		self.St = 0.5 * fr * (1.2 + 11.8 * np.sqrt(0.5*fr) * (self.cea.Pr - 1) * (self.cea.Pr**(-1/3)))**(-1)
		self.h_alpha = self.St * self.G_m * self.cea.Cp * K_t
		self.F_St = self.cea.Cp / self.dH_fg_star * ((self.cea.Tc - self.T_sat) + self.q_rad / self.h_alpha)
		
		# Stanton number reduction by transpiration of the vapour: Couette-flow solution St/St0 = ln(1 + B k_m)/(B k_m), B = F/St
		# (Kays & Crawford 2005), molecular weight correction k_m = (M_g/M_c)^0.6 (Meinert & Huhn 2001) (Shine 2013, Eq. 5.10 to 5.12)
		k_m = (self.cea.MW / self.coolant.MW)**0.6
		self.St_St0 = np.log(1 + self.F_St * k_m) / (self.F_St * k_m)
		
		# evaporation rate per unit area from the energy balance at the film surface
		self.h_alpha_film = self.St * self.St_St0 * self.G_m * self.cea.Cp * K_t
		self.m_v = (self.q_rad + self.h_alpha_film * (self.cea.Tc - self.T_sat)) / self.dH_fg_star
		
			
	def film_length(self):
		self.radiation()
		self.coolant_flow()
		self.stanton_number()
		
		if self.m_dot_c == 0.0:
			self.liquid_film_length = -1
		else:
			self.liquid_film_length = self.gamma / self.m_v
