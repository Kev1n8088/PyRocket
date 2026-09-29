#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3			                #	 
#                                                                   #
#####################################################################

import numpy as np
from matplotlib import pyplot as plt


def helix_angle_from_pitch(pitch_deg_per_mm, radius_m, cos_beta=1.0):
    """
    Local helix angle [deg] from the meridian for a constant twist rate pitch_deg_per_mm [deg per mm of axial distance] at the
    centreline radius radius_m [m]. An axial step dx is dx/cos(beta) along the meridian and r*pitch*dx around, so
    tan(alpha) = r * pitch * cos(beta).
    """
    circumferential_per_meridional = pitch_deg_per_mm * np.pi / 180.0 * (radius_m * 1e3) * cos_beta
    return np.degrees(np.arctan(circumferential_per_meridional))


class ChamberGeometry():
    """[Summary]
    Chamber contour: cylinder, arc r_2, cone phi_conv, arc r_1 into the throat, arc r_n out of it, then the divergent section as
    a thrust-optimised parabola (Rao 1960), either x(y) with its length set by phi_div and phi_e ('parabola') or a quadratic
    Bezier with length L_e (Newlands 2017), as in the RPA software ('bezier'). r_1, r_2 follow RPA naming. Full nozzles only.
    """
    def __init__(self, D_c, D_t, D_e, L_cyl, r_2, r_1, r_n, phi_conv, phi_div, phi_e, step_size=0.001, nozzle_type='parabola', L_e=None, step_size_throat=None, throat_zone=None):
        self.D_c = D_c                                              # chamber diameter [m]
        self.D_t = D_t                                              # throat diameter [m]
        self.D_e = D_e                                              # exit diamter [m]
        self.L_cyl = L_cyl                                          # cylindircal chamber length [m]
        self.r_2 = r_2                                              # converging section inlet radius, R2 in RPA [m]
        self.r_1 = r_1                                              # beginning of throat radius, R1 in RPA [m]
        self.r_n = r_n                                              # end of throat radius [m]
        self.phi_conv = phi_conv * np.pi / 180                      # convergence angle [deg]    
        self.phi_div = phi_div * np.pi / 180                        # divergence angle [deg]
        self.phi_e = phi_e * np.pi / 180                            # exit angle of parabolic nozzle
        self.expansion_ratio = D_e**2 / D_t**2                      # expansion ratio (Ae/At)
        self.dx = step_size                                         # step size for the discretisation [-]
        self.nozzle_type = nozzle_type                              # divergent section contour: 'parabola' or 'bezier'
        self.L_e = L_e                                              # nozzle length from throat to exit, only used for 'bezier' [m]
        self.step_size_throat = step_size_throat                    # finer step size around the throat [m], optional
        self.throat_zone = throat_zone                              # distance from the throat with the finer step size [m], blends back to step_size over the same distance

        if (step_size_throat is None) != (throat_zone is None):
            raise ValueError('step_size_throat and throat_zone must be given together')

        if nozzle_type not in ('parabola', 'bezier'):
            raise ValueError('Invalid nozzle type. Select: "parabola" or "bezier"')
        if nozzle_type == 'bezier' and L_e is None:
            raise ValueError('L_e (throat to exit length) must be specified for a bezier nozzle')

    def contour(self):
        # generate chamber countour x, y coordiantes
        if self.step_size_throat is not None:
            self.refined_contour()
            return
        
        # Cylindrical section
        x_cyl = np.arange(0, self.L_cyl, self.dx)
        y_cyl = np.ones(len(x_cyl)) * self.D_c / 2

        # Converging section radius (at least 3 points)
        n1 = max([round(self.r_2 * self.phi_conv / self.dx), 3])     # number of discretisations in the radius
        t = np.linspace(0, self.phi_conv, n1, endpoint=True)
        
        # x and y coordinates of the converging section radius
        L_r1 = self.r_2 * np.sin(self.phi_conv)
        x_B = self.L_cyl + self.r_2 * np.sin(t)
        y_B = (self.D_c / 2 - self.r_2) + self.r_2 * np.cos(t)
        

        # x and y coordinates of linear converging section
        L_con = ((self.D_c / 2 - self.r_2 + self.r_2 * np.cos(self.phi_conv)) - (self.D_t / 2 + self.r_1 - self.r_1 * np.cos(-self.phi_conv))) / np.tan(self.phi_conv)
        x_C = np.arange((self.L_cyl + L_r1 + self.dx), (self.L_cyl + L_r1 + L_con), self.dx)
        y_C = (self.D_c / 2 - self.r_2 + self.r_2 * np.cos(self.phi_conv)) - np.tan(self.phi_conv) * np.arange(self.dx, L_con, self.dx)


        # Converging radius up to throat (at least 3 points)
        n2 = max([round(self.r_1 * self.phi_conv / self.dx), 3])      # number of discretisations in the radius
        t = np.linspace(- self.phi_conv, 0, n2, endpoint=False)

        # x and y coordiantes of converging radius
        L_r21 = self.r_1 * np.sin(self.phi_conv)
        x_D = self.L_cyl + L_r1 + L_con + L_r21 + self.r_1 * np.sin(t)
        y_D = self.D_t / 2 + self.r_1 - self.r_1 * np.cos(t)


        # Throat radius (at least 3 points)
        n3  = max([round(self.r_n * self.phi_div / self.dx), 3])      # number of discretisations in the radius
        t = np.linspace(0, self.phi_div, n3, endpoint=True)

        # x and y coordiantes of throat radius
        L_r22 = self.r_n * np.sin(self.phi_div)
        x_E = self.L_cyl + L_r1 + L_con + L_r21 + self.r_n * np.sin(t)
        y_E = self.D_t / 2 + self.r_n - self.r_n * np.cos(t)

        # divergent section from the end of the throat arc
        P1x = x_E[-1]
        P1y = y_E[-1]
        # remove repeat points
        x_E = x_E[:-1]
        y_E = y_E[:-1]

        if self.nozzle_type == 'parabola':
            # parabola x = a1 y^2 + a2 y + a3 with wall angle phi_div at the start and phi_e at the exit (Rao 1960)
            a2  = (np.tan(self.phi_e)**(-1) - np.tan(self.phi_div)**(-1)*self.D_e*0.5 / P1y) * (1 - self.D_e*0.5 / P1y)**(-1)
            a1  = (np.tan(self.phi_div)**(-1) - a2) / (2 * P1y)
            a3  = P1x - a1 * P1y**2 - a2 * P1y
            P2x = a1 * (self.D_e*0.5)**2 + a2 * (self.D_e*0.5) + a3

            # number of points along exit parabola (at least 3)
            n4 = max(round((P2x - P1x)/self.dx), 3)

            y_F = np.linspace(P1y, self.D_e/2, n4, endpoint=True)
            x_F = a1 * y_F**2 + a2 * y_F + a3

        else:
            # quadratic Bezier from the end of the throat arc (N) to the exit (E), control point Q at the intersection of the wall
            # tangents at phi_div and phi_e (Newlands 2017)
            Ex  = self.L_cyl + L_r1 + L_con + L_r21 + self.L_e
            Ey  = self.D_e / 2
            m_N = np.tan(self.phi_div)
            m_E = np.tan(self.phi_e)
            Qx  = ((Ey - m_E * Ex) - (P1y - m_N * P1x)) / (m_N - m_E)
            Qy  = P1y + m_N * (Qx - P1x)

            if not (P1x < Qx < Ex and P1y < Qy < Ey):
                raise ValueError('Bezier nozzle not possible with the given L_e, phi_div and phi_e. The wall tangents do not intersect between throat radius and exit')

            # number of points along exit bezier (at least 3)
            n4 = max(round((Ex - P1x)/self.dx), 3)

            t = np.linspace(0, 1, n4, endpoint=True)
            x_F = (1 - t)**2 * P1x + 2 * t * (1 - t) * Qx + t**2 * Ex
            y_F = (1 - t)**2 * P1y + 2 * t * (1 - t) * Qy + t**2 * Ey

        self.x = np.concatenate((x_cyl, x_B, x_C, x_D, x_E, x_F), axis=0)
        self.y = np.concatenate((y_cyl, y_B, y_C, y_D, y_E, y_F), axis=0)

        # create 2 dimensional array containing all coordinates
        xy = [[self.x[i], self.y[i]] for i in range(len(self.x))]
        self.geometry = np.vstack(xy)


    def refined_contour(self):
        # contour with the finer step size around the throat: a fine contour resampled along its length with the local step size
        coarse, throat_step = self.dx, self.step_size_throat
        self.dx, self.step_size_throat = min(coarse, throat_step) / 10, None
        try:
            self.contour()
        finally:
            self.dx, self.step_size_throat = coarse, throat_step
        x_f, y_f = self.x, self.y
        s_f = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_f), np.diff(y_f)))))    # arc length of the fine contour
        x_t = x_f[np.argmin(y_f)]
        s_t = s_f[np.argmin(y_f)]

        def step(x):
            # finer step size within throat_zone of the throat, blending linearly back to step_size over the same distance
            blend = np.clip(2.0 - abs(x - x_t) / self.throat_zone, 0.0, 1.0)
            return self.dx + (self.step_size_throat - self.dx) * blend

        s = [0.0]
        while s[-1] < s_f[-1]:
            s.append(s[-1] + step(np.interp(s[-1], s_f, x_f)))
        s[-1] = s_f[-1]
        s = np.array(s)
        # keep the exact throat point, dropping points that would be much closer to it than the local step size
        s = s[np.abs(s - s_t) > 0.3 * step(x_t)]
        s = np.sort(np.append(s, s_t))
        if s[-1] - s[-2] < 0.3 * self.dx:
            s = np.delete(s, -2)

        self.x = np.interp(s, s_f, x_f)
        self.y = np.interp(s, s_f, y_f)
        self.geometry = np.column_stack([self.x, self.y])


    def plot_contour(self, path=False):
        plt.plot(self.x*1e3, self.y*1e3, color='b')
        plt.plot(self.x*1e3, -self.y*1e3, color='b')
        plt.xlabel('x_coordinate [mm]')
        plt.ylabel('y_coordinate [mm]')
        plt.title('Chamber Contour')
        plt.axis('equal')
        plt.grid()
        if path != False:
            plt.savefig(path + '/chamber_contour.png', dpi=150)
            plt.close()
        else:
            plt.show()



class CoolingGeometry():
    """[Summary]
    Class to calculate the cooling channel shapes along the chamber contour. Currently limited to rectangular channels.
    The input parameters 'h_c', 'psi' and 't_w_i' can be input as functions of the axial chamber coordinate. 
    The location of thermocouples can be set to log the temperature of the 2D thermal sim at certain chamber locations. 
    This is meant for program validation purposes and can be disabled in config.py.
    """
    def __init__(self, chamber_geometry, h_c, psi, t_w_i, t_w_o, n_channels, helix_angle=0.0, pitch_deg_per_mm=None, channel_width=None, passes=1):
        self.geom       = chamber_geometry                                 # points along the chamber contour [x,y]
        self.n_channels = n_channels                                       # number of cooling channels, of all passes together
        self.passes     = passes                                           # 1, or 2 for two-pass cooling with alternating forward and return channels

        if passes not in (1, 2):
            raise ValueError('passes must be 1 or 2')
        if passes == 2 and n_channels % 2:
            raise ValueError('two-pass cooling needs an even number of channels, half of them in each pass')
        
        # some inputs can be functions of x coordinates 
        if type(h_c) == float:
            # check if input is a single float
            self.h_c = np.ones(len(self.geom[:,1])) * h_c               # height of the cooling channels [m]
        elif callable(h_c):
            # check if h_c is a function pointer
            self.h_c = h_c(self.geom[:,0])                              # height of cooling channel as function of x 
        else: 
            raise ValueError('h_c must be a float or a function of x coordinate')
        
        if type(psi) == float:
            # check if phi is a single float
            self.psi = np.ones(len(self.geom[:,1])) * psi                # ratio of coverage of the cooling channels to solid wall
        elif callable(psi):
            # check if psi is a function pointer
            self.psi = psi(self.geom[:,0])                               # psi as a fucntion of chamber length if it is indeed  a fucntion
        else: 
            raise ValueError('psi must be a float or a function of x coordinate')

        if type(t_w_i) == float:
            # check if inner wall thickness is a single float
            self.t_w_i = np.ones(len(self.geom[:,1])) * t_w_i              #  inner chamber wall thickness [m]  
        elif callable(t_w_i):
            # check if psi is a function pointer
            self.t_w_i = t_w_i(self.geom[:,0])                               # t_w_i as a fucntion of chamber length if it is indeed  a fucntion
        else: 
            raise ValueError('t_w_i must be a float or a function of x coordinate')
        
        self.t_w_o = np.ones(len(self.geom[:,1])) * t_w_o              # outer chamber wall thickness [m] 

        self._helix_angle_input = helix_angle                           # helix angle [deg], float or callable of x
        self._channel_width     = channel_width                         # channel width normal to the flow at the channel bottom [m], float or callable of x; overrides psi if set
        self.pitch_deg_per_mm   = pitch_deg_per_mm                      # constant helix pitch [deg/mm]; overrides helix_angle if set

        self.psi_c = 2 * np.pi * self.psi / n_channels                 # radial section of single cooling channel 
        self.psi_w = 2 * np.pi * (1 - self.psi) / n_channels           # raidal section of single uncooled wall segment


    def channel_geometry(self):
        # generate the cooling channel geometry; area, hydraulic diameter and wall thickness
        self.r_i = self.geom[:,1] + self.t_w_i                                      # inner radius of cooling channels 
        self.r_o = self.geom[:,1] + self.t_w_i + self.h_c                           # outer radius of cooling channels 
        R_coil   = self.r_i + self.h_c / 2                                          # radius of the channel centreline [m]

        # local slope of the chamber wall relative to the engine axis, cos(beta) = dx/ds along the meridian
        self.cos_beta = 1.0 / np.sqrt(1.0 + np.gradient(self.geom[:,1], self.geom[:,0])**2)

        # helix angle per station [rad], measured from the local meridian; pitch_deg_per_mm overrides helix_angle if both are set
        if self.pitch_deg_per_mm is not None:
            self.alpha = np.radians(helix_angle_from_pitch(self.pitch_deg_per_mm, R_coil, self.cos_beta))
        else:
            if isinstance(self._helix_angle_input, (float, int)):
                self.alpha = np.radians(np.ones(len(self.geom[:, 1])) * self._helix_angle_input)
            elif callable(self._helix_angle_input):
                self.alpha = np.radians(self._helix_angle_input(self.geom[:, 0]))
            else:
                raise ValueError('helix_angle must be a float or a callable of x coordinate')
        cos_alpha = np.cos(self.alpha)

        # flow path length factor: channel path / meridional section length (= 1/cos(alpha), 1.0 for axial channels)
        self.L_factor = 1.0 / cos_alpha

        # channel width normal to the flow given directly: fill factor that gives this width at the channel bottom
        if self._channel_width is not None:
            if isinstance(self._channel_width, (float, int)):
                w_c = np.ones(len(self.geom[:,1])) * self._channel_width
            elif callable(self._channel_width):
                w_c = np.asarray(self._channel_width(self.geom[:,0]), dtype=float)
            else:
                raise ValueError('channel_width must be a float or a function of x coordinate')
            self.psi = w_c * self.n_channels / (2 * np.pi * self.r_i * cos_alpha)
            if np.any(self.psi >= 1):
                raise ValueError('channel_width leaves no rib: fill factor reaches %.3f, reduce the width or the number of channels' % np.max(self.psi))
            self.psi_c = 2 * np.pi * self.psi / self.n_channels
            self.psi_w = 2 * np.pi * (1 - self.psi) / self.n_channels

        # psi is measured in a cut normal to the engine axis; normal to a helical channel all widths are smaller by cos(alpha). The
        # wall temperature field is invariant along the channel, so the 2D section is solved in the plane normal to it
        self.psi_c_n = self.psi_c * cos_alpha                                      # radial section of single cooling channel normal to the flow
        self.psi_w_n = self.psi_w * cos_alpha                                      # radial section of single wall segment normal to the flow

        # flow area and wetted perimeter normal to the flow
        self.A_c = self.psi_c / (self.psi_c + self.psi_w) * np.pi * (self.r_o**2 - self.r_i**2) * cos_alpha                                                     # cooling channel area [m^2]
        self.U_c_top = self.psi_c_n * self.r_o										# circumference of top cooling channels 
        self.U_c_bottom = self.psi_c_n * self.r_i									# circumference of bottom cooling channels 
        self.U_c_side = self.h_c * 2												# circumference of side cooling channels 
        self.U_c = (self.U_c_top + self.U_c_bottom + self.U_c_side)*self.n_channels # cooling channel circumference [m]
        self.D_h = 4 * self.A_c / self.U_c											# cooling channel hydraulic diameter [m]

        # centreline curvature from winding around the chamber: normal curvature of the surface of revolution in the direction alpha,
        # Euler's theorem with the parallel-circle curvature cos(beta)/R from Meusnier's theorem (do Carmo 1976, Ch. 3).
        # On a cylinder this is the helix curvature sin(alpha)^2/R. The centre lies towards the axis, so the hot bottom wall is the
        # convex side. The meridional curvature of the contour itself (e.g. at the throat) is not modelled. np.inf for axial channels
        kappa = self.cos_beta * np.sin(self.alpha)**2 / R_coil
        self.R_curve = np.full(len(kappa), np.inf)
        self.R_curve[kappa > 1e-12] = 1.0 / kappa[kappa > 1e-12]
		

    def set_thermocouples(self, x, r):
        # set a number of thermocouples for validation purposes
        # loc refers to which boundary on the 2D mesh the thermocouple rests on 
        # input arrays for lnghtwise (x) and radial coordinates (r) 
        # creates an array of dictionaries for all thermocouples in the cooridnate system of the 2D sections
        thermocouples = []

        def closest_node(node):
        # find closest index to thermocouple location
            dist = (self.geom[:,0] - node)**2
            idx = np.argmin(dist)                    # calculates clostes chamber index to TC location
            return idx

        for i in range(len(r)):
            idx = closest_node(x[i])                # axial location index

            section_x = r[i] * np.sin((self.psi_c_n[idx] + self.psi_w_n[idx])/2)
            section_y = r[i] * np.cos((self.psi_c_n[idx] + self.psi_w_n[idx])/2)
            
            thermocouples.append([idx, section_x, section_y])

        self.thermocouples = np.vstack(thermocouples)

        

if __name__ == "__main__":
    # test functionality of the Chamber and Cooling Geometry Classes

    D_c = 69.05e-3
    D_t = 24.14e-3
    D_e = 50.7e-3
    L_cyl = 74.77e-3
    r_1 = 10e-3
    r_2 = 72e-3
    r_n = 5e-3
    phi_conv = 30
    phi_div = 19
    phi_e  = 14

    cg = ChamberGeometry(D_c, D_t, D_e, L_cyl, r_2, r_1, r_n, phi_conv, phi_div, phi_e, step_size=0.003)
    cg.contour()
    cg.plot_contour()

    n = 8 
    h_c = 3e-3
    phi = 1/3
    t_w_i = 1e-3
    t_w_o = 1e-3

    c = CoolingGeometry(cg.geometry, h_c, phi, t_w_i, t_w_o, n)
    c.channel_geometry()


