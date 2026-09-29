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

import matplotlib.pyplot as plt
import numpy as np

import Units


class Plotting1D():
    def __init__(self, save_path, geometry, save=True, show=False):
        self.save_path = save_path
        self.geometry = geometry
        self.save = save
        self.show = show
        self.data = np.genfromtxt(save_path+'/sim_data.csv', delimiter=",", skip_header=1)


    def two_pass(self):
        # sim_data.csv holds return pass columns with values for two-pass cooling
        return self.data.shape[1] > 14 and np.all(np.isfinite(self.data[:,13]))

    def temperature_plot(self):
        plt.plot(self.geometry[:,0]*1e3, Units.temperature(self.data[:,7]), color='b', label='Coolant bulk temperature')
        if self.two_pass():
            plt.plot(self.geometry[:,0]*1e3, Units.temperature(self.data[:,13]), color='b', linestyle=':', label='Coolant bulk temperature, return pass')
        plt.plot(self.geometry[:,0]*1e3, Units.temperature(self.data[:,6]), color='r', linestyle='--', label='Inner wall temperature')
        plt.xlabel('x coordiante [mm]')
        plt.ylabel('Temperature [' + Units.temperature_unit() + ']')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/temperature', dpi=150)
            plt.close()
            
        if self.show:
            plt.show()
        

    def pressure_plot(self):
        plt.plot(self.geometry[:,0]*1e3, self.data[:,8]/1e5, color='b', label='Coolant pressure')
        if self.two_pass():
            plt.plot(self.geometry[:,0]*1e3, self.data[:,14]/1e5, color='b', linestyle=':', label='Coolant pressure, return pass')
        plt.xlabel('x coordiante [mm]')
        plt.ylabel('Pressure [bar]')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/pressure', dpi=150)
            plt.close()
        
        if self.show:
            plt.show()
        

    def heat_transfer_coeff_plot(self):
        plt.plot(self.geometry[:,0]*1e3, self.data[:,2], color='b', label='$h_{alpha}$ gas')
        plt.plot(self.geometry[:,0]*1e3, self.data[:,3], color='r', linestyle='--',  label='$h_{alpha}$ coolant')
        plt.xlabel('x coordiante [mm]')
        plt.ylabel('Heat transfer coefficient [W/m^2/K]')
        plt.legend(loc='best')
        
        if self.save:
            plt.savefig(self.save_path + '/halpha', dpi=150)
            plt.close()

        if self.show:
            plt.show()
    

    def heat_flux_plot(self):
        plt.plot(self.geometry[:,0]*1e3, self.data[:,4]/1e6, color='b', label='q radiation')
        plt.plot(self.geometry[:,0]*1e3, self.data[:,5]/1e6, color='r', linestyle='--', label='q total')
        plt.xlabel('x coordiante [mm]')
        plt.ylabel('Heat flux [MW/m^2]')
        plt.legend(loc='best')
        
        if self.save:
            plt.savefig(self.save_path + '/heat_flux', dpi=150)
            plt.close()
        
        if self.show:
            plt.show()
    

    def reynolds_plot(self):
        plt.plot(self.geometry[:,0]*1e3, self.data[:,9], color='b', label='Coolant Re')
        plt.xlabel('x coordiante [mm]')
        plt.ylabel('Re')
        plt.legend(loc='best')

        if self.save:
            plt.savefig(self.save_path + '/reynolds', dpi=150)
            plt.close()
    
        if self.show:
            plt.show()
        

    def stress_plot(self, stress, longitudinal='shell'):
        # hot wall stresses against the yield strength (top) and the stress margin (bottom), see Stress.wall_stress
        x = self.geometry[:,0]*1e3
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.4))
        ax1.plot(x, stress['sigma_t']/1e6, color='b', label='Tangential')
        ax1.plot(x, stress['sigma_l']/1e6, color='g', label='Longitudinal (' + ('hot to coolant-side wall' if longitudinal == 'shell' else 'across the hot wall') + ')')
        ax1.plot(x, stress['sigma_vm']/1e6, color='r', label='Von Mises')
        ax1.plot(x, stress['yield']/1e6, color='k', linestyle='--', label='Yield strength at the hot wall temperature')
        ax1.set_ylabel('Stress [MPa]')
        ax1.legend(loc='best', fontsize=8)

        i = int(np.argmin(stress['margin']))
        ax2.plot(x, stress['margin'], color='r', label='Stress margin (yield / von Mises - 1)')
        ax2.axhline(0, color='k', linewidth=0.8)
        ax2.plot(x[i], stress['margin'][i], 'o', color='r')
        ax2.annotate('min %.3f' % stress['margin'][i], (x[i], stress['margin'][i]), textcoords='offset points', xytext=(6, -12), fontsize=8)
        ax2.set_xlabel('x coordinate [mm]')
        ax2.set_ylabel('Margin [-]')
        ax2.legend(loc='best', fontsize=8)
        f.tight_layout()

        if self.save:
            plt.savefig(self.save_path + '/stress', dpi=150)
            plt.close()

        if self.show:
            plt.show()


def multi_plot(x, data1, data2, data3, data4, label1, label2, label3, label4):
    f, axes = plt.subplots(4, 1)
    #f.subplots_adjust(hspace = 0)

    axes[0].plot(x, data1)
    axes[0].set_ylabel(label1)

    axes[1].plot(x, data2)
    axes[1].set_ylabel(label2)

    axes[2].plot(x, data3)
    axes[2].set_ylabel(label3)

    axes[3].plot(x, data4)
    axes[3].set_ylabel(label4)

    plt.xlabel('x coordiante [m]')
    plt.show()


if __name__ == "__main__":
    from CaseSetup import load_config

    config = load_config('config.py')
    plot = Plotting1D(save_path=config.save_path, geometry=config.geometry, save=True, show=True)
    plot.temperature_plot()
    plot.pressure_plot()
    plot.heat_transfer_coeff_plot()
    plot.heat_flux_plot()
    plot.reynolds_plot()
