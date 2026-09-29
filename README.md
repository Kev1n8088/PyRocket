# PyRocket
PyRocket is a thermal simulaton program for regeneratively cooled rocket engine thrust chambers. It is designed as a reasonably accurate predictive 
tool to assist in design and allow for integration into basic shape optimisation. The program relies on the use of Nusselt number correlations for 
deterimining the heat transfer coefficients from the hot combustion gases, as well as into the cooling fluid. The temperature inside the chamber 
wall is then determined using a 2D transient thermal simulation based on a Finite Volume method. This is repeated for a set number of sections 
along the chamber contour. The predicted heat flux to the cooling fluid in each section is used to calculate a temperature increase of the cooling fluid. 
Basic wall friction estimates are also used to track coolant pressure drop in the channels. 


## Installation
PyRocket relies on several external libraries, as well as GMSH. The installation of dependencies can be done using the install_dependencies.sh file. 


### Installation on WINDOWS 10
The rocektcea library relies on a FORTRAN compiler. It can be installed natively on Windows, but in my 
expereince it is a lot easier to get it to run in a Linux subsystem. 

```bash
    1. as admin, open PowerShell
    2. run: 'wsl --install'
    3. run: 'wsl --install -d Ubuntu-20.04'
    4. run: 'python3 --version' to check the python install (should be above V3.8)

    5. clone PyRocket Files into a folder (locally, not on any network)
    6. open wsl and naviagate to the folder with the PyRocket files 
    7. run: 'sudo bash install_dependencies.sh'
```


### Installation on Ubuntu
The install_dependencies.sh file is set up for installation on Ubuntu, but can easily be altered for use in other Linux distributions. 
```bash
    1. clone PyRocket Files into a folder
    2. open a terminal window and naviagate to the folder with the PyRocket files  
    3. run: 'sudo bash install_dependencies.sh'
```


## Verify Installation
In order to verify that all pacakges were installed correctly, run 'SectionThermalSim.py' in your terminal window. 
```bash
    python3 SectionThermalSim.py
```
This will run a single transient thermal simulation. Do not alter the contents of the config.py file before running. The expected ouput is the maximum temperature
at every time step printed in the terminal window. Additionally a directory called 'TestSectionThermalSim' is created, where an image of the temperature profile 
of the test section is saved. This unit test can be repeated after altering the config.py file for each use case to verify the stability of your new time step and 
mesh resolution settings. 


## Using PyRocket
PyRocket is designed to be used with nearly arbitrary combinations of propellants, cooling fluids and chamber materials. All inputs are changed in config.py 
New propellants can be set in PropLib.py, with new matierals added in MaterialLib.py. The selected coolant must be a fluid present in 
[thermo](https://thermo.readthedocs.io/). A debug mode can be set in config.py to examine the results of the 2D transient simulation to verfiy the chosen mesh and time 
step settings. Run the full simulation by executing 'run.py' in your terminal window. By default every config in the configs folder is run, a single 
config can be run by passing its path.

```bash
    python3 run.py              # every config in configs/
    python3 run.py config.py    # a single config
```

### Browser interface
Single runs, batch runs and helix sweeps can also be started from a browser. The interface runs run.py and sweep.py in the background (same results 
as the command line), shows the progress of every case section by section, and displays the summary charts, key statistics and plots when a run 
is done. The Results tab browses earlier results. It only listens on this computer.

```bash
    python3 gui.py                # then open http://localhost:8765
    python3 gui.py --port 9000
```

### Multiple configs and summary
Config files only contain input parameters, the gas, coolant and geometry objects are built from them in CaseSetup.py. Any number of config files can be run 
at once, e.g. by placing copies of config.py in the configs folder. Each case is run in its own process and written to results/<config file name>/. 
After all cases are finished, a comparison of the critical statistics (maximum, median and throat wall temperature, coolant outlet temperature, 
coolant pressure drop, remaining pressure over Pc for the injector, peak heat flux and total heat load) is saved to results/summary.png and results/summary.csv. 
A failing case does not stop the other cases.

Every case also gets the hot wall stress margin (Stress.py, the RPE estimate described under Cooling optimisation) for materials with temperature 
dependent mechanical properties in MaterialLib.py (IN718, CuCr1Zr, AlSi10Mg): stress.png (stresses against the yield strength and the margin along 
the chamber), stress.csv, and the minimum margin with its location and stresses in summary.json and the summary. The longitudinal stress uses 
stress_longitudinal = 'shell' (default) or 'wall' from the config.

```bash
    python3 run.py other_folder/              # run every config in another folder
    python3 run.py configs/a.py configs/b.py  # run selected configs
    python3 run.py --jobs 4                   # run 4 cases in parallel, output goes to results/<case>/log.txt
    python3 run.py --summary-only             # rebuild the summary from existing results without running
    python3 run.py --out my_results           # use a different output folder
    python3 run.py --section-images           # keep the per-section temperature images (off in batch runs and sweeps)
    python3 run.py --logs                     # write the output of every case to results/<case>/log.txt
```

### Helix sweep
sweep.py finds the helix pitch (or angle) of one config with the lowest maximum inner wall temperature, with a maximum pitch/angle, a maximum coolant 
pressure drop, an optional maximum coolant-side wall temperature ('--max-coolant-wall-temp' in K, e.g. to stay below the coking limit of RP-1) and 
optional minimum channel and rib widths. The channel count and fill factor psi of the config are kept, so channels and ribs get narrower 
normal to the flow as the helix angle increases (by cos(alpha)); the width limits act on these widths and cap the search range before anything is simulated. 
The search runs an even grid from axial channels to the maximum and refines around the best point that meets the pressure drop limit. Results, a sweep 
plot and a summary of all points are written to results/sweep_<config>_<param>/. Finished points are reused when the sweep is repeated with an unchanged 
config (use --fresh to run them again).

```bash
    python3 sweep.py configs/rp1_lox.py --param pitch --max 3 --max-dp 10 --jobs 4
    python3 sweep.py configs/rp1_lox.py --param angle --max 50 --max-dp 8 --min-channel-width 0.8 --min-rib-width 0.8
```

### Cooling optimisation
optimize.py searches the cooling channel design of a config for the largest stress margin. Fixed from the config: chamber contour, propellants and 
mixture ratio, film cooling, coolant inlet (start_idx) and one or two passes. The hot wall thickness is fixed for all designs: '--wall' in mm, by 
default the config's t_w_i. Varied: wall material (IN718, AlSi10Mg, CuCr1Zr), channel count, channel width normal to the flow (constant along the 
chamber, or growing with the radius at a constant fill factor), channel height and a constant helix angle. Designs must respect the maximum helix angle, 
the minimum rib and channel width and channel height, the maximum coolant pressure drop, the maximum service temperature of the material, and 
optionally a maximum coolant-side wall temperature ('--max-coolant-wall-temp' in K, e.g. to stay below the coking limit of RP-1) and a minimum margin 
of the coolant wall to boiling.

The designs are evaluated with a fast model (FastCooling.py): the same cooling march as the 2D solver, but each section is solved on a small 
structured finite volume grid (harmonic-mean face conductances [Patankar 1980]) instead of the FiPy/gmsh mesh, with coolant properties from a pre-computed table. It runs a chamber in about 0.2 s 
(1 s two-pass) and agrees with the 2D solver within a few K. A thermal resistance network ('--section-model network', ribs as fins [Incropera et al. 2007, Sec. 3.6]) is faster still, but misses the 
sideways conduction to wide ribs in low conductivity walls such as IN718. A grid of designs is refined by a compass (pattern) search [Kolda et al. 2003] around the best ones, then 
the best designs (the best of each material first) are verified with the 2D solver, and the config lines of the best verified design are written out.

The stress margin follows Rocket Propulsion Elements [Sutton & Biblarz 2017, Ch. 8]: tangential stress (p_coolant - p_gas)/2 (w/t)^2 + E a q t / (2 (1 - nu) k), longitudinal 
stress E a dT, combined as von Mises (both compressive at the hot face) against the yield strength at the hot wall temperature. dT is the hot wall 
temperature less the coolant-side wall temperature, the hot wall being restrained by the cooled outer shell ('--longitudinal shell'), or the temperature 
drop across the hot wall ('--longitudinal wall'). The same estimate is made for every single run (see Multiple configs and summary). The full restraint makes this conservative, real chambers often yield in compression and are designed for 
low cycle fatigue, so the margin is best used to compare designs. The temperature dependent material data in MaterialLib.py are representative 
published values; replace them with your supplier's data.

```bash
    python3 optimize.py configs/a.py --max-dp 10 --max-angle 30 --min-rib 0.8 --min-channel-width 1 --min-channel-height 1
    python3 optimize.py configs/a.py --materials IN718 CuCr1Zr --density coarse --verify 2 --min-boil-margin 0 --max-dp 10 --min-rib 0.8 --min-channel-width 1 --min-channel-height 1
    python3 optimize.py configs/rp1.py --wall 0.8 --max-coolant-wall-temp 560 --max-dp 15 --min-rib 0.8 --min-channel-width 1 --min-channel-height 1
```

Results are written to results/optimize_<config>/: designs_fast.csv (every design), optimize.png, verified.csv, verified/<design>/ (2D results), 
verified_summary.png and best_design.txt. The browser interface has an Optimize tab for the same.

### Temperature units
Plots, section images, the summary and terminal output show temperatures in K by default. Set TEMPERATURE_UNIT = 'C' in Units.py to switch all of them to °C. 
Config inputs and the data files (sim_data.csv, summary.json, summary.csv) always use K.

### Debug mode
Set the 'print_result' option to True in config.py in order to display more detialed information on the each transient thermal simulation. The output will show each time 
step and the resulting maximum temperature. This can show non convergence, in the form of reasonable, but oscillating values, or obvious numerical errors in the 
simulation. In the case of non convergence, the time step can be lowered, or the mesh resolution can be increased. Additionally the maximum iteration count
can be increased if necessary. 


### Setting up chamber geometries
The combustion chamber is defined using a series of parameters in config.py that describe the cylindrical chamber section, and the converging and diverging sections 
respectively. The variable convention follows [RPA](https://www.rocket-propulsion.com/index.htm): r_2 is the radius from the cylindrical section into the 
converging section (R2) and r_1 the radius upstream of the throat (R1), so RPA output can be copied directly. The divergent section is a 
thrust-optimised parabola [Rao 1960]: nozzle_type = 'parabola' fits x(y) to the wall angles phi_div and phi_e, nozzle_type = 'bezier' uses a 
quadratic Bezier curve of length L_e between the same tangents [Newlands 2017], as RPA does. A plot of the chamber contour is saved in the target directory before each run. The variable 'step_size' in config.py determines the 
number of sections by imposing a minimum resolution in axial direction. Optionally, 'step_size_throat' and 'throat_zone' (both in m) give a finer step size 
within throat_zone of the throat, blending back to step_size over the same distance, e.g. step_size = 0.006, step_size_throat = 0.002, throat_zone = 0.03. 
This resolves the throat better with fewer sections in total (and a shorter run time) than a uniform step size.


### Setting up cooling channel geometries
The basic program is set up for a rectangular cooling channel geometry with cooling channel areas as a fraction of local circumference. The basic properties of the cooling 
geometry, such as area and hydraulic diameter are computed in GeomClass.py. The mesh for each chamber section is created in SectionMesh.py, with the final thermal 
simulation and evaluation of results in SectionThermalSim.py. The following naming convention is used for the individual faces (this is relevant for allocating 
boundary conditions and evaluating results).

Instead of the fill factor psi, the channel width normal to the flow can be set directly with 'channel_width' (float or function of x, in m), 
which is how channel and rib widths are usually specified for manufacturing. The results report the narrowest channel and rib width normal to 
the flow, the smallest channel height and the largest helix angle, as well as the margin of the coolant wetted wall and of the coolant bulk 
to the saturation temperature at the local pressure (negative: the wall is above saturation, i.e. subcooled boiling, which the heat transfer 
correlations do not model). The coolant side wall temperature is written to sim_data.csv as the last column.

Helical channels are enabled with 'helix_pitch_deg_per_mm' (degrees of revolution per mm of axial length) or 'helix_angle' (angle from the meridian, 
float or function of x). The fill factor psi stays the fraction of the circumference covered by channels. Since the temperature field in the wall is 
invariant along the channel direction, the 2D section is solved in the plane normal to the channel, where channel and rib widths are smaller by 
cos(alpha). The flow area, hydraulic diameter and velocity are taken normal to the flow, the flow path is 1/cos(alpha) times the contour length. 
Curvature from winding around the chamber (normal curvature cos(beta) sin(alpha)^2 / R of the surface of revolution [do Carmo 1976]) is included 
for turbulent flow: friction is increased with Ito's correlation [Ito 1959], the coolant side heat transfer is reduced on the hot (convex) wall 
and increased on the outer (concave) wall [Niino et al. 1982]. For typical pitches these curvature 
corrections are small, the flow area and path length dominate.

Boundary Name 		| Description
------------------- | -------------------------------------------------------------
ChamberWall   		| Inner chamber wall in contact with combustion gases
OuterWall     	 	| Outer wall of combustion chamber
CoolantBottomWall	| Wall of the cooling channel closest to the inner chamber wall
CoolantTopWall		| Wall of the cooling channel closest to the outer chamber wall 
CoolantSideWall		| Wall of the cooling channel rib

If the geometry of the cooling channels is fundamentally changed, several functions in several files need to be altered. In order to edit the basic properties, such as 
area and hydraulic diameter, edit the 'CoolingGeometry' class in GeomClass.py. In order to alter the mesh itself, edit the SectionMesh.py file. The 2D mesh geometry is 
produced using [GMSH](https://gmsh.info/). Several tutorials exist on how to use GMSH. Lastly, SectionThermalSim.py needs to be edited. This extends to the faces on which 
boundary conditions are applied, as well as the faces over which boundary fluxes along normal vectors are calculated. These can be found starting in line 52 and 126 respectively. 
Changes to the mesh can be verified by running SectionThermalSim.py


### 2D section solver 
By default (run_time = 'steady_state', steady_solver = 'direct') the steady state of each section is solved directly: the heat transfer to the hot gas, the 
coolant and the ambient enters the linear system as a conductance per boundary face (the film coefficient in series with the conduction over the half cell 
next to the wall [Patankar 1980]; radiation to ambient linearised as in [Incropera et al. 2007, Eq. 1.9]), and the temperature dependent heat transfer coefficients and thermal conductivity are updated until the temperatures stop changing, 
typically in about 5 iterations. Wall temperatures are reported at the wall surface. Meshes of identical sections are reused and the meshes of a run are 
generated in the background while the sections are solved. Setting steady_solver = 'pseudo-transient' in the config uses the transient solver described 
below to reach steady state instead (starting at 50 times the configured time step), setting run_time to a time in s runs a real transient.

A transient thermal simulation is run on each section of the combustion chamber geomerty. Setting up the finite volume system and boundary conditions is done using 
[FiPy](https://www.ctcms.nist.gov/fipy/) [Guyer et al. 2009], on meshes from gmsh [Geuzaine & Remacle 2009]. The thermal simualtion takes advantage of symmetry in circumferetial direction, with the flux over the symmetry walls being zero. 

```math
\frac{\partial T}{\partial t} = \alpha \nabla^2 T
```

With 

```math
 q = - k \frac{\partial T}{\partial x}
```

A constant thermal diffusivity is assumed, witht the thermal conductivity dependent on temerpature. The convective heat flux boundary conditions take the form of 

```math
\frac{\partial T}{\partial x} = \frac{1}{k} h_{\alpha} (T_{external} - T_{wall})
```

In varying forms this BC is applied to the boundaries: 'ChmaberWall', 'CoolantBottomWall', 'CoolantTopWall' and 'CoolantSideWall' . The radiation boundary condition
is expressed as

```math
\frac{\partial T}{\partial x} = \frac{\sigma \epsilon}{k} (T_{ambient}^4 - T_{wall}^4)
```

and is applied to the 'OuterWall' boundary. The heat equation is solved for each time step, after which the boundary conditions are updated using the results 
of the previous time step before solving the next time step. The simulation is either terminated when the maximum temperature reaches steady state (within a set tolerance),
or after a set maximum number of iterations is reached. All settings for the 2D thermal sim are found in config.py. During a full run of the program, the boundary flux over
all boundaries is calculated and used to determine the total heat flux, as well as the temperature increase of the cooling fluid as it passes throug each section. 

## Liquid Film Cooling Model
Optional addition of film cooling to supplement regenerative cooling. This is an implementation of the analytical 0D liquid film cooling model proposed by [Shine et al. 2012](https://www.researchgate.net/publication/256718300_A_new_generalised_model_for_liquid_film_cooling_in_rocket_combustion_chambers), as detailed in [Shine 2013, Ch. 5]: 
friction factor [McKeon et al. 2005], dry-wall Stanton number [Friend & Metzner 1958], free-stream turbulence factor [Pletcher 1988], transpiration 
correction [Kays et al. 2005] with the molecular weight factor of [Meinert & Huhn 2001] and entrainment [Sawant et al. 2008].The model relies on an energy balance between the convective and radiative heat transfer of the combustion gases and the evaporation of the coolant. The model produces an estimate of the liquid film length and assumes that the temperature rise of the wall under the liquid film is very small. The liquid film cooling length and the resultant reduction in convective heat transfer coefficient are passed to the regenerative cooling method. The film cooling is only assumed to be effective as long as the chamber section is within the liquid film length. 

### Plotting Options 
Results of the full chamber simulation are saved in the target directory as a csv file. A select set of saved data is automatically plotted by calling 'PlottingFunctions.py'.
These include the heat transfer coefficients, coolant properties, as well as maximum wall temperature. 


### Chamber Side Heat Transfer Coefficient Models
The heat flux to the combustion chamber is dominated by convective heat transfer, with some contribution from radiation. Three Nusselt number correlations can be chosen for 
the convective heat transfer coefficients. 

Option | Source
------ | ------
'standard-bartz' | [Bartz 1957], with the throat curvature factor taken as 1; scaled by bartz_knockdown
'modified-bartz' | Bartz form with the local mass flux and properties at the [Eckert 1955] reference temperature; scaled by bartz_knockdown. Source of this variant not given in the original PyRocket
'cinjarev' | St = 0.0162 Re^-0.18 Pr^-0.18 (T_hot/T_w)^0.35 [Schmidt 1999; Kirchberger et al. 2009; Betti et al. 2015], written in mass flow and local diameter (0.0162 (4/pi)^0.82 = 0.01975), with T_hg for T_hot. Not affected by bartz_knockdown
		
Especially for small chamber geomerties, the 'cinjarev' correlation is recommended. Gas properties come from NASA CEA [Gordon & McBride 1994] via 
[RocketCEA](https://rocketcea.readthedocs.io/). Local static conditions follow from quasi-1D isentropic flow [Anderson 2003; Sutton & Biblarz 2017], the adiabatic 
wall temperature from a recovery factor Pr^(1/3) upstream and Pr^(1/2) downstream of the throat [Kays et al. 2005]. All three models drive the heat flux 
with T_hg = eta_c*^2 T_aw, the adiabatic wall temperature as in RPA's Bartz method [Ponomarenko 2012], corrected for the c* efficiency (c* ~ sqrt(T)). 
The original PyRocket used T_s + 0.9 (eta_c*^2 T_aw - T_s), which applied a recovery factor a second time. Compared with it, T_hg is now 
about 1 % higher at the throat and 4 % higher at the nozzle exit (an IPA/LOX case with eta_c* = 1); with eta_c* < 1 it is lower in the chamber 
(5 % at eta_c* = 0.8), where the old form weighted in 10 % of the uncorrected static temperature. The gas radiation from H2O and CO2 is an empirical correlation from the original PyRocket without a 
given source.


### Coolant Side Heat Transfer Coefficient Models
Similar to the hot gas side, the convection of heat to the cooling fluid can be estimated using one of three Nusselt number correlations. 

Option | Source
------ | ------
'gnielinski' | [Gnielinski 1976], with the [Colebrook 1939] friction factor (includes wall roughness)
'dittus-boelter' | Actually the [Sieder & Tate 1936] form 0.027 Re^0.8 Pr^(1/3) (mu/mu_wall)^0.14; the option name is historical
'dittus-boelter-simple' | [Dittus & Boelter 1930] in the McAdams form 0.023 Re^0.8 Pr^0.4 [Winterton 1998]

Validity ranges are those in [Incropera et al. 2007, Ch. 8]. The pressure drop uses Darcy-Weisbach with the [Colebrook 1939] friction factor. 
The near-wall fluid temperature for property evaluation is the logarithmic mean of wall and bulk temperature (an assumption of the original PyRocket). 

These correlations have different ranges of applicability in terms of Reynolds number. Check the comments in RegenCooling.py for details. The 'gnielinski' model can lead to
negative Nusselt numbers when the Reynolds number of the cooling fluid is too low. If this error is encountered, use the 'dittus-boelter-simple' correlation instead. The 
'dittus-boelter' correlation also features a viscosity correction for the near-wall fluid film. In practive the 'gnielinski' model is the most accurate, if applicable, as it 
includes information about surface roughness, especially relevant for 3D printed designs. 


### Using Material Library
Five chamber materials can be selected from MaterialLib.py: 'Ti-6Al-4V', 'IN718', 'CuCr1Zr', 'SS 1.4404' and 'AlSi10Mg'. The sources of the 
thermal conductivity tables are given next to them. IN718, CuCr1Zr and AlSi10Mg also carry temperature dependent Young's modulus, expansion 
and yield strength for the stress estimate; these are representative published values and should be replaced with supplier data. These can be selected in config.py.
If a differnet material is desired, it can be added in MaterialLib.py, similar to the existing examples. 


### Creating New Propellant Blends
In config.py any fuel and oxidiser present in [RocketCEA](https://rocketcea.readthedocs.io/en/latest/) can be input. If a new propellant is desired, it can be 
created as a string input, as shown in the PropLibrary.py file. Several examples are already present of new chemicals, as well as propellant blends. 


## Two-pass cooling
Setting two_pass = True in a config splits the n channels (n must be even) into two alternating passes. The first pass carries the coolant from the end 
given by start_idx to the other end, where a manifold turns it into the channels of the second pass, which carry it back, so inlet and outlet are both at 
the start_idx end. Each pass carries the full coolant mass flow through half of the channels.

The 2D section then spans from the centre of a first pass channel over a full rib to the centre of the neighbouring return channel, with the coolant 
temperature and heat transfer coefficient of each pass on its own channel walls, so heat is also exchanged between the passes through the ribs. Both 
passes are solved together along the chamber: the return pass state at the outlet is guessed and corrected (typically in 3 to 5 iterations of the whole 
chamber) until the return pass leaves the turnaround manifold at the temperature of the first pass and at its pressure less the turnaround loss 
(turnaround_loss times the dynamic pressure in the channels, default 1.5, of the order of sharp 180 degree return bends [Idelchik 1994]; the 
manifold geometry itself is not modelled). Results contain the return pass as additional columns of sim_data.csv and as 
dashed lines in the plots; the coolant outlet temperature and pressure drop in the summary are those at the outlet of the return pass.

## Returning cooling channels 
Designs whose channels first move along the chamber in one direction and return through another set of channels are simulated directly with 
two_pass = True, see "Two-pass cooling" above. This replaces the earlier approach of two separate runs with the full number of channels and double 
the coolant mass flow, which could not capture the heat exchange between the passes or the turnaround pressure loss.

## Optimisation and Parametric Sweeps 
PyRocket can be used to optimise a cooling channel geometry, as well as determining the impact of certain parameters on cooling performance. A sequential optimisation 
routine can be implemented using scipy.optimize.minimize. Using a Monte Carlo approach or parametric sweeps can be run in parallel using python multiprocessing. 
This approach is recommended in most cases as single case run times typically exceed 5 mins and are intrinsically difficult to parallelise. 

## Program Validation
PyRocket has been validated using data gathered on 22 N N2O/Hydrocarbon thrusters. Using data from nine steady state tests, an average total heat flux 
error of 8.3% with a standard deviation of 5.6% was determined. The wall temperature has been validated at discrete locations with a mean error of 11.2% and a standard deviation of 10%.
For Validation purposes the axial and radial coordinate of thermocouples along the camber can be set in config.py. The temperature at that location will then be displayed in the 
terminal output for the simulation. Set 'log_TC' to True to enable the output.  


## References
Short citations in the code and this README refer to:

* Anderson, J. D. (2003). *Modern Compressible Flow*, 3rd ed. McGraw-Hill.
* Andrade, E. N. da C. (1930). The viscosity of liquids. *Nature*, 125, 309–310.
* Bartz, D. R. (1957). A simple equation for rapid estimation of rocket nozzle convective heat transfer coefficients. *Jet Propulsion*, 27(1), 49–51.
* Bell, C. et al. (2016–). Thermo: Chemical properties component of the Chemical Engineering Design Library. https://github.com/CalebBell/thermo
* Betti, B., Liuzzi, D., Nasuti, F., & Onofri, M. (2015). Development of heat transfer correlations for LOX/CH4 thrust chambers. *6th European Conference for Aeronautics and Space Sciences (EUCASS)*, Kraków.
* Colebrook, C. F. (1939). Turbulent flow in pipes, with particular reference to the transition region between the smooth and rough pipe laws. *Journal of the Institution of Civil Engineers*, 11(4), 133–156.
* Dittus, F. W., & Boelter, L. M. K. (1930). Heat transfer in automobile radiators of the tubular type. *University of California Publications in Engineering*, 2(13), 443–461.
* do Carmo, M. P. (1976). *Differential Geometry of Curves and Surfaces*. Prentice-Hall.
* Eckert, E. R. G. (1955). Engineering relations for friction and heat transfer to surfaces in high velocity flow. *Journal of the Aeronautical Sciences*, 22(8), 585–587.
* Friend, W. L., & Metzner, A. B. (1958). Turbulent heat transfer inside tubes and the analogy among heat, mass and momentum transfer. *AIChE Journal*, 4, 393–402.
* Geuzaine, C., & Remacle, J.-F. (2009). Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities. *International Journal for Numerical Methods in Engineering*, 79(11), 1309–1331.
* Gnielinski, V. (1976). New equations for heat and mass transfer in turbulent pipe and channel flow. *International Chemical Engineering*, 16(2), 359–368.
* Gordon, S., & McBride, B. J. (1994). *Computer Program for Calculation of Complex Chemical Equilibrium Compositions and Applications, Part I: Analysis*. NASA RP-1311.
* Guyer, J. E., Wheeler, D., & Warren, J. A. (2009). FiPy: Partial differential equations with Python. *Computing in Science & Engineering*, 11(3), 6–15.
* Idelchik, I. E. (1994). *Handbook of Hydraulic Resistance*, 3rd ed. CRC Press.
* Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007). *Fundamentals of Heat and Mass Transfer*, 6th ed. Wiley.
* Ito, H. (1959). Friction factors for turbulent flow in curved pipes. *Journal of Basic Engineering*, 81(2), 123–134.
* Kays, W. M., Crawford, M. E., & Weigand, B. (2005). *Convective Heat and Mass Transfer*, 4th ed. McGraw-Hill.
* Kirchberger, C., Hupfer, A., Kau, H., Soller, S., Martin, P., Bouchez, M., & Dufour, E. (2009). Improved prediction of heat transfer in rocket combustor for GOX/kerosene. AIAA Paper 2009-1214.
* Kolda, T. G., Lewis, R. M., & Torczon, V. (2003). Optimization by direct search: New perspectives on some classical and modern methods. *SIAM Review*, 45(3), 385–482.
* McKeon, B. J., Zagarola, M. V., & Smits, A. J. (2005). A new friction factor relationship for fully developed pipe flow. *Journal of Fluid Mechanics*, 538, 429–443.
* Meinert, J., & Huhn, J. (2001). Turbulent boundary layers with foreign gas transpiration. *Journal of Spacecraft and Rockets*, 38, 191–198.
* Newlands, R. (2017). *The Thrust Optimised Parabolic Nozzle*. Aspire Space technical paper.
* Niino, M. et al. (1982). Heat transfer characteristics of liquid hydrogen as a coolant for the LO2/LH2 rocket thrust chamber with the channel wall construction. AIAA Paper 82-1107.
* Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere.
* Perakis, N., Sternin, A., Roth, C., & Haidn, O. J. (2017). Heat transfer and combustion simulation of a 7-element GOX/GCH4 rocket combustor. *SFB/TRR 40 Summer Program Report 2017*, TU München.
* Pletcher, R. H. (1988). Progress in turbulent forced convection. *Journal of Heat Transfer*, 110, 1129–1144.
* Ponomarenko, A. (2012). *RPA: Tool for Rocket Propulsion Analysis. Thermal Analysis of Thrust Chambers*. https://www.rocket-propulsion.com/downloads/pub/RPA_ThermalAnalysis.pdf
* Rao, G. V. R. (1960). Approximation of optimum thrust nozzle contour. *ARS Journal*, 30(6), 561.
* Sawant, P., Ishii, M., & Mori, M. (2008). Droplet entrainment correlation in vertical upward co-current annular two-phase flow. *Nuclear Engineering and Design*, 238(6), 1342–1352.
* Schmidt, G. (1999). *Technik der Flüssigkeits-Raketentriebwerke*. DaimlerChrysler Aerospace, Munich.
* Shine, S. R. (2013). *Studies on Film Cooling in Rocket Combustion Chambers*. PhD thesis, Indian Institute of Space Science and Technology.
* Shine, S. R., Kumar, S. S., & Suresh, B. N. (2012). A new generalised model for liquid film cooling in rocket combustion chambers. *International Journal of Heat and Mass Transfer*, 55(19–20), 5065–5075.
* Sieder, E. N., & Tate, G. E. (1936). Heat transfer and pressure drop of liquids in tubes. *Industrial & Engineering Chemistry*, 28(12), 1429–1435.
* Sutton, G. P., & Biblarz, O. (2017). *Rocket Propulsion Elements*, 9th ed. Wiley.
* Winterton, R. H. S. (1998). Where did the Dittus and Boelter equation come from? *International Journal of Heat and Mass Transfer*, 41(4–5), 809–810.

Models used without a known source (inherited from the original PyRocket): the 'modified-bartz' variant, the H2O/CO2 radiation 
correlation and the log-mean near-wall coolant temperature. [Betti et al. 2015] and [Perakis et al. 2017] were checked 
for these: Betti et al. give the Cinjarev correlation above and an Eckert reference temperature for Krueger's correlation, not this Bartz 
variant; Perakis et al. use a "modified Cinjarev" correlation [Schmidt 1999] without stating the modification.

## Common Issues

### Single section takes more than 10 seconds to solve

This can either be caused by the solver not reaching a steady state solution or the time step being set too small. Abort the simulation and try again with a different adaptive time step configuration. The function 'adaptive_time_step' in 'SectionThermalSim.py' is responsible for setting the time step based on the previous solution. For better convergence behaviour reduce the value of 'step_up' in the 'adaptive_time_step' function (default is 1.2). 

### Fluid propery returned as 'None' type

The library [thermo](https://thermo.readthedocs.io/) does not produce reliable results for Prandtl Number, Cp or viscosity near the supercritical region, 
especially for mixtures. The error will occur in the function 'heat_transfer_coeff_coolant' in the file 'RegenCooling.py'. If this issue occurs, use a 
simplified single componant fluid as coolant. You can also write your own coolant class based on the outputs of the thermo Mixture object and replace the
standard implementation. For this changes need to be made in 'RegenCooling.py' and 'config.py'


    
