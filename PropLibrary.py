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

import math
import rocketcea
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_oxidizer, add_new_propellant

# ethanol blends
ethanol90 = rocketcea.blends.newFuelBlend(fuelL=['C2H5OH', 'H2O'], fuelPcentL=[90,10]) 
ethanol98 = rocketcea.blends.newFuelBlend(fuelL=['C2H5OH', 'H2O'], fuelPcentL=[98,2]) 

# isopropanol (IPA) blends; RocketCEA name is 'Isopropanol', thermo coolant name is 'isopropanol'
isopropanol90  = rocketcea.blends.newFuelBlend(fuelL=['Isopropanol', 'H2O'], fuelPcentL=[90,10])
isopropanol95  = rocketcea.blends.newFuelBlend(fuelL=['Isopropanol', 'H2O'], fuelPcentL=[95,5])
isopropanol99  = rocketcea.blends.newFuelBlend(fuelL=['Isopropanol', 'H2O'], fuelPcentL=[99,1])
isopropanol100 = rocketcea.blends.newFuelBlend(fuelL=['Isopropanol'], fuelPcentL=[100])
ipa            = isopropanol100

# RP1 (kerosene); thermo has no RP1 entry, use 'n-dodecane' as coolant surrogate
rp1 = rocketcea.blends.newFuelBlend(fuelL=['RP1'], fuelPcentL=[100])

# methanol blends
methanol80 = rocketcea.blends.newFuelBlend(fuelL=['Methanol', 'H2O'], fuelPcentL=[80,20]) 
methanol85 = rocketcea.blends.newFuelBlend(fuelL=['Methanol', 'H2O'], fuelPcentL=[85,15]) 
methanol90 = rocketcea.blends.newFuelBlend(fuelL=['Methanol', 'H2O'], fuelPcentL=[90,10]) 

# Example of RocketCEA propellant blends. Works the same way for fuels
# Hydrogen Peroxide  
peroxide98 = rocketcea.blends.newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[98,2]) 
peroxide96 = rocketcea.blends.newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[96,4]) 
peroxide94 = rocketcea.blends.newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[94,6]) 
peroxide90 = rocketcea.blends.newOxBlend(oxL=['H2O2', 'H2O'], oxPcentL=[90,10])

# Nitromethane — not in the RocketCEA built-in database
# Formula: CH3NO2  |  MW = 61.04 g/mol  |  rho = 1.139 g/cm³ @ 20°C (CHRIS 9.7)
# ΔHf°(liquid) = -113.1 kJ/mol (NIST, Cass et al. 1958 reanalysed)
card_str = """
fuel CH3NO2(L)  C 1.0   H 3.0   N 1.0   O 2.0  wt%=100
h,kj/mol=-113.1    t(k)=298.15   rho=1.139
"""
add_new_fuel('ch3no2', card_str)
nitromethane = rocketcea.blends.newFuelBlend(fuelL=['ch3no2'], fuelPcentL=[100])

nitromethane5050 = rocketcea.blends.newFuelBlend(fuelL=['ch3no2', 'Methanol'], fuelPcentL=[50, 50])


# Examples of adding new propellants
# aniline
card_str = """
fuel C6H7N(L)  C 6.0   H 7.0    N 1.0  wt%=100
h,kj/mol=31.3    t(k)=298.15   rho=1.03
"""
add_new_fuel( 'aniline', card_str )                              # fixed: card is a fuel, not an oxidizer
aniline = rocketcea.blends.newFuelBlend(fuelL=['aniline'], fuelPcentL=[100])


# furfurylalcohol
card_str = """
fuel C5H6O2(L)  C 5.0   H 6.0    O 2.0  wt%=100
h,kj/mol=-276.2    t(k)=298.15   rho=1.13 
"""
add_new_fuel( 'furfurylalcohol', card_str )
furfurylalcohol = rocketcea.blends.newFuelBlend(fuelL=['furfurylalcohol'], fuelPcentL=[100])


######################################################################
# Coolant factory — drop-in replacement for thermo.Mixture
######################################################################

class CustomCoolant:
    """Drop-in replacement for thermo.Mixture for coolants not present in the
    thermo database.  Exposes every attribute that RegenCooling.py and
    FilmCooling.py consume from a thermo.Mixture object.

    Construct instances via :func:`create_coolant` rather than calling
    this class directly.

    Property arguments (*rho_l*, *mu_l*, …) accept either a callable
    ``f(T [K], P [Pa]) -> float`` or a plain numeric constant.
    """

    def __init__(self, IDs, ws, T, P,
                 rho_l, mu_l, Cp_l, Pr_l,
                 Tbs, Hvap_Tbs, MW,
                 sigma=None,
                 rho_g=None, mu_g=None, Cp_g=None, Pr_g=None):
        self.IDs      = list(IDs)
        self.ws       = list(ws)
        self.T        = float(T)
        self.P        = float(P)
        self.MW       = float(MW)
        self.Tbs      = list(Tbs)
        self.Hvap_Tbs = list(Hvap_Tbs)

        # Keep references so create_coolant can reconstruct at a new (T, P)
        self._rho_l = rho_l;  self._mu_l = mu_l
        self._Cp_l  = Cp_l;   self._Pr_l = Pr_l
        self._sigma = sigma
        self._rho_g = rho_g;  self._mu_g = mu_g
        self._Cp_g  = Cp_g;   self._Pr_g = Pr_g

        def _ev(fn):
            return fn(T, P) if callable(fn) else float(fn)

        # Phase: dominant component boiling point determines l / g
        dominant   = self.ws.index(max(self.ws))
        self.phase = 'l' if T < self.Tbs[dominant] else 'g'

        # Liquid-phase properties (always evaluated)
        self.rhol = _ev(rho_l);  self.mul = _ev(mu_l)
        self.Cpl  = _ev(Cp_l);   self.Prl = _ev(Pr_l)

        # Gas-phase properties (fall back to liquid values when not supplied)
        self.rhog = _ev(rho_g) if rho_g is not None else self.rhol
        self.mug  = _ev(mu_g)  if mu_g  is not None else self.mul
        self.Cpg  = _ev(Cp_g)  if Cp_g  is not None else self.Cpl
        self.Prg  = _ev(Pr_g)  if Pr_g  is not None else self.Prl

        # Phase-selected bulk aliases used in the energy balance and Nu correlations
        if self.phase == 'l':
            self.rho = self.rhol;  self.mu = self.mul
            self.Cp  = self.Cpl;   self.Pr = self.Prl
        else:
            self.rho = self.rhog;  self.mu = self.mug
            self.Cp  = self.Cpg;   self.Pr = self.Prg

        # Surface tension (required by FilmCooling; None disables film cooling)
        self.sigma = _ev(sigma) if sigma is not None else None


# Maps lower-cased name -> keyword dict passed to CustomCoolant.__init__
_custom_coolant_registry: dict = {}


def register_coolant(name, rho_l, mu_l, Cp_l, Pr_l,
                     Tbs, Hvap_Tbs, MW,
                     sigma=None,
                     rho_g=None, mu_g=None, Cp_g=None, Pr_g=None):
    """Register a custom coolant so that :func:`create_coolant` can use it.

    Args:
        name (str):              identifier used in the ``cooling_fluid`` list
                                 in config.py
        rho_l:                   liquid density [kg/m^3], callable(T,P) or const
        mu_l:                    liquid dynamic viscosity [Pa*s], callable or const
        Cp_l:                    liquid specific heat [J/kg/K], callable or const
        Pr_l:                    liquid Prandtl number [-], callable or const
        Tbs  (list[float]):      boiling temperature per component [K]
        Hvap_Tbs (list[float]):  latent heat per component at Tb [J/kg]
        MW   (float):            molecular weight [g/mol]
        sigma:                   surface tension [N/m], callable or const;
                                 required when film cooling is active
        rho_g, mu_g, Cp_g, Pr_g: gas-phase equivalents; fall back to liquid
                                 values when omitted

    For temperature-dependent properties pass a lambda or function::

        import PropLibrary as proplib
        # linear density approximation for illustration
        proplib.register_coolant(
            'my_coolant',
            rho_l=lambda T, P: 1200.0 - 0.8 * (T - 293.0),
            mu_l=2.0e-4, Cp_l=2000.0, Pr_l=5.0,
            Tbs=[420.0], Hvap_Tbs=[500_000.0], MW=80.0,
            sigma=0.025,
        )
    """
    _custom_coolant_registry[name.lower()] = dict(
        rho_l=rho_l, mu_l=mu_l, Cp_l=Cp_l, Pr_l=Pr_l,
        Tbs=Tbs, Hvap_Tbs=Hvap_Tbs, MW=MW, sigma=sigma,
        rho_g=rho_g, mu_g=mu_g, Cp_g=Cp_g, Pr_g=Pr_g,
    )


def create_coolant(IDs, ws=None, T=298.15, P=101325.0):
    """Coolant factory -- the single point for constructing and updating the
    coolant object throughout the simulation.

    Returns a ``thermo.Mixture`` for any fluid the thermo database recognises,
    or a :class:`CustomCoolant` for anything registered via
    :func:`register_coolant`.

    Replace every direct ``thermo.Mixture(...)`` call for the coolant with
    this function.

    Args:
        IDs (list[str]):   species identifiers (thermo names or registered names)
        ws  (list[float]): mass fractions (must sum to 1); defaults to ``[1.0]``
        T   (float):       temperature [K]
        P   (float):       pressure [Pa]
    """
    import thermo as _thermo

    if ws is None:
        ws = [1.0]

    try:
        return _thermo.Mixture(IDs, ws=ws, T=float(T), P=float(P))
    except Exception as exc:
        ids_lower = [i.lower() for i in IDs]
        missing   = [i for i in ids_lower if i not in _custom_coolant_registry]
        if len(missing) == len(ids_lower):
            # none of the fluids is a custom coolant: pass on the thermo error, e.g. an invalid state such as a negative
            # pressure after an excessive pressure drop, or an unknown fluid name
            raise
        if missing:
            raise ValueError(
                f"Chemical ID(s) {missing} not recognised by thermo and not "
                "registered as a custom coolant. "
                "Call PropLibrary.register_coolant() before running the simulation."
            ) from exc

        # Single-component short-circuit
        if len(IDs) == 1:
            spec = _custom_coolant_registry[ids_lower[0]]
            return CustomCoolant(IDs, ws, T, P, **spec)

        # Multi-component blend: mass-average every property function
        def _blend(attr):
            fns = [_custom_coolant_registry[i][attr] for i in ids_lower]
            def _blended(T, P):
                return sum(
                    (fn(T, P) if callable(fn) else float(fn)) * w
                    for fn, w in zip(fns, ws)
                )
            return _blended

        all_Tbs  = [Tb for i in ids_lower for Tb in _custom_coolant_registry[i]['Tbs']]
        all_Hvap = [Hv for i in ids_lower for Hv in _custom_coolant_registry[i]['Hvap_Tbs']]
        blend_MW = sum(_custom_coolant_registry[i]['MW'] * w for i, w in zip(ids_lower, ws))
        gas_ok   = all(_custom_coolant_registry[i][a] is not None
                       for i in ids_lower for a in ('rho_g', 'mu_g', 'Cp_g', 'Pr_g'))
        sig_ok   = all(_custom_coolant_registry[i]['sigma'] is not None for i in ids_lower)

        return CustomCoolant(
            IDs, ws, T, P,
            rho_l=_blend('rho_l'), mu_l=_blend('mu_l'),
            Cp_l =_blend('Cp_l'),  Pr_l=_blend('Pr_l'),
            Tbs=all_Tbs, Hvap_Tbs=all_Hvap, MW=blend_MW,
            sigma=_blend('sigma') if sig_ok else None,
            rho_g=_blend('rho_g') if gas_ok else None,
            mu_g =_blend('mu_g')  if gas_ok else None,
            Cp_g =_blend('Cp_g')  if gas_ok else None,
            Pr_g =_blend('Pr_g')  if gas_ok else None,
        )



# CUSTOM COOLANTS:
# ---------------------------------------------------------------------------
# Nitromethane (CH3NO2, CAS 75-52-5)
#
# Source: NOAA/USCG Chemical Hazards Response Information System (CHRIS) datasheet
#         + literature viscosity (Viswanath & Natarajan, 1989)
#
# Property variation 21–75 °C (294–348 K):
#   rho  : -5.6 %   --> linear fit to CHRIS table (9.20), converted from lb/ft³
#   Cp   : +5.0 %   --> linear fit to CHRIS table (9.21), converted from BTU/lb-°F
#   k    : -4.3 %   --> linear fit to CHRIS table (9.22), BTU·in/(hr·ft²·°F) → W/m·K
#   mu   : -32  %   --> Arrhenius fit to published data; 627 µPa·s @ 20 °C
#   sigma: -16  %   --> linear fit anchored at CHRIS 9.8 (37.0 mN/m @ 20 °C)
#   Pr   : -25  %   --> derived (Cp·mu/k), significant variation
# ---------------------------------------------------------------------------
register_coolant(
    'nitromethane',

    # Density [kg/m³]  — linear fit to CHRIS 9.20 (±0.1% over 275–345 K)
    rho_l   = lambda T, P: 1520.0 - 1.300 * T,

    # Dynamic viscosity [Pa·s]  — Arrhenius fit; NOT in datasheet (marked N/A)
    # Reference values: 6.27×10⁻⁴ Pa·s @ 293 K, 4.90×10⁻⁴ Pa·s @ 333 K
    # If more accurate viscosity data is available, replace this lambda.
    mu_l    = lambda T, P: math.exp(-9.444 + 607.0 / T),

    # Specific heat capacity [J/kg·K]  — linear fit to CHRIS 9.21 (±0.2% over 250–333 K)
    Cp_l    = lambda T, P: 1282.6 + 1.658 * T,

    # Prandtl number [-]  — derived from Cp, mu, k
    Pr_l    = lambda T, P: (
        (1282.6 + 1.658 * T) *
        math.exp(-9.444 + 607.0 / T) /
        (0.25671 - 1.777e-4 * T)
    ),

    # Boiling point [K] and latent heat at Tb [J/kg]  — CHRIS 9.3 and 9.12
    Tbs      = [374.4],
    Hvap_Tbs = [5.61e5],

    # Molecular weight [g/mol]
    MW      = 61.04,

    # Surface tension [N/m]  — linear fit anchored at CHRIS 9.8 (37.0 mN/m @ 293.15 K)
    # Slope from published nitromethane data (~−0.111 mN/m per K)
    sigma   = lambda T, P: max(0.0, 0.0370 - 1.11e-4 * (T - 293.15)),
)
# 'ch3no2' is an alias so config.py entries using the formula string still work
_custom_coolant_registry['ch3no2'] = _custom_coolant_registry['nitromethane']
