"""
1-d thermal modeling functions

With ability to handle surface fluxes on slopes surfaces and under atmospheric conditions
"""

# Physical constants:
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
kb = 1.38066e-23 #Boltzmann's Constant [J.K-1]
#S0 = 1361.0 # Solar constant at 1 AU [W.m-2]
chi = 2.7 # Radiative conductivity parameter [Mitchell and de Pater, 1994]
R350 = chi/350**3 # Useful form of the radiative conductivity
Rg = 8.314 #universal gas constant [J.K-1.mol-1]

#CO2 ice parameters
L_CO2 = 6e5 #latent heat of CO2 [J.kg-1] (Aharonson and Schorghofer 2006)
A_CO2 = 0.40 #0.65 #Albedo of CO2 frost (Aharonson and Schorghofer 2006)
emis_CO2 = 1. #emissivity of CO2 frost (Aharonson and Schorghofer 2006)
mc = 0.044 #molar mass of CO2 [kg]

#H2O ice parameters
L_H2O = 2.83e6 #latent heat of H2O [J.kg-1] (Dundas and Byrne 2010)
M_w = 2.99e-26 #molecular mass of water [kg]
A_drag = 0.002 #drag coefficient (Dundas and Byrne 2010)
Hc = 1e4 #estimate of Mars northern summer water vapor condensation level [m] from (Smith 2002)
h_vap = 5e-5 #estimate of Mars northern summer water vapor column abundance [m] from (Smith 2002)
rho_liq = 997 #density of liquid water [kg.m-3]
mw = 0.018 #molar mass of water [kg]
ki = 3.2 #conductivity of ice [W.m-1.K-1]
rho_ice = 927 #density of water ice [kg.m-3]
A_NPLD = 0.32 #Albedo of NPLD during summer

#Mars atmospheric properties
rho_surf = 0.020 #mean Martian atmospheric surface density [kg.m-3]
H_atm = 11e3 #scale height of Mars atmosphere [m]

# Numerical parameters:
F = 0.5 # Fourier Mesh Number, must be <= 0.5 for stability
m = 10 # Number of layers in upper skin depth [default: 10]
n = 5 # Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5]
b = 20 # Number of skin depths to bottom layer [default: 20]

# Accuracy of temperature calculations
# The model will run until the change in temperature of the bottom layer
# is less than DTBOT over one diurnal cycle
DTSURF = 0.01 # surface temperature accuracy [K]
DTBOT = DTSURF # bottom layer temperature accuracy [K]
NYEARSEQ = 1 # equilibration time [orbits]
NPERDAY = 24 # minimum number of time steps per diurnal cycle
Ddmdt = 1e-7 #CO2 frost flux accuracy [kg.m-2.s-1]

# NumPy is needed for various math operations
import numpy as np

# Pandas is needed to read the Mars optical depth file
# interp2d is used to interpolate optical depth based on latitude and Ls
import pandas as pd
from scipy.interpolate import interp2d

# MatPlotLib and Pyplot are used for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Methods for calculating solar angles from orbits
import orbits_slopes

# Planets database
import planets

from numba import jit

# Models contain the profiles and model results
class model(object):
    
    # Initialization
    def __init__(self, planet=planets.Mars, lat=0, ndays=1, nu=0, alpha=0, beta=0, f_IR=0.04, f_scat=0.02, u_wind=3, \
                 b=0.2, ice=False, atmExt=False, elev=-4.23, Gamma=700., obliq=25.19*(np.pi/180.)):
        
        #If surface has a non-zero slope, get modeled temperatures of horizontal surface for its contribution to IR emission
        if (np.abs(alpha) > 0):
            self.T_land, self.m_CO2_land, self.Qs_land  = landTemp(planet, lat, ndays, nu, f_IR, f_scat, ice, atmExt, Gamma)
            self.landcount = 0
        
        self.T_land_min = 200. #diurnal minimum of horizontal surface temperature
        
        # Initialize
        self.planet = planet
        self.lat = lat
        self.Sabs = self.planet.S
        self.r = self.planet.rAU # solar distance [AU]
        self.nu = nu # orbital true anomaly [rad]
        self.nuout = nu # orbital true anomaly [rad] at which simulation starts
        self.nudot = np.float() # rate of change of true anomaly [rad/s]
        self.dec = np.float() # solar declination [rad]
        self.alpha = alpha #slope of surface [rad] (positive slopes are pole-facing)
        self.beta = beta #azimuth of slope surface [rad]
        self.hnoon = np.float() #hour angle in previous simulation run
        self.c_noon = np.float() #cosine of solar zenith angle at noon of most recent day in simulation
        self.f_IR = f_IR #fraction of noontime insolation that contributes to atmospheric IR surface flux
        self.f_scat = f_scat # fraction of scattered light from atmosphere
        self.u_wind = u_wind #wind speed for forced water ice sublimation [m.s-1]
        self.T_atm = 200. #Temperature of atmosphere that will affect H2O sublimation
        self.b = b #parameter that affects dependence of T_atm on T_land_min and T_land
        self.ice = ice #is the surface of interest ice?
        self.h = np.float() #hour angle
        self.tau = np.float()
        self.extCoeff = np.float()
        self.atmExt = atmExt # whether or not to include atmospheric extinction. default False
        self.Ls = self.nu + (self.planet.Lp*180./np.pi)
        self.elev = elev
        self.Gamma = Gamma #Thermal Inertia of Surface
        self.planet.obliquity = obliq
        
        # Initialize Mars Optical Depth and Atmospheric Temperature Functions
        self.OpDepth = MarsOpticalDepth()
        self.AtmTemp = MarsAtmTemps()
        
        # Initialize arrays
        self.Qs = np.float() # surface flux
        self.Q_solar = np.float # solar flux
        self.Q_IR = np.float # atmospheric IR emission
        self.Q_scat = np.float # atmospheric scattering
        self.Q_land = np.float # surface IR flux
        self.c = np.float() #cosSolarZenith
        
        # Initialize model profile
        self.profile = profile(planet, lat, u_wind, ice, Gamma)
        
        # Model run times
        # Equilibration time -- TODO: change to convergence check
        self.equiltime = NYEARSEQ*planet.year - \
                        (NYEARSEQ*planet.year)%planet.day
        self.equildone = 0
        # Run time for output
        self.endtime = self.equiltime + ndays*planet.day
        self.t = 0.
        self.dt = getTimeStep(self.profile, self.planet.day)
        # Check for maximum time step
        self.dtout = self.dt
        dtmax = self.planet.day/NPERDAY
        if self.dt > dtmax:
            self.dtout = dtmax
        
        # Array for output temperatures and local times
        self.N_steps = np.int( (ndays*planet.day)/self.dtout )
        self.N_day = np.int( (planet.day)/self.dtout ) #number of steps in a day
        self.N_z = np.size(self.profile.z)
        self.T = np.zeros([self.N_steps, self.N_z])
        self.Qst = np.zeros([self.N_steps]) #Qs as a funciton of t
        self.Q_solart = np.zeros([self.N_steps])
        self.Q_IRt = np.zeros([self.N_steps])
        self.Q_scatt = np.zeros([self.N_steps])
        self.Q_landt = np.zeros([self.N_steps])
        self.m_CO2t = np.zeros([self.N_steps])
        self.dmdtt = np.zeros([self.N_steps])
        self.T_land_mint = np.zeros([self.N_steps])
        self.T_atmt = np.zeros([self.N_steps])
        self.m_H2Ot = np.zeros([self.N_steps])
        self.m_H2O_free = np.zeros([self.N_steps])
        self.m_H2O_forced = np.zeros([self.N_steps])
        self.e_sat = np.zeros([self.N_steps])
        self.e_vap = np.zeros([self.N_steps])
        self.Patm = np.zeros([self.N_steps])
        self.ht = np.zeros([self.N_steps])
        self.ct = np.zeros([self.N_steps]) #cosSolarZenith as a funciton of t
        self.lt = np.zeros([self.N_steps])
        self.nut = np.zeros([self.N_steps])
        self.Lst = np.zeros([self.N_steps])
        self.taut = np.zeros([self.N_steps])
        self.extCoefft = np.zeros([self.N_steps])
        self.f = np.zeros([self.N_steps]) #energy balance term
    
    def run(self):
        
        # Equilibrate the model
        while (self.t < self.equiltime):
            self.advance()
            if (np.abs(self.alpha) > 0):
                if (self.landcount == self.T_land.size):
                    self.landcount = 0 #reset T_land index if it exceeds size of T_land
        self.equildone = 1
        # Run through end of model and store output
        self.dt = self.dtout
        self.t = 0. # reset simulation time
        self.landcount = 0 #reset T_land index so T_land simulation times align with slope simulation times
        self.profile.m_H2O = 0. #reset mass of H2O ice lost
        self.nu = self.nuout
        print ('START SIM: ', self.nu)
        for i in range(0,self.N_steps):
            self.advance()
            self.T[i,:] = self.profile.T # temperature [K]
            self.Qst[i] = self.Qs # Surface flux
            self.Q_solart[i] = self.Q_solar
            self.Q_IRt[i] = self.Q_IR
            self.Q_scatt[i] = self.Q_scat
            self.Q_landt[i] = self.Q_land
            self.m_CO2t[i] = self.profile.m_CO2
            self.dmdtt[i] = self.profile.dmdt
            self.T_land_mint[i] = self.T_land_min
            self.T_atmt[i] = self.T_atm
            self.m_H2Ot[i] = self.profile.m_H2O
            self.m_H2O_free[i] = self.profile.m_free
            self.m_H2O_forced[i] = self.profile.m_forced
            self.e_sat[i] = self.profile.e_sat
            self.e_vap[i] = self.profile.e_vap
            self.Patm[i] = self.profile.Patm
            self.ht[i] = self.h
            self.ct[i] = self.c # Surface flux
            self.lt[i] = self.t/self.planet.day*24.0 # local time [hr]
            self.nut[i] = self.nu
            self.Lst[i] = self.Ls
            self.taut[i] = self.tau
            self.extCoefft[i] = self.extCoeff
            self.f[i] = self.profile.f
        print ('END SIM: ', self.nu)
            
    def advance(self):
        self.updateOrbit()
        self.surfFlux()
        
        if (self.planet.name == 'Mars' and self.ice==True):
            if (self.profile.m_CO2 <= 0 and self.profile.dmdt < 0):
                self.profile.update_T_ice(self.dt, self.Ls, self.elev, self.Qs, self.planet.Qb, self.T_atm)
                self.profile.m_CO2 = 0
                self.profile.dmdt = 0
            elif (self.lat > 0 and self.profile.T[0] <= 147.): ######TO DO: CALCULATE frost point
                self.profile.T[0] = 147.
                self.profile.CO2Flux_ice(self.dt, self.Ls, self.elev, self.Qs, self.planet.Qb, self.T_atm)
            elif (self.lat <= 0 and self.profile.T[0] <= 143.):
                self.profile.T[0] = 143.
                self.profile.CO2Flux_ice(self.dt, self.Ls, self.elev, self.Qs, self.planet.Qb, self.T_atm)
            else:
                self.profile.update_T_ice(self.dt, self.Ls, self.elev, self.Qs, self.planet.Qb, self.T_atm)
        
        elif (self.planet.name == 'Mars' and self.ice==False):
            if (self.profile.m_CO2 <= 0 and self.profile.dmdt < 0):
                self.profile.update_T(self.dt, self.Qs, self.planet.Qb)
                self.profile.m_CO2 = 0
                self.profile.dmdt = 0
            elif (self.lat > 0 and self.profile.T[0] <= 147.):
                self.profile.T[0] = 147.
                self.profile.CO2Flux(self.dt, self.Qs, self.planet.Qb)
            elif (self.lat <= 0 and self.profile.T[0] <= 143.):
                self.profile.T[0] = 143.
                self.profile.CO2Flux(self.dt, self.Qs, self.planet.Qb)
            else:
                self.profile.update_T(self.dt, self.Qs, self.planet.Qb)
        
        else:
            self.profile.update_T(self.dt, self.Qs, self.planet.Qb)
        
        
        self.profile.update_cp()
        self.profile.update_k()
        self.t += self.dt # Increment time
    
    def updateOrbit(self):
        orbits_slopes.orbitParams(self)
        self.nu += self.nudot * self.dt
    
    # Surface heating rate
    # May include solar and infrared contributions, reflectance phase function
    def surfFlux(self):
        h = orbits_slopes.hourAngle(self.t, self.planet.day) # hour angle
        self.h = h
        c = orbits_slopes.cosSlopeSolarZenith(self.lat, self.dec, h, self.alpha, self.beta) # cosine of incidence angle
        cosz = orbits_slopes.cosSolarZenith(self.lat, self.dec, h) #cosine of incidence angle for horizontal surface
        self.c = c
        if (h < self.hnoon):
            #noon cosSolarZenith reset every day for use in atmospheric IR flux term
            self.c_noon = orbits_slopes.cosSolarZenith(self.lat, self.dec, h)
        self.hnoon = h
        
        i = np.arccos(c) # solar incidence angle [rad]
        a = self.planet.albedoCoef[0]
        b = self.planet.albedoCoef[1]
        
        #Albedo
        #######TO DO: add albedo of water ice!
        if (self.profile.m_CO2 > 0):
            f = (1.0 - A_CO2)
        elif (self.ice == True):
            f = (1.0 - A_NPLD)
        else:
            f = (1.0 - albedoVar(self.planet.albedo, a, b, i))
        
        #Calculation of Ls
        Ls = (self.nu + self.planet.Lp)*180./np.pi
        if (Ls >= 360.):
            Ls = Ls - 360.
        self.Ls = Ls
        #Visible Optical Depths
        tau = self.OpDepth(Ls, self.lat*180./np.pi)
        self.tau = tau
        
        #Insolation and Visible Atmospheric Scattering
        #0.04 term comes from maximum path length being limited by radius of curvature of planet (Aharonson and Schorghofer 2006)
        if (self.planet.name == 'Mars' and self.atmExt == True and cosz > 0.04):
            #Insolation
            Q_solar = f * self.Sabs * (self.r/self.planet.rAU)**-2 * c * np.exp(-tau/cosz)
            self.extCoeff = np.exp(-tau/cosz)
            
            #Visible Atmospheric Scattering
            self.f_scat = 1 - np.exp(-0.9*tau/cosz)
            Q_scat = 0.5 * self.f_scat * f * self.Sabs * (self.r/self.planet.rAU)**-2 * cosz * np.cos(self.alpha/2.)**2
            
        elif (self.planet.name == 'Mars' and self.atmExt == True and cosz <= 0.04):
            #Insolation
            Q_solar = f * self.Sabs * (self.r/self.planet.rAU)**-2 * c * np.exp(-tau/0.04)
            self.extCoeff = np.exp(-tau/0.04)
            
            #Visible Atmospheric Scattering
            self.f_scat = 1 - np.exp(-0.9*tau/0.04)
            Q_scat = 0.5 * self.f_scat * f * self.Sabs * (self.r/self.planet.rAU)**-2 * cosz * np.cos(self.alpha/2.)**2
        else:
            Q_solar = f * self.Sabs * (self.r/self.planet.rAU)**-2 * c
            
            if (cosz > 0):
                Q_scat = 0.5 * 0.02 * self.Sabs * \
                         (self.r/self.planet.rAU)**-2 * np.cos(self.alpha/2.)**2
            else:
                Q_scat = 0
        self.Q_solar = Q_solar
        self.Q_scat = Q_scat
        
        #Atmospheric IR emission
        T_IR_atm = self.AtmTemp(Ls, self.lat*180./np.pi)
        tau_IR = tau*0.14
        eps = 1 - np.exp(-tau_IR)
        if (self.planet.name == 'Mars' and self.atmExt == True):
            Q_IR = self.f_IR * self.Sabs * \
                  (self.r/self.planet.rAU)**-2 * np.cos(self.alpha/2.)**2 * self.c_noon + eps*sigma*T_IR_atm**4
        else:
            Q_IR = self.f_IR * self.Sabs * \
                  (self.r/self.planet.rAU)**-2 * np.cos(self.alpha/2.)**2 * self.c_noon
        
        self.Q_IR = Q_IR
        
        #Atmospheric scattering when sun is above horizon. Half of scattered light is lost to space.
        #if (cosz > 0):
        #    self.f_scat = 1 - np.exp(-0.9*tau)
        #    Q_scat = 0.5 * self.f_scat * self.Sabs * \
        #             (self.r/self.planet.rAU)**-2 * np.cos(self.alpha/2.)**2
        #else:
        #    Q_scat = 0
        #self.Q_scat = Q_scat
        
        #IR flux contribution from nearby horizontal surfaces
        if (np.abs(self.alpha) > 0):
            Q_land = np.sin(self.alpha/2.)**2 * self.planet.emissivity * sigma * self.T_land[self.landcount]**4
            
            if (self.landcount > int(self.N_day/2) and self.T_land[self.landcount] <= \
                np.amin(self.T_land[self.landcount-int(self.N_day/2):self.landcount+int(self.N_day/2)])):
                self.T_land_min = self.T_land[self.landcount]

            self.T_atm = self.T_land_min**self.b * self.T_land[self.landcount]**(1-self.b)
            self.landcount += 1
        else:
            Q_land = 0
        self.Q_land = Q_land
        
        self.Qs = Q_solar + Q_IR + Q_scat + Q_land
        

class profile(object):
    """
    Profiles are objects that contain the model layers
    
    The profile class defines methods for initializing and updating fields
    contained in the model layers, such as temperature and conductivity.
    
    """
    
    def __init__(self, planet=planets.Moon, lat=0, u_wind=0, ice=False, Gamma=700):
        
        self.planet = planet
        self.lat = lat
        self.u_wind = u_wind
        self.T_bl = np.float() #boundary layer temperature
        self.T_atm = np.float()
        self.Ls = np.float()
        self.elev = np.float()
        self.Patm = np.float()
        self.WaterVolMix = MarsWaterVolumeMixingRatio()
        self.ice = ice
        self.Gamma = Gamma
        self.e_sat = np.float()
        self.e_vap = np.float()
        
        # The spatial grid
        self.emissivity = planet.emissivity
        ks = planet.ks
        kd = planet.kd
        rhos = planet.rhos
        rhod = planet.rhod
        H = planet.H
        cp0 = planet.cp0
        kappa = ks/(rhos*cp0)
        
        self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b)
        self.nlayers = np.size(self.z) # number of model layers
        self.dz = np.diff(self.z)
        self.d3z = self.dz[1:]*self.dz[0:-1]*(self.dz[1:] + self.dz[0:-1])
        self.g1 = 2*self.dz[1:]/self.d3z[0:] # A.K.A. "p" in the Appendix
        self.g2 = 2*self.dz[0:-1]/self.d3z[0:] # A.K.A. "q" in the Appendix
        
        # Initialize temperature profile
        self.init_T(planet, lat)
        
        # Initialize heat capacity profile
        self.update_cp()
        
        if (ice == True):
            # Ice thermophysical properties
            self.rho = rho_ice*np.ones_like(self.z)
            self.kc = (Gamma**2)/(self.rho*self.cp)
            print ('self.kc[0]:', self.kc[0])
        else:
            # Thermophysical properties
            self.kc = kd - (kd-ks)*np.exp(-self.z/H)
            self.rho = rhod - (rhod-rhos)*np.exp(-self.z/H)
            
        # Initialize conductivity profile
        self.update_k()
        
        # Initialize CO2 mass balance
        self.dmdt = -1. #change in CO2 areal mass density over time
        self.m_CO2 = np.float() #CO2 frost areal mass density [kg.m-2]
        
        # Initialize H2O ice mass balance
        self.m_sub = np.float() #this is the rate of change of areal H2O ice mass
        self.m_H2O = np.float()
        self.m_forced = np.float()
        self.m_free = np.float()
        
        #Track energy balance
        self.f = np.float()
        
    # Temperature initialization
    def init_T(self, planet=planets.Moon, lat=0):
        self.T = np.zeros(self.nlayers) \
                 + T_eq(planet, lat)
    
    # Heat capacity initialization
    def update_cp(self):
        if (self.ice == True):
            self.cp = heatCapacityIce(self.planet, self.T)
        else:
            self.cp = heatCapacity(self.planet, self.T)
    
    # Thermal conductivity initialization (temperature-dependent)
    def update_k(self):
        self.k = thermCond(self.kc, self.T)
    
    ##########################################################################
    # Core thermal computation                                               #
    # dt -- time step [s]                                                    #
    # Qs -- surface heating rate [W.m-2]                                     #
    # Qb -- bottom heating rate (interior heat flow) [W.m-2]                 #
    ##########################################################################
    def update_T(self, dt, Qs=0, Qb=0):
        
        # Coefficients for temperature-derivative terms
        alpha = self.g1*self.k[0:-2]
        beta = self.g2*self.k[1:-1]
        
        # Temperature of first layer is determined by energy balance
        # at the surface
        surfTemp(self, Qs)
        
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1] = self.T[1:-1] + dt/(self.rho[1:-1]*self.cp[1:-1]) * \
                     ( alpha*self.T[0:-2] - \
                       (alpha+beta)*self.T[1:-1] + \
                       beta*self.T[2:] )
     ##########################################################################   
    
    def update_T_ice(self, dt, Ls, elev, Qs=0, Qb=0, T_atm=0):
        self.Ls = Ls
        self.elev = elev
        self.Patm = P_atm(self.Ls, self.elev)
        
        # Coefficients for temperature-derivative terms
        alpha = self.g1*self.k[0:-2]
        beta = self.g2*self.k[1:-1]
        
        self.T_atm = T_atm
        self.T_bl = 0.5*(self.T[0] + self.T_atm)
        
        self.e_sat = satVaporPressureIce(self)
        self.e_vap = self.WaterVolMix(self.Ls, self.lat*180./np.pi) * self.Patm #waterVaporPressure(self)
        
        # Temperature of first layer is determined by energy balance
        # at the surface
        surfTempIce(self, Qs, dt)
        
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1] = self.T[1:-1] + dt/(self.rho[1:-1]*self.cp[1:-1]) * \
                     ( alpha*self.T[0:-2] - \
                       (alpha+beta)*self.T[1:-1] + \
                       beta*self.T[2:] )
    
    def CO2Flux(self, dt, Qs=0, Qb=0):
        Ts = self.T[0]
        dmdt = self.dmdt
        delta_dmdt = 2.*Ddmdt

        while (np.abs(delta_dmdt) > Ddmdt):
            x = self.emissivity*sigma*Ts**3
            y = 0.5*thermCond(self.kc[0], Ts)/self.dz[0]

            # f is the function whose zeros we seek
            f = x*Ts - Qs - y*(-3*Ts+4*self.T[1]-self.T[2]) - L_CO2*dmdt
            self.f = f
            # fp is the first derivative w.r.t. 'dmdt', which is the change in areal mass density of CO2 frost per time step       
            fp = -L_CO2

            # Estimate of the temperature increment
            delta_dmdt = -f/fp
            dmdt += delta_dmdt
        # Update mass of CO2 frost
        self.m_CO2 = self.m_CO2 + dmdt*dt
        self.dmdt = dmdt
        
        # Coefficients for temperature-derivative terms
        alpha = self.g1*self.k[0:-2]
        beta = self.g2*self.k[1:-1]
        
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1] = self.T[1:-1] + dt/(self.rho[1:-1]*self.cp[1:-1]) * \
                     ( alpha*self.T[0:-2] - \
                       (alpha+beta)*self.T[1:-1] + \
                       beta*self.T[2:] )
        
    def CO2Flux_ice(self, dt, Ls, elev, Qs=0, Qb=0, T_atm=0):
        self.Ls = Ls
        self.elev = elev
        self.Patm = P_atm(self.Ls, self.elev)
        Ts = self.T[0]
        dmdt = self.dmdt
        delta_dmdt = 2.*Ddmdt
        
        self.T_atm = T_atm
        self.T_bl = 0.5*(self.T[0] + self.T_atm)
        
        self.e_sat = satVaporPressureIce(self)
        self.e_vap = self.WaterVolMix(self.Ls, self.lat*180./np.pi) * self.Patm #waterVaporPressure(self)
        
        m_forced = iceSubForced(self)
        m_free = iceSubFree(self)
        self.m_forced = -m_forced
        self.m_free = -m_free
        self.m_sub = -(m_forced + m_free)
        self.m_H2O = self.m_H2O + self.m_sub*dt
        #print ('CO2, H2O:', self.m_H2O)
        
        while (np.abs(delta_dmdt) > Ddmdt):
            x = self.emissivity*sigma*Ts**3
            y = 0.5*thermCond(self.kc[0], Ts)/self.dz[0]

            # f is the function whose zeros we seek
            f = x*Ts - Qs - y*(-3*Ts+4*self.T[1]-self.T[2]) - L_CO2*dmdt - L_H2O*self.m_sub
            self.f = f
            # fp is the first derivative w.r.t. 'dmdt', which is the change in areal mass density of CO2 frost per time step       
            fp = -L_CO2

            # Estimate of the temperature increment
            delta_dmdt = -f/fp
            dmdt += delta_dmdt
        # Update mass of CO2 frost
        self.m_CO2 = self.m_CO2 + dmdt*dt
        self.dmdt = dmdt
        
        # Coefficients for temperature-derivative terms
        alpha = self.g1*self.kc[0:-2]
        beta = self.g2*self.kc[1:-1]
        
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1] = self.T[1:-1] + dt/(self.rho[1:-1]*self.cp[1:-1]) * \
                     ( alpha*self.T[0:-2] - \
                       (alpha+beta)*self.T[1:-1] + \
                       beta*self.T[2:] )
    
    # Simple plot of temperature profile
    def plot(self):
        ax = plt.axes(xlim=(0,400),ylim=(np.min(self.z),np.max(self.z)))
        plt.plot(self.T, self.z)
        ax.set_ylim(1.0,0)
        plt.xlabel('Temperature, $T$ (K)')
        plt.ylabel('Depth, $z$ (m)')
        mpl.rcParams['font.size'] = 14

#---------------------------------------------------------------------------
"""

The functions defined below are used by the thermal code.

"""
#---------------------------------------------------------------------------

# Thermal skin depth [m]
# P = period (e.g., diurnal, seasonal)
# kappa = thermal diffusivity = k/(rho*cp) [m2.s-1]
def skinDepth(P, kappa):
    return np.sqrt(kappa*P/np.pi)

# The spatial grid is non-uniform, with layer thickness increasing downward
def spatialGrid(zs, m, n, b):
    dz = np.zeros(1) + zs/m # thickness of uppermost model layer
    z = np.zeros(1) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (z[i] < zmax):
        i += 1
        h = dz[i-1]*(1+1/n) # geometrically increasing thickness
        dz = np.append(dz, h) # thickness of layer i
        z = np.append(z, z[i-1] + dz[i]) # depth of layer i
    
    return z

# Solar incidence angle-dependent albedo model
# A0 = albedo at zero solar incidence angle
# a, b = coefficients
# i = solar incidence angle
def albedoVar(A0, a, b, i):
    return A0 + a*(i/(np.pi/4))**3 + b*(i/(np.pi/2))**8

# Radiative equilibrium temperature at local noontime
def T_radeq(planet, lat):
    return ((1-planet.albedo)/(sigma*planet.emissivity) * planet.S * np.cos(lat))**0.25

# Equilibrium mean temperature for rapidly rotating bodies
def T_eq(planet, lat):
    return T_radeq(planet, lat)/np.sqrt(2)

# Heat capacity of regolith (temperature-dependent)
# This polynomial fit is based on data from Ledlow et al. (1992) and
# Hemingway et al. (1981), and is valid for T > ~10 K
# The formula yields *negative* (i.e. non-physical) values for T < 1.3 K
def heatCapacity(planet, T):
    c = planet.cpCoeff
    return np.polyval(c, T)

def heatCapacityIce(planet, T):
    c = planet.ice_cpCoeff
    return np.polyval(c, T)

# Temperature-dependent thermal conductivity
# Based on Mitchell and de Pater (1994) and Vasavada et al. (2012)
def thermCond(kc, T):
    return kc*(1 + R350*T**3)

# Surface temperature calculation using Newton's root-finding method
# p -- profile object
# Qs -- heating rate [W.m-2] (e.g., insolation and infared heating)
def surfTemp(p, Qs):
    Ts = p.T[0]
    deltaT = Ts
    
    while (np.abs(deltaT) > DTSURF):
        x = p.emissivity*sigma*Ts**3
        y = 0.5*thermCond(p.kc[0], Ts)/p.dz[0]
    
        # f is the function whose zeros we seek
        f = x*Ts - Qs - y*(-3*Ts+4*p.T[1]-p.T[2])
        p.f = f
        # fp is the first derivative w.r.t. temperature        
        fp = 4*x - \
             3*p.kc[0]*R350*Ts**2 * \
                0.5*(4*p.T[1]-3*Ts-p.T[2])/p.dz[0] + 3*y
        
        # Estimate of the temperature increment
        deltaT = -f/fp
        Ts += deltaT
    # Update surface temperature
    p.T[0] = Ts

# Bottom layer temperature is calculated from the interior heat
# flux and the temperature of the layer above
def botTemp(p, Qb):
    p.T[-1] = p.T[-2] + (Qb/p.k[-2])*p.dz[-1]

def getTimeStep(p, day):
    dt_min = np.min( F * p.rho[:-1] * p.cp[:-1] * p.dz**2 / p.k[:-1] )
    return dt_min

def landTemp(planet, lat, ndays, nu, f_IR, f_scat, ice, atmExt, Gamma):
    land = model(planet=planet, lat=lat, ndays=ndays, nu=nu, alpha=0, beta=0, \
                 f_IR=f_IR, f_scat=f_scat, ice=ice, atmExt=atmExt, Gamma=Gamma)
    land.run()
    T_land = land.T[:,0]
    m_CO2_land = land.m_CO2t
    Qs_land = land.Qst
    return T_land, m_CO2_land, Qs_land

def iceSubForced(p):
    #(Dundas and Byrne 2010)
    e_sat = satVaporPressureIce(p)
    Patm = p.Patm #P_atm(p.Ls, p.elev)
    e_vap = p.WaterVolMix(p.Ls, p.lat*180./np.pi) * Patm #waterVaporPressure(p)
    
    m_forced = M_w * A_drag * p.u_wind * (e_sat - e_vap) / (kb*p.T_bl)
    return m_forced

def waterVaporPressure(p):
    #(Schorghofer and Aharonson 2005)
    #x = p.g * h_vap * rho_liq * (44./18.) / (1 - np.exp(-Hc/H_atm))
    
    #(Marti and Mauersberger 1993)
    x = 10**(-2663.5/p.T_bl + 12.537)
    return x

def satVaporPressureIce(p):
    T = p.T_bl
    
    #(Buck 1981) originally from (Wexler 1977)
    x = np.exp( -5865.3696/T + 22.241033 + 0.013749042*T - \
                      0.34031775e-4*T**2 + 0.26967687e-7*T**3 + 0.6918651*np.log(T) )
    return x

def iceSubFree(p):
    #(Dundas and Byrne 2010)
    Ts = p.T_bl
    e_sat = satVaporPressureIce(p)
    dp = densityRatio(p)
    
    rho_atm = (p.Patm*mc)/(Rg*p.T_atm)
    rho_ave = rho_atm/(1 - 0.5*dp)
    
    n = ((e_sat*mw)/(Rg*Ts))/rho_ave # (Ingersoll 1970)
    D = diffusionCoeff(p)
    g = p.planet.g
    visc = kineViscosity(p)
    
    #(Dundas and Byrne 2010)
    m_free = 0.14 * n * rho_ave * D * np.cbrt( dp * (g/(visc**2)) * (visc/D) )
    return m_free

def diffusionCoeff(p):
    T = p.T_bl
    Patm = p.Patm #P_atm(p.Ls, p.elev)
    
    #(Dundas and Byrne 2010)
    D = 1.387e-5 * (T/273.15)**1.5 * (1e5/Patm)
    return D

def kineViscosity(p):
    T = p.T_bl
    Patm = p.Patm #P_atm(p.Ls, p.elev)
    
    #(Dundas and Byrne 2010)
    x = 1.48e-5 * ((Rg*T)/(mc*Patm)) * ((240.+293.15)/(240.+T)) * (T/293.15)**1.5
    return x

def densityRatio(p):
    e_sat = satVaporPressureIce(p)
    Ts = p.T[0]
    Tatm = p.T_atm
    Patm = p.Patm #P_atm(p.Ls, p.elev)
    e_vap = p.WaterVolMix(p.Ls, p.lat*180./np.pi) * Patm #waterVaporPressure(p)
    #ratio of the density difference between saturated and background atmospheric gases to an atmospheric density
    #(Dundas and Byrne 2010)
    x = mc * (Patm * ((Ts/Tatm) - 1.)) + (mc - mw) * (e_sat - (Ts/Tatm)*e_vap) 
    
    y = 0.5 * ( mc*Patm*((Ts/Tatm) + 1.) - (mc - mw)*((Ts/Tatm)*e_vap + e_sat) )
    
    return x/y

def surfTempIce(p, Qs, dt):
    Ts = p.T[0]
    deltaT = Ts
    
    m_forced = iceSubForced(p)
    m_free = iceSubFree(p)
    p.m_forced = -m_forced
    p.m_free = -m_free
    p.m_sub = -(m_forced + m_free)
    p.m_H2O = p.m_H2O + p.m_sub*dt
    #print ('No CO2, H2O:', p.m_H2O)
    
    while (np.abs(deltaT) > DTSURF):
        x = p.emissivity*sigma*Ts**3
        y = 0.5*thermCond(p.kc[0], Ts)/p.dz[0]
    
        # f is the function whose zeros we seek
        f = x*Ts - Qs - y*(-3*Ts+4*p.T[1]-p.T[2]) - L_H2O*p.m_sub
        p.f = f
        # fp is the first derivative w.r.t. temperature        
        fp = 4*x - \
             3*p.kc[0]*R350*Ts**2 * \
                0.5*(4*p.T[1]-3*Ts-p.T[2])/p.dz[0] + 3*y
        
        # Estimate of the temperature increment
        deltaT = -f/fp
        Ts += deltaT
    # Update surface temperature
    p.T[0] = Ts

def MarsOpticalDepth():
    #Read optical depth data from Mars Climate Database
    df = pd.read_csv('MarsOpticalDepths.txt', delim_whitespace=True, skiprows=10, header=0, names=np.arange(0, 50))
    #Reformat data into a usable one
    df = df.drop(df.index[1])
    df = df.drop(1, axis=1)
    df.columns = np.arange(0, df.shape[1])
    df = df.reset_index(drop=True)
    lat = df.values[0,1:]
    Ls = df.values[1:,0]
    Ls = [float(i) for i in Ls]
    dfn = df.values[1:,1:]
    #Create interpolation function that will return an optical depth for any given lat and Ls (nu)
    x, y = np.meshgrid(Ls, lat)
    f = interp2d(Ls, lat, np.transpose(dfn), kind='cubic')
    return f

def MarsAtmTemps():
    #Read optical depth data from Mars Climate Database
    df = pd.read_csv('MarsAtmTemps.txt', delim_whitespace=True, skiprows=10, header=0, names=np.arange(0, 50))
    #Reformat data into a usable one
    df = df.drop(df.index[1])
    df = df.drop(1, axis=1)
    df.columns = np.arange(0, df.shape[1])
    df = df.reset_index(drop=True)
    lat = df.values[0,1:]
    Ls = df.values[1:,0]
    Ls = [float(i) for i in Ls]
    dfn = df.values[1:,1:]
    #Create interpolation function that will return an optical depth for any given lat and Ls (nu)
    x, y = np.meshgrid(Ls, lat)
    f = interp2d(Ls, lat, np.transpose(dfn), kind='cubic')
    return f

def MarsWaterVolumeMixingRatio():
    #Read optical depth data from Mars Climate Database
    df = pd.read_csv('MarsWaterVolMixing.txt', delim_whitespace=True, skiprows=10, header=0, names=np.arange(0, 50))
    #Reformat data into a usable one
    df = df.drop(df.index[1])
    df = df.drop(1, axis=1)
    df.columns = np.arange(0, df.shape[1])
    df = df.reset_index(drop=True)
    lat = df.values[0,1:]
    Ls = df.values[1:,0]
    Ls = [float(i) for i in Ls]
    dfn = df.values[1:,1:]
    #Create interpolation function that will return an optical depth for any given lat and Ls (nu)
    x, y = np.meshgrid(Ls, lat)
    f = interp2d(Ls, lat, np.transpose(dfn), kind='cubic')
    return f

def P_vik(S, SR=360.50865):
    #Viking 2 Lander Pressure Curve as a function of martian sol (Tillman et al. 1993)
    SYR = 668.59692
    P0 = 8.66344
    Pi = np.array([0.798, 0.613, 0.114, 0.063, 0.018])
    phi = np.array([93.59, -131.37, -67.50, 17.19, 98.84])
    
    P = P0*np.ones_like(S)
    
    for i in range(0, Pi.size):
        P = P + Pi[i] * np.sin( 2.*np.pi*(i+1)*(S-SR)/SYR + (phi[i]*np.pi/180.) )
    
    return P * 100 #convert mbar to Pa

def MJD(Ls, n=0):
    #Conversion from Ls to modified Julian date
    #n is the nth orbit of Mars since the epoch 1874.0
    #(Allison and McEwen 2000 Eq. 14)
    mjd = 51507.5 + 1.90826*(Ls-251) - 20.42*np.sin((Ls-251)*np.pi/180.) + 0.72*np.sin(2.*(Ls-251)*np.pi/180.) + \
                ( 686.9726 + 0.0043*np.cos((Ls-251)*np.pi/180.) - 0.0003*np.cos(2.*(Ls-251)*np.pi/180.) )*(n-66)
    return mjd

def MSD(mjd):
    #Conversion from modified Julian date to Martian Sol Date
    #(Allison and McEwen 2000 Eq. 32)
    k = 0.001
    msd = ( (mjd - 51549.0)/1.02749125 ) + 44796.0 - k - 395.77492432270083
    return msd

def P_atm(Ls, elev=-4.23):
    #Atmospheric pressure at a given Ls and elevation [km] based on Viking 2 Lander data and hydrostatic assumption
    mjd = MJD(Ls)
    msd = MSD(mjd)
    Pv = P_vik(msd)
    P0 = Pv/np.exp(4.23/11.1)
    
    P = P0*np.exp(-elev/11.1)
    return P