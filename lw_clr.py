"""
Methods for calculating clear sky downward longwave radiation, and atmospheric emissivity.

RELEASE NOTES
    Version 1.0 Written by Mark Raleigh (raleigh@ucar.edu), Oct 2013)
    Version 2.0 Overhauled by Mark Raleigh (Feb 2015) to have structure inputs
        and to correct errors in the Flerchinger formulas
    Version 3.0 Rewritten in python by Steven Pestana (spestana@uw.edu, Apr 2020)
        adapted from original MATLAB script (from Mark Raleigh, Ryan Currier)
        and changed output to downward longwave radiance rather than emissivities

INPUTS
    Ta = air temperature [K] 
    RH = relative humidity [%]
    elev = elevation [m]
    Qsi = incoming shortwave radiation [W m^-2]
      
METHODS
    From Flerchinger et al. (2009):
        Angstrom (1918)
        Brunt (1932)
        Brutsaert (1975)
        Garratt (1992)
        Idso and Jackson (1969) (Idso-1)
        Idso (1981) (Idso-2)
        Iziomon et al. (2003)
        Keding (1989)
        Niemela et al. (2001)
        Prata (1996)
        Satterlund (1979)
        Swinbank (1963)
        Dilley and O'Brien (1998)
    From Juszak and Pellicciotti (2013):
        Maykut and Church (1973) 
        Konzelmann et al. (1994)
        Dilley and O'Brien (A) (1998)
    Others:
        Campbell and Norman (1998) as cited by Walter et al (2005)
        Long and Turner (2008) - based on Brutsaert (1975)
        Ohmura (1982) as cited by Howard and Stull 2013
        Efimova (1961) as cited by Key et al (1996)

"""

# To do:
# - Fix longturner2008() so we can include Qsi as an input for that method
# - Fix konzelmann1994() (I suspect I have a typo or coefficients are wrong?)
# - Add other methods?


#-------------------------------------------------------#

import pandas as pd
import numpy as np
import xarray as xr

#----------------- Ensemble function -------------------#
def ensemble(Ta,RH,elev):
    # Run the ensemble and add results of each run to a dataset:
    lw_ensemble = xr.Dataset()
    lw_ensemble['lclr_angstrom1918'] = angstrom1918(Ta,RH)
    lw_ensemble['lclr_brunt1932'] = brunt1932(Ta,RH)
    lw_ensemble['lclr_brutsaert1975'] = brutsaert1975(Ta,RH)
    lw_ensemble['lclr_garratt1992'] = garratt1992(Ta,RH)
    lw_ensemble['lclr_idsojackson1969'] = idsojackson1969(Ta,RH)
    lw_ensemble['lclr_idso1981'] = idso1981(Ta,RH)
    lw_ensemble['lclr_iziomon2003'] = iziomon2003(Ta,RH,elev)
    lw_ensemble['lclr_keding1989'] = keding1989(Ta,RH)
    lw_ensemble['lclr_niemela2001'] = niemela2001(Ta,RH)
    lw_ensemble['lclr_prata1996'] = prata1996(Ta,RH)
    lw_ensemble['lclr_satturlund1979'] = satturlund1979(Ta,RH)
    lw_ensemble['lclr_swinbank1963'] = swinbank1963(Ta)
    lw_ensemble['lclr_dilleyobrien1998'] = dilleyobrien1998(Ta,RH)
    lw_ensemble['lclr_maykutchurch1973'] = maykutchurch1973(Ta)
    #lw_ensemble['lclr_konzelmann1994'] = konzelmann1994(Ta,RH)
    lw_ensemble['lclr_campbellnorman1998'] = campbellnorman1998(Ta)
    #lw_ensemble['lclr_longturner2008'] = longturner2008(Ta,RH,Qsi)
    lw_ensemble['lclr_ohmura1982'] = ohmura1982(Ta)
    lw_ensemble['lclr_efimova1961'] = efimova1961(Ta,RH)
    
    # Take mean, min, max of all methods:
    lclr_mean = lw_ensemble.to_array(dim='new').mean('new')
    lclr_min = lw_ensemble.to_array(dim='new').min('new')
    lclr_max = lw_ensemble.to_array(dim='new').max('new')
    lw_ensemble = lw_ensemble.assign(lclr_mean=lclr_mean)
    lw_ensemble = lw_ensemble.assign(lclr_min=lclr_min)
    lw_ensemble = lw_ensemble.assign(lclr_max=lclr_max)
    return lw_ensemble

#----------------- Ancillary functions -------------------# 
def vap_pres(Ta,RH):
    '''Calculate actual vapor pressure [kPa] from T [K] and RH [%]
    Clausius-Clapeyron e_sat in mb (hPa) from Murray 1967'''
    sat_vap_pres = 6.1078 * np.exp((17.2693882*(Ta-273.15)) / (Ta-35.86)) # saturated vapor pressure [hPa]
    return (RH/100 * sat_vap_pres) / 10 # actual vapor pressure [kPa]

def prata1996_w(Ta,RH):
    '''from Prata 1996'''
    return (465 * vap_pres(Ta,RH)) / Ta # note: corrected equation constant to be 465 instead of 4650 (error in Flerchinger Table 1 footnote c)

def L_sb(e,T):
    '''Calculate radiance [W m^-2] given an emissivity and temperature [K] w/ Stefan-Boltzmann law'''
    sb = 5.67 * (10**-8)        # Stefan-Boltzmann constant (J/s/m^2/K^4)
    return e * sb * T**4
    
def e_sb(L,T):
    '''Calculate emissivity given a radiance [W m^-2] and temperature [K] w/ Stefan-Boltzmann law'''
    sb = 5.67 * (10**-8)        # Stefan-Boltzmann constant (J/s/m^2/K^4)
    return Lclr/(sb*(Ta**4)) 
     
    

#----------------- Clear-sky longwave functions -------------------#
def angstrom1918(Ta,RH):
    '''Angstrom (1918)'''
    a = 0.83
    b = 0.18 
    c = 0.67  # note: corrected coefficient c to be 0.67 instead of 0.067 (error in Flerchinger Table 1)
    e_clr = (a - b*10**(-1*c*vap_pres(Ta,RH)))
    return L_sb(e_clr,Ta)

def brunt1932(Ta,RH):
    '''Brunt (1932)'''
    a = 0.52
    b = 0.205
    e_clr = (a + b*np.sqrt(vap_pres(Ta,RH)))
    return L_sb(e_clr,Ta)

def brutsaert1975(Ta,RH):
    '''Brutsaert (1975)'''
    a = 1.723
    b = (1/7)
    e_clr = a * (vap_pres(Ta,RH)/Ta)**(b);
    return L_sb(e_clr,Ta)

def garratt1992(Ta,RH):
    '''Garratt (1992)'''
    a = 0.79
    b = 0.17
    c = 0.96
    e_clr = a - b*np.exp(-1*c*vap_pres(Ta,RH));
    return L_sb(e_clr,Ta)

def idsojackson1969(Ta,RH):
    '''Idso and Jackson (1969) (Idso-1)'''
    a = 0.261
    b = 0.00077
    e_clr = 1 - a*np.exp(-1*b*(Ta-273.16)**2);
    return L_sb(e_clr,Ta)

def idso1981(Ta,RH):
    '''Idso (1981) (Idso-2)'''
    a = 0.70
    b = 5.95 * 10**-4
    c = 1500
    e_clr = a + (b*vap_pres(Ta,RH)*np.exp(c/Ta));
    return L_sb(e_clr,Ta)

def iziomon2003(Ta,RH,elev):
    '''Iziomon et al. (2003)'''
    a = 0.35
    b = 100
    c = 212 # a,b,c for a low land site (were c = elevation in m)
    d = 0.43
    e = 115
    f = 1489 # d,e,f for a higher elevation site (were f = elevation in m)
    Mxz = (d-a)/(f-c) # change in first parameters with elevation change
    Myz = (e-b)/(f-c) # change in second parameters with elevation change
    X = Mxz*(elev - c) + a
    Y = Myz*(elev - c) + b
    e_clr = 1 - X*np.exp(-Y*vap_pres(Ta,RH)/Ta)
    return L_sb(e_clr,Ta)

def keding1989(Ta,RH):
    '''Keding (1989)'''
    a = 0.92
    b = 0.7
    c = 1.2
    e_clr = a - b*10**(-1*c*vap_pres(Ta,RH));
    return L_sb(e_clr,Ta)

def niemela2001(Ta,RH):
    '''Niemela et al. (2001)'''
    a = 0.72
    b = 0.09
    c = 0.2
    d = 0.76
    e_clr = a + b*(vap_pres(Ta,RH)-c);
    e_clr[vap_pres(Ta,RH)<c] = a - d*(vap_pres(Ta,RH)[vap_pres(Ta,RH)<c]-c);
    return L_sb(e_clr,Ta)

def prata1996(Ta,RH):
    '''Prata (1996)'''
    a = 1.2
    b = 3
    c = 0.5
    e_clr = 1 - (1+prata1996_w(Ta,RH))*np.exp(-1*(a + b*prata1996_w(Ta,RH))**c);
    return L_sb(e_clr,Ta)

def satturlund1979(Ta,RH):
    '''Satturlund (1979)'''
    a = 1.08
    b = 2016
    e_clr = a*(1-np.exp(-((10*vap_pres(Ta,RH))**(Ta/b))));  # note: corrected so that b paramter is in the exponent ^(Ta/b), error in Flerchinger Table 1
    return L_sb(e_clr,Ta)

def swinbank1963(Ta):
    '''Swinbank (1963)'''
    a = 5.31 * (10**-13)
    b = 6
    L_clr = a*(Ta**b);
    return L_clr

def dilleyobrien1998(Ta, RH):
    '''Dilley and O’Brien (1998)'''
    a = 59.38
    b = 113.7
    c = 96.96
    L_clr = a + b*(Ta/273.16)**6 + c*np.sqrt(prata1996_w(Ta,RH)/2.5) # note: corrected to be 2.5 instead of 25 (error in Flerchinger Table 1)
    return L_clr

def maykutchurch1973(Ta):
    '''Maykut and Church (1973) from Juszak and Pellicciotti (2013)'''
    a = 0.7855
    e_clr = a;
    return L_sb(e_clr,Ta)

#def konzelmann1994(Ta,RH):
#    '''Konzelmann et al. (1994) from Juszak and Pellicciotti (2013)'''
#    a = 0.23 # from Juszak and Pellicciotti (2013), clear-sky emittance of a completely dry atmosphere (calcualted by LOWTRAN7)
#    b = 0.484 # value from Juszak and Pellicciotti (2013), original value from Konzelmann et al. = 0.443
#    c = 1.8 #  value from Juszak and Pellicciotti (2013), original value from Konzelmann et al. = 1/8
#    e_clr = a + b*(1000*vap_pres(Ta,RH)/Ta)**c; # factor of 1000 to convert vap_pres [kPa] to vap_pres [Pa]
#    return L_sb(e_clr,Ta)

def campbellnorman1998(Ta):
    '''Campbell and Norman (1998) as cited by Walter et al (2005)'''
    a = 0.72 - 0.005*273.15 # modified so Ta is in (K) instead of (C) ....should be -0.6458 after the maths...
    b = 0.005
    e_clr = a + b*Ta;
    return L_sb(e_clr,Ta)

#def longturner2008(Ta,RH,Qsi):
#    '''Long and Turner (2008) - based on Brutsaert (1975)'''
#    # simplify the k-value determination - use one value for day, one for
#    # night (where night/day is based on threshold in Qsi data)
#    kd = 1.18 # daytime k value, varies diurnally (and likely spatially) - see Fig 2 
#    kn = 1.28 # nighttime k value
#    a = np.mean([1.39e-11, 3.36e-12, 1.47e-11, 4.07e-12]) # "a" coefficient - average of four datasets
#    b = np.mean([4.8769, 5.1938, 4.8768, 5.1421]) # "b" coefficient - average of four datasets
#    c = (1/7) # Brutsaert (1975) exponent
#    
#    
#    k_array = np.ones_like(Ta) * kd # set all values to daytime k
#    k_array[Qsi<50] = kn      # set night time values to night k
#
#    Ccoeff = k_array + a*(RH)**b # here RH should be a percentage, not fractional
#    e_clr = Ccoeff * (vap_pres(Ta,RH)*10/Ta)**c;      # vap_pres(Ta,TH) times 10 to convert from kPa to mb
#    return L_sb(e_clr,Ta)

def ohmura1982(Ta):
    '''Ohmura (1982) as cited by Howard and Stull 2013'''
    a = 8.733 * 10**-3
    b = 0.788
    e_clr = a * Ta**(b);
    return L_sb(e_clr,Ta)

def efimova1961(Ta,RH):
    '''Efimova (1961) as cited by Key et al (1996)'''
    a = 0.746
    b = 0.0066 * 10 # multiply by 10 to account for vap_pres in mb isntead of kPa
    e_clr = a + b*vap_pres(Ta,RH);
    return L_sb(e_clr,Ta)

