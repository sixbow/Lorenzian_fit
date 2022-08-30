import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.integrate import quad

def conv_power_quarter_CPW(P_read, e_r, Ql, Qc, l, Fress, W, S, use_Pint=0, output='n', Zres=50, Zfeed=50):
    '''
    Converts readout power [dBm] (or optionally internal power) of a quarter wave CPW resonator to photon number 
    (or optionally electric field at open end, or RMS voltage at open end).
    W is the gap width [m], S is the center line width [m], l is the resonator length [m]. Fress is in GHz.
    
    Calculations are Based on Rami Barends' PhD thesis (photon-detecting superconducting resonators, 2009)
    '''
    h = 6.6261 * 1e-34 # Planck constant
    eps_0 = 8.8542 * 1e-12 # Electrical constant

    f = Fress * 1e9 # Convert GHz to Hz
    eps_eff = (e_r + 1)/2

    P_read_W = 1e-3 * 10**(P_read/10) # Convert from dBm to Watt
    P_int_W = 2/np.pi*Ql**2/Qc*Zfeed/Zres*P_read_W

    # Calculate from pre-calculated P_int instead of P_read
    if use_Pint==1:
        P_int_W = 1e-3 * 10**(P_read/10) # Convert from dBm to Watt

    # elliptical integral 
    def integrand(x,k):
        return 1/np.sqrt((1-x**2)*(1-k**2*x**2))

    k1 = S/(S+2*W) 
    k2 = np.sqrt(1-k1**2)
    K1 = quad(integrand, 0, 1, args=(k1))[0]
    K2 = quad(integrand, 0, 1, args=(k2))[0]

    C = 4*eps_0*eps_eff*K1/K2

    V_3 = np.sqrt(P_int_W * Zres)
    V_r = 2*V_3 # RMS voltage of standing wave at open end.
    U = 0.5*C*V_r**2*l # Energy
    n = U/(h*f) # Number of photons
    E = V_r / W # Electric field at open end
    
    if output == 'n':
        return n
    elif output == 'V':
        return V_r
    elif output == 'E':
        return E
    elif output =='Pint':
        return 30 + 10 * np.log10(P_int_W) # in dBm
    elif output == 'Pread':
        return P_read
    else:
        raise ValueError("The output parameter must be 'n', 'V', or 'E' (defaults to 'n').") 
        
# Half-wavelength microstrip
def conv_power_half_MS(P_read, e_r, Ql, Qc, Fress, l, w, d, Z0 = 50):
    '''
    Converts readout power [dBm] of a half wave microstrip resonator to photon number.
    w is the line width [m], l is the resonator length [m]. Fress is in GHz.
    '''
    f = Fress * 1e9 # Convert from GHz to Hz
    
    P_read_W = 1e-3 * 10**(P_read/10) # Convert from dBm to Watt
    e_0 = 8.8542e-12 # electric constant
    h = 6.6261e-34 # Planck constant
    
    C = e_0 * e_r * w/d # Parallel plate approximation
    E = 1/d * np.sqrt(8*Z0/np.pi * Ql**2/Qc * P_read_W) # From Mazin paper 2010 (https://doi.org/10.1063/1.3314281)
    V_r = d * E # RMS voltage of standing wave at open end
    U = 0.5*C*V_r**2*l # Energy
    n = U/(h*f) # Number of photons
    return n

def tandelta_func(x,tand0,x0,beta,tand_HP):
    '''
    TLS model of loss tangent versus photon number (energy in the resonator). 
    '''
    return tand0 * (1 + x/x0)**(-beta/2) + tand_HP

def log_tandelta_func(x,tand0,x0,beta,tand_HP):
    '''
    Log-space version of TLS model of loss tangent versus photon number (energy in the resonator). 
    '''
    return np.log10(tand0 * (1 + x/x0)**(-beta/2) + tand_HP)

def read_data(resonator_name, datafolder):
    '''
    * Reads S21 data from a csv file, needs filename convention: "`resonator`_`power`.csv". Example file name: MS1_20.csv
    * The power is read from the file name, power should be in dBm. No minus sign in the name. 
    * First column of CSV file should be frequencies [Hz], second column should be |S21| [dB].
    ''' 
    cwd = os.getcwd()
    data_path = cwd + datafolder
    
    datasets = []
    powers = []
    measurements = glob.glob(data_path + "/" + resonator_name + "*_S21lowT.csv")
    for measurement in measurements:
        
    datasets.append(np.genfromtxt(measurement, delimiter=",", skip_header=1, usecols=[0,1]))

    datasets = np.array(datasets)
    return datasets

def line(x,a,b):
    '''
    A line, that is used as a background fit of the S21 data. 
    '''
    return a*x+b

def Lorentzian_sq_dB(f, S21min, Q, f_res):
    '''
    |S21|**2, |S21| expressed in [dB]. Symmetric Lorentzian resonance dip.
    '''
    return 20*np.log10(1 + (S21min**2 - 1)/(1 + (2*Q*(f - f_res)/f_res)**2))

def fit_S21(dataset, power, T = None, FF = None, baseline_range=0.05):
    '''
    Fit |S21|^2 to a symmetric Lorentzian resonance dip in dB space. 
    Baseline correction is performed first, then a parameter guess is done before fitting. 
    
    FF is the filling fraction. Power is only 
    '''
    frequencies = dataset[:,0]
    mag_sq_dB_uncorr = 2*dataset[:,1] # 'uncorr' refers to no baseline correcten applied yet
    
    # Fit baseline in baseline_range (for example baseline_range = 0.05 -> first and last 5% of points are used) 
    line_window = np.concatenate((np.arange(0,len(frequencies)*baseline_range).astype(int), np.arange(len(frequencies)*(1-baseline_range), len(frequencies)).astype(int)))
    popt_baseline, pcov_baseline = curve_fit(line, frequencies[line_window], mag_sq_dB_uncorr[line_window])
    
    # Baseline correction
    mag_sq_dB = mag_sq_dB_uncorr - line(frequencies, *popt_baseline)

    # Guess parameters f_res, S21min, Ql. Gaussian filter is applied to reduce noise before guessing 
    mag_sq_dB_filtered = gaussian_filter1d(mag_sq_dB, 10)
    minimum_idx = np.argmin(mag_sq_dB_filtered)
    f_res_guess = frequencies[minimum_idx]
    S21min_sq_dB_guess = mag_sq_dB_filtered[minimum_idx]
    S21min_sq_guess = 10**(S21min_sq_dB_guess/20)
    S21min_guess = np.sqrt(S21min_sq_guess)
    HM_sq_dB_guess = S21min_sq_dB_guess + 20*np.log10(2)
    F_HM_min_guess_idx = np.argmin(np.abs(mag_sq_dB_filtered[frequencies < f_res_guess] - HM_sq_dB_guess))
    F_HM_max_guess_idx = np.argmin(np.abs(mag_sq_dB_filtered[frequencies > f_res_guess] - HM_sq_dB_guess))
    FWHM_guess = frequencies[frequencies > f_res_guess][F_HM_max_guess_idx] - frequencies[frequencies < f_res_guess][F_HM_min_guess_idx]
    Ql_guess = f_res_guess / FWHM_guess
    
    # Fit Lorentzian in dB space (in mag space S21min will fit to 0)
    p0 = [S21min_guess, Ql_guess, f_res_guess]
    bounds = ((0, 0, 0),(1, np.inf, np.inf))
    popt, pcov = curve_fit(Lorentzian_sq_dB, frequencies, mag_sq_dB, p0=p0, bounds=bounds)
    stds = np.sqrt(np.diag(pcov))

    # Calculate quantities from fit results
    S21min = popt[0]
    Ql = popt[1]
    Qi = Ql/S21min
    Qc = (Qi*Ql)/(Qi-Ql)
    f_res = popt[2]
    
    # Calculate loss tangent
    if T != None and FF != None:
        h = 6.6261 * 1e-34
        k_B = 1.3806e-23
        tandelta = 1 / Qi / np.tanh(h*f_res*1e9/(2*k_B*T)) / FF
    else:
        tandelta = 1 / Qi
    
    fig, ax = plt.subplots(1,3, figsize = (25,7))
    
    #ax[0].scatter(frequencies, mag_sq_dB_uncorr, s=1)  
    #ax[0].scatter(frequencies[line_window], mag_sq_dB_uncorr[line_window], s=1, color='g')  
    #ax[0].plot(frequencies, line(frequencies, *popt_baseline), color='g', label='Baseline')
    #ax[0].set_ylabel('$|S21|^2$ [mag], uncorrected', fontsize=14)
    #ax[0].set_xlabel('F [GHz]', fontsize=14)
    #ax[0].legend(fontsize=14)
    #ax[0].set_title('VNA output power: ' + str(power) + ' [dBm]')

    #ax[1].scatter(frequencies, mag_sq_dB, s=1)  
    #ax[1].plot(frequencies, mag_sq_dB_filtered, label="Filtered", color='orange')
    #ax[1].scatter(f_res_guess, S21min_sq_dB_guess, color='green', label='$f_r$, $S21_\mathrm{min}$ guesses')
    #ax[1].plot(np.array([frequencies[frequencies < f_res_guess][F_HM_min_guess_idx], 
                       #     frequencies[frequencies > f_res_guess][F_HM_max_guess_idx]]), 
                 # np.array([HM_sq_dB_guess, HM_sq_dB_guess]), color='green', label='FWHM guess')
    #ax[1].set_ylabel('$|S21|^2$ [mag]', fontsize=14)
    #ax[1].set_xlabel('F [GHz]', fontsize=14)
    #ax[1].legend(fontsize=14)
    #ax[1].set_title('VNA output power: ' + str(power) + ' [dBm]')

    #ax[2].scatter(frequencies, mag_sq_dB, s=1)  
    #ax[2].scatter(f_res.n, 20*np.log10(S21min.n**2), color='k')
    #ax[2].plot(frequencies, Lorentzian_sq_dB(frequencies, *popt), color = 'k', label='Fit')
    #ax[2].set_ylabel('$|S21|^2$ [dB]', fontsize=14)
    #ax[2].set_xlabel('F [GHz]', fontsize=14)
    #ax[2].legend(fontsize=14)
    #ax[2].set_title("$Q_l$: " + "{:.2e}".format(Ql.n) + ", $Q_i$: " + "{:.2e}".format(Qi.n) + ", $Q_c$: " + "{:.2e}".format(Qc.n) + ", $f_r$: " + "{:.4e}".format(f_res.n), fontsize=14)


    plt.close(fig)
    return {'S21min':S21min, 'Ql':Ql, 'Qi':Qi, 'tandelta': tandelta, 'Qc':Qc, 'f_res':f_res, 'power':power, 'pcov':pcov, 'fig':fig}
