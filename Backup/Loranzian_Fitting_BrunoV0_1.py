import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
#from uncertainties import ufloat
#from uncertainties.umath import tanh
from scipy.integrate import quad
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
  measurements = glob.glob(data_path +''/'' + resonator_name + "*_S21lowT.csv")
  for measurement in measurements:
    powers.append(-int(re.split("dBm", re.split("[_.]", measurement)[1])[0]))
    datasets.append(np.genfromtxt(measurement, delimiter=",", skip_header=1, usecols=[0,1]))
  powers = np.array(powers)
  datasets = np.array(datasets)
  return datasets, powers
def Lorentzian_sq_dB(f, S21min, Q, f_res):
  '''
  |S21|**2, |S21| expressed in [dB]. Symmetric Lorentzian resonance dip.
  '''
  return 20*np.log10(1 + (S21min**2 - 1)/(1 + (2*Q*(f - f_res)/f_res)**2))
def fit_S21(dataset, power, T = None, FF = None, baseline_range=0.05):
  '''
  Fit |S21|^2 to a symmetric Lorentzian resonance dip in dB space. 
  FF is the filling fraction. Power is only 
  '''
  frequencies = dataset[:,0]
  mag_sq_dB_uncorr = 2*dataset[:,1] # 'uncorr' refers to no baseline correcten applied yet
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
  S21min = ufloat(popt[0], stds[0])
  Ql = ufloat(popt[1], stds[1])
  Qi = Ql/S21min
  Qc = (Qi*Ql)/(Qi-Ql)
  f_res = ufloat(popt[2], stds[2])
  # Calculate loss tangent
  if T != None and FF != None:
    h = 6.6261 * 1e-34
    k_B = 1.3806e-23
    tandelta = 1 / Qi / tanh(h*f_res*1e9/(2*k_B*T)) / FF
  else:
    tandelta = 1 / Qi
  fig, ax = plt.subplots(1,3, figsize = (25,7))
  ax[0].scatter(frequencies, mag_sq_dB_uncorr, s=1)  
  ax[0].scatter(frequencies[line_window], mag_sq_dB_uncorr[line_window], s=1, color='g')  
  ax[0].plot(frequencies, line(frequencies, *popt_baseline), color='g', label='Baseline')
  ax[0].set_ylabel('$|S21|^2$ [mag], uncorrected', fontsize=14)
  ax[0].set_xlabel('F [GHz]', fontsize=14)
  ax[0].legend(fontsize=14)
  ax[0].set_title('VNA output power: ' + str(power) + ' [dBm]')
  ax[1].scatter(frequencies, mag_sq_dB, s=1)  
  ax[1].plot(frequencies, mag_sq_dB_filtered, label="Filtered", color='orange')
  ax[1].scatter(f_res_guess, S21min_sq_dB_guess, color='green', label='$f_r$, $S21_\mathrm{min}$ guesses')
  ax[1].plot(np.array([frequencies[frequencies < f_res_guess][F_HM_min_guess_idx], 
              frequencies[frequencies > f_res_guess][F_HM_max_guess_idx]]), 
         np.array([HM_sq_dB_guess, HM_sq_dB_guess]), color='green', label='FWHM guess')
  ax[1].set_ylabel('$|S21|^2$ [mag]', fontsize=14)
  ax[1].set_xlabel('F [GHz]', fontsize=14)
  ax[1].legend(fontsize=14)
  ax[1].set_title('VNA output power: ' + str(power) + ' [dBm]')
  ax[2].scatter(frequencies, mag_sq_dB, s=1)  
  ax[2].scatter(f_res.n, 20*np.log10(S21min.n**2), color='k')
  ax[2].plot(frequencies, Lorentzian_sq_dB(frequencies, *popt), color = 'k', label='Fit')
  ax[2].set_ylabel('$|S21|^2$ [dB]', fontsize=14)
  ax[2].set_xlabel('F [GHz]', fontsize=14)
  ax[2].legend(fontsize=14)
  ax[2].set_title("$Q_l$: " + "{:.2e}".format(Ql.n) + ", $Q_i$: " + "{:.2e}".format(Qi.n) + ", $Q_c$: " + "{:.2e}".format(Qc.n) + ", $f_r$: " + "{:.4e}".format(f_res.n), fontsize=14)
  plt.close(fig)
  return {'S21min':S21min, 'Ql':Ql, 'Qi':Qi, 'tandelta': tandelta, 'Qc':Qc, 'f_res':f_res, 'power':power, 'pcov':pcov, 'fig':fig}