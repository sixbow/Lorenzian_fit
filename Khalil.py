'''
Python implementation of the Matlab script that fits |S21| data to the Khalil model.
I checked that it gives the same results as the Matlab script. 

I experiment with a new version in the V2 functions.
'''
import lmfit as lmf
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

def read_complex_data(resonator_name, data_path, headers=1, fcol=0, recol=3, imcol=4, magcol=1, tcol=2):
    '''
    * Reads S21 data from a csv file, needs filename convention: "`resonator`_`power`.csv". Example file name: MS1_20.csv
    * The power is read from the file name, power should be in dBm. No minus sign in the name. 
    * First column of CSV file should be frequencies [Hz], second column should be |S21| [dB].
    ''' 
    f = []
    S21 = []
    mag_dB = []
    powers = []
    theta = []
    measurements = glob.glob(data_path + "/" + resonator_name + "*_S21lowT.csv")
    for measurement in measurements:
        powers.append(-int(re.split("dBm", re.split("[_.]", measurement)[1])[0]))
        data = np.genfromtxt(measurement, delimiter=",", skip_header=headers)
        f.append(data[:,fcol])
        S21.append(data[:,recol] + 1j * data[:,imcol])
        mag_dB.append(data[:,magcol])
        theta.append(data[:,tcol])
    powers = np.array(powers)
    f = np.array(f)
    S21 = np.array(S21)
    mag_dB = np.array(mag_dB)
    theta = np.array(theta)
    return powers, f, S21, mag_dB, theta
 
def estimate_initials(f, mag_dB):
    mag = 10**(mag_dB/20)
    a_guess = np.average([mag[-1], mag[0]]) 
    b_guess = (mag_dB[-1] - mag_dB[0]) / (f[-1] - f[0])
    mag_c = mag / a_guess
    min_index = np.argmin(mag_c)
    f0_guess = f[min_index]
    S21min_guess = mag_c[min_index]
    bwindex = np.argmin(np.abs(mag_c**2-((S21min_guess**2+1)/2)))
    if bwindex-min_index==0:
        Ql_guess=f0_guess/(np.abs(2*(f[min_index+1]-f[min_index])))
    else:
        Ql_guess=f0_guess/(np.abs(2*(f[bwindex]-f[min_index])))
        
    Qi_guess = Ql_guess/S21min_guess
    Qc_guess = (Ql_guess*Qi_guess)/(Qi_guess - Ql_guess)
    BW_guess = f0_guess/Ql_guess
    return Ql_guess, f0_guess, Qc_guess, a_guess, b_guess, BW_guess

def Khalil_func_1(f, f0, Ql, Qc_re, dw, a):
    return 20*np.log10(np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * (f - f0) / f0)))*a)
        
        
class KhalilModel_1(lmf.model.Model):
    def __init__(self, f, mag_dB, *args, **kwargs):
        super().__init__(Khalil_func_1, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag_dB)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a', min = 0) 
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                  a=a_guess, b=b_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)

def Khalil_func_2(f, f0, Ql, Qc_re, dw, a, b):
    return 20*np.log10(np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * (f - f0) / f0)))*np.abs((b*(f-f0)+a)))
        
        
class KhalilModel_2(lmf.model.Model):
    def __init__(self, f, mag_dB, *args, **kwargs):
        super().__init__(Khalil_func_2, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag_dB)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a', min = 0) 
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                  a=a_guess, b=b_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)
        
def Khalil_func_phi_2(f, f0, Ql, Qc_norm, phi, a, b):
    Qc = Qc_norm * np.exp(1j*phi)
    return 20*np.log10(np.abs(1 - (Ql * Qc**-1 / (1 + 2j * Ql * (f - f0) / f0)))*(b*(f-f0)+a))
        
class KhalilModel_phi_2(lmf.model.Model):
    def __init__(self, f, mag_dB, *args, **kwargs):
        super().__init__(Khalil_func_phi_2, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag_dB)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_norm', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('phi', min = -np.pi/2, max=np.pi/2)
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a', min = 0) 
        #self.set_param_hint('x', min = 0, expr='Qc_norm*cos(phi) - Ql', vary=True) # Creates the constraint: Re{Qc} - Ql > 0
        
        params = self.make_params(Ql=Ql_guess, Qc_norm=Qc_norm_guess, phi=0, f0=f0_guess, 
                                  a=a_guess, b=b_guess, x=Qc_norm_guess - Ql_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)
        
def Khalil_func_phi_1(f, f0, Ql, Qc_norm, phi, a):
    Qc = Qc_norm * np.exp(1j*phi)
    return 20*np.log10(np.abs(1 - (Ql * Qc**-1 / (1 + 2j * Ql * (f - f0) / f0)))*a)
        
class KhalilModel_phi_1(lmf.model.Model):
    def __init__(self, f, mag_dB, *args, **kwargs):
        super().__init__(Khalil_func_phi_1, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, _, BW_guess = estimate_initials(f, mag_dB)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_norm', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('phi', min = -np.pi/2, max=np.pi/2)
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a', min = 0) 
        #self.set_param_hint('x', min = 0, expr='Qc_norm*cos(phi) - Ql', vary=True) # Creates the constraint: Re{Qc} - Ql > 0
        
        params = self.make_params(Ql=Ql_guess, Qc_norm=Qc_norm_guess, phi=0, f0=f0_guess, 
                                  a=a_guess, x=Qc_norm_guess - Ql_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)
        

        
def inspect_results(f, powers, mag_dBs, results, datanames, models, plot_powers=None, plot_idx=None, sort_idx=True, shift=0):
    '''
    Helper function to inspect an individual result, either provide plot_powers or plot_idx
    '''
    if plot_idx is None:
        plot_idx = []
        for plot_power in plot_powers:
            plot_idx.append(np.argmin(np.abs(powers - plot_power)))
    elif sort_idx == True:
        plot_idx = np.argsort(powers)[plot_idx]
        print('plot_idx: ' + str(plot_idx))
        
    fig, ax = plt.subplots()
    if len(plot_idx) == 1:
        ax.set_title(models[0].name)
    ax.set_xlabel('F (GHz)'), 
    ax.set_ylabel('|S21| (dBm)')
    ax.tick_params(axis='both', direction='in', which='both')
    for i in plot_idx:
        fit = models[i].eval(params=results[i].params, f=f[i])
        ax.scatter(f[i] + i*shift, mag_dBs[i], s=2, alpha=0.6, label=datanames[i] + ', Pread: ' + str(powers[i]) + ' dBm')
        ax.plot(f[i] + i*shift, fit, alpha=0.8, color='k')
        f0 = results[i].params['f0'].value
        ax.scatter(f0 + i*shift, models[i].eval(params=results[i].params, f=f0), alpha=0.8, color='k')
    
        print('Power: ' + str(powers[i]) + ' dBm')
        print('Dataset: ' + datanames[i]+ '\n')
        results[i].params.pretty_print()
        print('\n\n')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),
          ncol=1)
        
def inspect_result(power, powers, results, datanames, idx=None):
    '''
    Helper function to inspect an individual result
    '''
    if idx == None:
        idx = np.argmin(np.abs(powers - power))
    results[idx].plot(xlabel='F (GHz)', ylabel='|S21| (dBm)')
    print('Power: ' + str(powers[idx]) + ' dBm')
    print('Dataset: ' + datanames[idx]+ '\n')
    results[idx].params.pretty_print()
