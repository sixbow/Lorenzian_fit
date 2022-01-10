import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

#%% Loading data and importing data
# Loading in the data -> See Import.md for info on how to perform an import from sonnet.
cwd = os.getcwd()
datafolder = cwd + r'\datafolder'
Ocoupler = 14;
Ocoupler_overlap = 4;
Ocoupler_str = str(Ocoupler)
Ocoupler_overlap_str = str(Ocoupler_overlap)
#---> Insert your datapath here! You can uncomment lines to use different data
#data = '\Full_modelV1_4_0_3Target_T3DataDelete_CouplerParameterization2500Try2Ocoupler' + Ocoupler_str + '.csv'#4 Coupler data
data = '\Full_modelV2_1_0VerifyVersion2_1Zoom54_55.csv' #Latest data 10-01-2022
#---> End Insert datapath!
#---> skip controls the amount of header lines to ignore
skip = 15;#Old data 13
#---> End
datafile = datafolder + data
datasets = np.genfromtxt(datafile, delimiter=",", skip_header=skip, usecols=[0,5,6])

#Bruno Khalil import 
from Khalil import KhalilModel_2
Model = KhalilModel_2
frequencies = datasets[:,0]
S21_dB = 20*np.log10(np.absolute(np.power(datasets[:,1],2) + 1j*np.power(datasets[:,2],2)))
model = Model(frequencies, S21_dB)
result = model.fit(S21_dB, f=frequencies, params=model.guess)
Ql_bruno2 = result.params['Ql'].value
Qc_bruno2 = result.params['Qc_re'].value
Qi_bruno2 = result.params['Qi'].value
f0_bruno2 = result.params['f0'].value
evaluation = model.eval(params=result.params, f=frequencies)

plt.plot(frequencies,S21_dB)

# Begin Guess
mag_sq_dB = 2*S21_dB
minimum_idx = np.argmin(S21_dB)
f_res_guess = frequencies[minimum_idx]
S21min_sq_dB_guess = mag_sq_dB[minimum_idx]
S21min_sq_guess = 10**(S21min_sq_dB_guess/20)
S21min_guess = np.sqrt(S21min_sq_guess)
HM_sq_dB_guess = S21min_sq_dB_guess + 20*np.log10(2)
F_HM_min_guess_idx = np.argmin(np.abs(mag_sq_dB[frequencies < f_res_guess] - HM_sq_dB_guess))
F_HM_max_guess_idx = np.argmin(np.abs(mag_sq_dB[frequencies > f_res_guess] - HM_sq_dB_guess))
FWHM_guess = frequencies[frequencies > f_res_guess][F_HM_max_guess_idx] - frequencies[frequencies < f_res_guess][F_HM_min_guess_idx]
Ql_guess = f_res_guess / FWHM_guess
# End Guess
#%%

# S21min_guess = ;
Ql_guess = 20000;
# f_res_guess = 5.43157E9;

p0 = [S21min_guess, Ql_guess, f_res_guess]


def Lorentzian_sq_dB(f, S21min, Q, f_res):
    '''
    |S21|**2, |S21| expressed in [dB]. Symmetric Lorentzian resonance dip.
    '''
    return 20*np.log10(1 + (S21min**2 - 1)/(1 + (2*Q*(f - f_res)/f_res)**2))

def fit_lor(dataset_f, dataset_S):
    popt, pcov = curve_fit(Lorentzian_sq_dB, frequencies, 2*S21_dB, p0=p0)

    return popt

# Begin fit + Plot Lorenzian fit and sonnet data
output = fit_lor(frequencies, S21_dB)
S_21_min_fit = output[0]
Q_fit = output[1]
f_res_fit = output[2]
fig, ax = plt.subplots()
ax.scatter(frequencies, 2*S21_dB, s=8,c='b')
ax.plot(frequencies, Lorentzian_sq_dB(frequencies,S_21_min_fit,Q_fit,f_res_fit),c='r')
plt.ylabel('$S_{21}\:\:    (dB)$')
plt.xlabel('f      (GHz)')
ax.legend(['Lorenzian fit','Sonnet Data'])
ax.set_xlim(5.44,5.45)
ax.set_title("Lorenzian fit of the S21 Parameters of Full Sonnet. @ $O_{coupler overlap}$ = "+Ocoupler_overlap_str)
plt.savefig('Images/Lorenzian_fit_O_coupler'+Ocoupler_str+'.pdf')
plt.savefig('Images/Lorenzian_fit_O_coupler'+Ocoupler_str+'.png',format='png', dpi=1200)

# Calculate relevant parameters
Qi = Q_fit/S_21_min_fit
Qc = 1/((1/Q_fit)-(1/Qi))
print('A_coupler = '+str(Ocoupler*(10**-6)*(4*(10**-6)))+'|\nQ:'+str(Q_fit)+'|\n'+'Qi:'+ str(Qi) + '|\nQc:'+str(Qc)+'|\nS21_min:'+ str(S_21_min_fit) + '|\nf_res:'+ str(f_res_fit))
