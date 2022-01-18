from Khalil import KhalilModel_2
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.dpi"] = 300 #Makes the plots render in 300dpi



cwd = os.getcwd()
datafolder = cwd + r'\datafolder'
Ocoupler = 14;
Ocoupler_overlap = 4;
Ocoupler_str = str(Ocoupler)
Ocoupler_overlap_str = str(Ocoupler_overlap)
skip = 15
measurement = datafolder + '\Full_modelV2_1_0VerifyVersion2_1Zoom54_55.csv'
data = np.genfromtxt(measurement, delimiter=",", skip_header=skip)

# #%% Loading data and importing data
# # Loading in the data -> See Import.md for info on how to perform an import from sonnet.
# cwd = os.getcwd()
# datafolder = cwd + r'\datafolder'
# Ocoupler = 14;
# Ocoupler_overlap = 4;
# Ocoupler_str = str(Ocoupler)
# Ocoupler_overlap_str = str(Ocoupler_overlap)
# #---> Insert your datapath here! You can uncomment lines to use different data
# #data = '\Full_modelV1_4_0_3Target_T3DataDelete_CouplerParameterization2500Try2Ocoupler' + Ocoupler_str + '.csv'#4 Coupler data
# data = '\Full_modelV2_1_0VerifyVersion2_1Zoom54_55.csv' #Latest data 10-01-2022
# #---> End Insert datapath!
# #---> skip controls the amount of header lines to ignore
# skip = 15;#Old data 13
# #---> End
# datafile = datafolder + data
# datasets = np.genfromtxt(datafile, delimiter=",", skip_header=skip, usecols=[0,5,6])



#%% Loading data and importing data
# Loading in the data -> See Import.md for info on how to perform an import from sonnet.
cwd = os.getcwd()
datafolder = cwd + r'\datafolder'
Ocoupler = 6;
Ocoupler_overlap = Ocoupler ;
Ocoupler_str = str(Ocoupler)
Ocoupler_overlap_str = str(Ocoupler_overlap)
#---> Insert your datapath here! You can uncomment lines to use different data
#measurement = '\Full_modelV1_4_0_3Target_T3DataDelete_CouplerParameterization2500Try2Ocoupler' + Ocoupler_str + '.csv'#4 Coupler data
measurement = '\Full_modelV2_3_0_CouplerCharaterizationHighFreqTry2Oeff' + Ocoupler_str + '.csv'#4 Coupler data

#measurement = '\Full_modelV2_1_0VerifyVersion2_1Zoom54_55.csv' #Latest data 10-01-2022
#---> End Insert datapath!
#---> skip controls the amount of header lines to ignore
skip = 15;#Old data 13
#---> End
datafile = datafolder + measurement
data = np.genfromtxt(datafile, delimiter=",", skip_header=skip)

f = data[:,0]
S21 = data[:,5] + 1j* data[:,6]
S21_dB = 20*np.log10(np.abs(S21))

Model = KhalilModel_2

model = Model(f, S21_dB)
result = model.fit(S21_dB, f=f, params=model.guess)

Ql = result.params['Ql'].value 
Qc = result.params['Qc_re'].value
Qi = result.params['Qi'].value
f0 = result.params['f0'].value

evaluation = model.eval(params=result.params, f=f)

fig, ax = plt.subplots()
ax.scatter(f,S21_dB, s=1, label='Data')
ax.plot(f,evaluation, color='green', label='Fit')
ax.set_title('...'+measurement[(len(measurement)-15):len(measurement)])
ax.set_xlim([f0-10*f0/Ql, f0+10*f0/Ql])
ax.set_xlabel('F (GHz)')
ax.set_ylabel('|S21| [dB]')
ax.legend()
print('Oc_overlap = ' + Ocoupler_overlap_str + '\n Ql = '+str(Ql)+'\n Qc = ' + str(Qc) + '\n Qi = ' + str(Qi) + '\n f0 = ' + str(f0) ) 

