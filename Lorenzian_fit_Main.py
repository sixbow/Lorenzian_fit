#Title: Handpicked Coupler values 5.75 GHZ
from Khalil import KhalilModel_2
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.dpi"] = 300 #Makes the plots render in 300dpi



#%% Loading data and importing data
# Loading in the data -> See Import.md for info on how to perform an import from sonnet.
cwd = os.getcwd()
datafolder = cwd + r'\datafolder'
Ocoupler_overlap = 4.0 ;
Ocoupler_str = str(Ocoupler_overlap)
Ocoupler_overlap_str = "{:2.1f}".format(Ocoupler_overlap)
print(Ocoupler_str)

#---> Insert your datapath here! You can uncomment lines to use different data
measurement = '\MW2500_2-2-2_Oeff4_0' + '.txt'#4 Coupler data
print(measurement)
#---> skip controls the amount of header lines to ignore
skip = 15;#Old data 13
#---> End
datafile = datafolder + measurement
data = np.genfromtxt(datafile, delimiter="\t", skip_header=skip)

f = data[:,0]
#S21 = data[:,2] #+ 1j* data[:,6] #This is the |S(2,1)| [dB] dataset.
S21_dB = data[:,2] # Moeten we hier 2* 

fig1, ax1 =  plt.subplots()
ax1.plot(f,S21_dB, color='red', label='Raw Data')

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

