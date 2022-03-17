#Title: Handpicked Coupler values 5.75 GHZ
from Khalil import KhalilModel_2
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.dpi"] = 300 #Makes the plots render in 300dpi



#%% Loading data and importing data
# Loading in the data -> See Import.md for info on how to perform an import from sonnet.
cwd = os.getcwd()
datafolder = cwd + r'\datafolder\MWOffice_Output_csv'
Group = 1;
Appc_to_be_squared = 50.0;
Oeff = 4.50 ;

Group_str = "{:1.0f}".format(Group)
Appc_to_be_squared_str = "{:2.1f}".format(Appc_to_be_squared)
Oeff_str = "{:1.2f}".format(Oeff)
print(Group_str)
print(Appc_to_be_squared_str)
print(Oeff_str)

#---> Insert your datapath here! You can uncomment lines to use different data
measurement = '\MWG'+ Group_str + '_'+ Appc_to_be_squared_str +'_Oeff_' + Oeff_str +'.txt'#4 Coupler data
print(measurement)
#---> skip controls the amount of header lines to ignore
skip = 15;#Old data 13
#---> End
datafile = datafolder + measurement
data = np.genfromtxt(datafile, delimiter="\t", skip_header=skip)

f = data[:,0]
# If data[:,2] is |S(2,1)| in [dB] --> 
S21_dB = data[:,2] 

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
ax.set_title('G' + Group_str + '|' +'$\sqrt{A}$ =' +Appc_to_be_squared_str + '|' + 'O_{eff}= ' + Oeff_str + '->' +'Q_{c} = ' + "{:2.2e}".format(Qc))
ax.set_xlim([f0-10*f0/Ql, f0+10*f0/Ql])
ax.set_xlabel('F (GHz)')
ax.set_ylabel('|S21| [dB]')
ax.legend()

#Save Each figure:
plt.savefig(cwd + "\Images\SiC_final_fitting\\" + measurement[1:-4] + '.png')

print('Oc_overlap = ' + Oeff_str + '\n Ql = '+str(Ql)+'\n Qc = ' + str(Qc) + '\n Qi = ' + str(Qi) + '\n f0 = ' + str(f0) ) 
data_str = measurement + '\t' + Group_str + '\t' + Appc_to_be_squared_str + '\t' + Oeff_str + '\t' + "{:1.5f}".format(f0) + '\t' + "{:1.3e}".format(Ql) + '\t' + "{:1.3e}".format(Qc) + '\t' + "{:1.3e}".format(Qi) 
text = [ data_str ]
with open(cwd + "\Output_data\\" + 'SiC_fitting_data_append.csv','a') as file:
    for line in text:
        file.write(line)
        file.write('\n')
# Now we want to save these result to a text file!










