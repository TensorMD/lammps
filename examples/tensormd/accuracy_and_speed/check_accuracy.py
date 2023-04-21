import numpy as np
from ase.io.lammpsrun import read_lammps_dump


fnn = read_lammps_dump('dynamic_ref/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr1 = read_lammps_dump('dynamic/dr_0.1/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr2 = read_lammps_dump('dynamic/dr_0.05/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr3 = read_lammps_dump('dynamic/dr_0.01/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr4 = read_lammps_dump('dynamic/dr_0.005/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr5 = read_lammps_dump('dynamic/dr_0.001/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr6 = read_lammps_dump('dynamic/dr_0.0005/dump.fnn', index=slice(0, None, 1), specorder=['W'])
dr7 = read_lammps_dump('dynamic/dr_0.0001/dump.fnn', index=slice(0, None, 1), specorder=['W'])

f_fnn = np.zeros((1001, 1024, 3))
f_dr1 = np.zeros_like(f_fnn)
f_dr2 = np.zeros_like(f_fnn)
f_dr3 = np.zeros_like(f_fnn)
f_dr4 = np.zeros_like(f_fnn)
f_dr5 = np.zeros_like(f_fnn)
f_dr6 = np.zeros_like(f_fnn)
f_dr7 = np.zeros_like(f_fnn)

for i in range(len(fnn)):
    f_fnn[i] = fnn[i].get_forces()    
    f_dr1[i] = dr1[i].get_forces()
    f_dr2[i] = dr2[i].get_forces()
    f_dr3[i] = dr3[i].get_forces()    
    f_dr4[i] = dr4[i].get_forces()
    f_dr5[i] = dr5[i].get_forces()
    f_dr6[i] = dr6[i].get_forces()
    f_dr7[i] = dr7[i].get_forces()    

print(np.abs(f_fnn - f_dr1).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr1).mean()))
print(np.abs(f_fnn - f_dr2).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr2).mean()))
print(np.abs(f_fnn - f_dr3).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr3).mean()))
print(np.abs(f_fnn - f_dr4).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr4).mean()))
print(np.abs(f_fnn - f_dr5).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr5).mean()))
print(np.abs(f_fnn - f_dr6).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr6).mean()))
print(np.abs(f_fnn - f_dr7).flatten().mean(), np.sqrt(np.square(f_fnn - f_dr7).mean()))
