from ase.io.lammpsrun import read_lammps_dump
import numpy as np

fnn = read_lammps_dump("fnn/dump.fnn", specorder=['W'], index=1)
dr1 = read_lammps_dump("dr_0.1/dump.interp", specorder=['W'], index=0)
dr2 = read_lammps_dump("dr_0.05/dump.interp", specorder=['W'], index=0)
dr3 = read_lammps_dump("dr_0.01/dump.interp", specorder=['W'], index=0)
dr4 = read_lammps_dump("dr_0.005/dump.interp", specorder=['W'], index=0)
dr5 = read_lammps_dump("dr_0.001/dump.interp", specorder=['W'], index=0)
dr6 = read_lammps_dump("dr_0.0005/dump.interp", specorder=['W'], index=0)
dr7 = read_lammps_dump("dr_0.0001/dump.interp", specorder=['W'], index=0)

rmse1 = np.sqrt(np.mean((fnn.get_forces() - dr1.get_forces())**2))
rmse2 = np.sqrt(np.mean((fnn.get_forces() - dr2.get_forces())**2))
rmse3 = np.sqrt(np.mean((fnn.get_forces() - dr3.get_forces())**2))
rmse4 = np.sqrt(np.mean((fnn.get_forces() - dr4.get_forces())**2))
rmse5 = np.sqrt(np.mean((fnn.get_forces() - dr5.get_forces())**2))
rmse6 = np.sqrt(np.mean((fnn.get_forces() - dr6.get_forces())**2))
rmse7 = np.sqrt(np.mean((fnn.get_forces() - dr7.get_forces())**2))

np.savetxt("rmse.txt", np.array([rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse7]))
