import numpy as np
import pandas as pd


def read_thermo(filename):
    with open(filename) as fp:
        read_data = False
        data = []
        for line in fp:
            line = line.strip()
            if not read_data:
                if line.startswith("Step          CPU            Temp"):
                    read_data = True
                    continue
            else:
                data.append([float(x) for x in line.split()[2:]])
                if len(data) == 1001:
                    break
        return np.array(data)


fnn = read_thermo('dynamic_ref/log.lammps')
dr1 = read_thermo('dynamic/dr_0.1/log.lammps')
dr2 = read_thermo('dynamic/dr_0.05/log.lammps')
dr3 = read_thermo('dynamic/dr_0.01/log.lammps')
dr4 = read_thermo('dynamic/dr_0.005/log.lammps')
dr5 = read_thermo('dynamic/dr_0.001/log.lammps')
dr6 = read_thermo('dynamic/dr_0.0005/log.lammps')
dr7 = read_thermo('dynamic/dr_0.0001/log.lammps')

drlist = [dr1, dr2, dr3, dr4, dr5, dr6, dr7]

labels = ["Temp", "KinEng", "PotEng", "TotEng", "Press", "Volume", "Pxx", 
          "Pyy", "Pzz"]

results = {key: [] for key in labels}

for idx in range(len(drlist)):
    for col in range(fnn.shape[1]):
        mae = np.abs(fnn[:, col] - drlist[idx][:, col]).mean()
        rmse = np.sqrt(np.square(fnn[:, col] - drlist[idx][:, col]).mean())
        results[labels[col]].append(rmse)

df = pd.DataFrame(results, index=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
print(df.to_string(float_format="% 18.12f"))
