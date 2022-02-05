import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

if not os.path.isdir("image"):
    os.mkdir("image")


env = ['SA2','SB1','SB2']
direction = ['left','right']
n_particle = ['10','25','50','100']
df_all = {'10':None,'25':None,'50':None,'100':None}
# n_particle = ['100']
# df_all = {'100':None}

for e in env:
    for d in direction:
        rootdir = os.path.join('cls_features',e,d)
        for dirpath, dnames, fnames in os.walk(rootdir):
            for f in fnames:
                for n in n_particle:
                    if f.endswith("%s.pkl"%n):
                        if df_all[n] is None:
                            df_all[n] = pd.read_pickle(os.path.join(dirpath, f))
                        else:
                            df_all[n] = pd.concat([df_all[n], pd.read_pickle(os.path.join(dirpath, f))], ignore_index=True)

        with plt.style.context(("seaborn-paper",)):
            for n, df in df_all.items():
                sns.lineplot(data=df, x='time', y='x_var', label='%s particles'%n)
            plt.xlabel("time [sec]")
            plt.ylabel("particle std [m]")
            plt.legend()
            plt.title(e + " " + d)
            plt.savefig("image/" + e + "-" + d + ".png")
            plt.close()