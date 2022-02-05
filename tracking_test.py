from re import T
from ParticleFilter import Map, ParticleFilter
from config.mapInfo import map_info

import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

import math
import time

N = 20


def main(args):
    log = open("log.txt", 'a')
    print("file : %s"%(args.input))
    log.write("file : %s\n"%(args.input))
    all_ASPLE, all_ASPLE_var = {}, {}
    # for N_PARTICLE in [10,25,50,100]:  
    for N_PARTICLE in [10,25,50,100]:  

        try:
            os.system("rm -rf tmp")
        except:
            pass
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")

        _,env,*_ = args.input.split('/')

        features = pd.read_csv(args.input)
        i_max = 1
        for i, col in enumerate(features.columns):
            if 'feat' in col:
                i_max = i + 1 
        features, times = np.split(features.to_numpy(), [i_max], axis=1)
        features = np.array(features, dtype=np.float32)

        ###############################
        # Likelihood shaping function #
        ###############################
        features = np.power(features,4)
        features = features - np.quantile(features, 0.75, axis=1).reshape(-1,1)

        total_pred_x = []
        total_pred_x_var = []

        for n in range(N):
            _map = Map(map_info[env])
            pf = ParticleFilter(N_PARTICLE, _map, args.epsilon)

            f = open(args.input.replace(".csv", "_result.txt"), 'w')
            pred_t, pred_x, pred_x_var = [], [], []
            start = time.time()
            for i in range(times.shape[0]):
                
                if i != times.shape[0] - 1:
                    dt = times[i+1]-times[i]
                    pf.propagate(dt[0])

                pf.update(features[i])
                
                f.write("%f, %f, %f\n"%(pf.time, pf.prediction.x, pf.prediction.y))
                pred_t.append(pf.time)
                pred_x.append(pf.prediction.x)
                
                pf.resample()

                # Variance check
                x_var = [particle.x for particle in pf.particles][:int((pf.n)*(1-pf.epsilon))] # only non-random sampled particles
                x_var = np.std(x_var)
                pred_x_var.append(x_var)
                
                if args.render:
                    pf.render("tmp", args.ray)
            
            if n == 0:
                print("%d particle execution time : %.2f ms" %(N_PARTICLE, (time.time()-start)*1000/times.shape[0]))
                log.write("%d_particle_execution_time(ms) : %.2f\n" %(N_PARTICLE, (time.time()-start)*1000/times.shape[0]))

            # output_gif = args.input.replace("/","_").replace("trck_features_", "results/tracking/").replace(".csv", ".gif")
            if args.render:
                output_gif = args.input.replace(".csv", "_result.gif")    
                # if not os.path.isdir("results/tracking"):
                #     os.mkdir("results/tracking")
                os.system("convert -delay {} -loop 0 ./tmp/*.png {}".format(int(dt*100),output_gif))
                os.system("rm -rf tmp")
                os.mkdir("tmp")
            f.close()

            total_pred_x.append(pred_x)
            total_pred_x_var.append(pred_x_var)
        
        
        # ground truth data
        dt = dt[0]
        gt = np.loadtxt(args.input.replace("out_multi.csv", "ground_truth.txt"), dtype=float, delimiter=",")
        total_pred_x = np.array(total_pred_x)[:,int((gt[0,0]-2)/dt)+1:]
        total_pred_x_var = np.array(total_pred_x_var)
        end_time = (total_pred_x.shape[1]*dt-2)
        gt_grad = (gt[1,1]-gt[0,1])/(gt[1,0]-gt[0,0])
        real_gt = {'time': [0, end_time], 'x_position': [gt[0,1],end_time*gt_grad+gt[0,1]]}
        regress_gt = {'time':[-2, end_time], 'x_position':[-2*gt_grad+gt[0,1],end_time*gt_grad+gt[0,1]]}
        
        # maximum DoA baseline
        start2 = time.time()
        doa = (np.argmax(features, axis=1)-90) * math.pi / 180
        doa_f = 12 * np.tan(doa)
        doa_f = doa_f[int((gt[0,0]-2)/dt):]
        doa_f = doa_f[:total_pred_x.shape[1]]
        doa_f = {'time': np.linspace(-2, end_time, doa_f.shape[0]), 'x_position': doa_f}
        print("DOA time : %f"%((time.time()-start2)/features.shape[0]*1000))

        # ASPLE
        ASPLE_df = pd.DataFrame(columns=['time','x_position'])
        for t in range(total_pred_x.shape[1]):
            for i in range(total_pred_x.shape[0]):
                ASPLE_df = ASPLE_df.append({'time':t*dt-2, 'x_position':total_pred_x[i,t]}, ignore_index=True)
        all_ASPLE[N_PARTICLE] = ASPLE_df

        # ASPLE variance
        ASPLE_df_var = pd.DataFrame(columns=['time','x_var'])
        for t in range(total_pred_x_var.shape[1]):
            for i in range(total_pred_x_var.shape[0]):
                ASPLE_df_var = ASPLE_df_var.append({'time':t*dt-gt[0,0], 'x_var':total_pred_x_var[i,t]}, ignore_index=True)
        all_ASPLE_var[N_PARTICLE] = ASPLE_df_var

        with plt.style.context(("seaborn-paper",)):
            sns.lineplot(data=real_gt, x='time', y='x_position', color='k', label='Ground truth')
            sns.lineplot(data=regress_gt, x='time', y='x_position', color='k', linestyle="dashed", label='Ground truth regression')
            sns.lineplot(data=doa_f, x='time', y='x_position', color='b', label='Max DoA prediction')
            sns.lineplot(data=ASPLE_df, x='time', y='x_position', color='r', label='Particle filter prediction')
            plt.xlabel("time [sec]")
            plt.ylabel("x position [m]")
            plt.legend()
            plt.savefig(args.input.replace("out_multi.csv", "%d_%d.png"%(N_PARTICLE, N)))
            plt.close()
            # plt.savefig("%d_%d_%s_img.png"%(N_PARTICLE, N, args.input.replace("/","_").strip("_out_multi.csv").strip("cls_features_")))

        # Variance
        with plt.style.context(("seaborn-paper",)):
            sns.lineplot(data=ASPLE_df_var, x='time', y='x_var', color='r', label='Particle filter prediction')
            plt.xlabel("time [sec]")
            plt.ylabel("particle std [m]")
            plt.savefig(args.input.replace("out_multi.csv", "variance_%d_%d.png"%(N_PARTICLE, N)))
            plt.close()

        avg_var = np.mean(np.var(total_pred_x, axis=0))
        avg_mean = np.mean(total_pred_x, axis=0)
        gt_mean = np.array([t*gt_grad+gt[0,1] for t in np.linspace(-2, end_time, avg_mean.shape[0])])
        rmse = mean_squared_error(gt_mean, avg_mean, squared=False)
        rmse_doa = mean_squared_error(gt_mean, doa_f['x_position'], squared=False)
        print("Average variance : ", avg_var)
        print("Center location error : ", rmse)
        print("Center location error for Max DoA : ", rmse_doa)
        log.write("Average_variance : %f\n"%avg_var)
        log.write("Center_location_error : %f\n"%rmse)
        log.write("Center_location_error_for_Max_DoA : %f\n"%rmse_doa)

    log.close()

    # final all    
    with plt.style.context(("seaborn-paper",)):
        sns.lineplot(data=real_gt, x='time', y='x_position', color='k', label='Ground truth')
        sns.lineplot(data=regress_gt, x='time', y='x_position', color='k', linestyle="dashed", label='Ground truth regression')
        sns.lineplot(data=doa_f, x='time', y='x_position', color='b', label='Max DoA prediction')
        for n, ASPLE_df in all_ASPLE.items():
            sns.lineplot(data=ASPLE_df, x='time', y='x_position', label='%d particles'%n)
        plt.xlabel("time [sec]")
        plt.ylabel("x position [m]")
        plt.legend()
        name = args.input.split('/')
        plt.title(name[1] + " " + name[2])
        plt.savefig(args.input.replace("out_multi.csv", "tracking.png"))
        plt.close()

    # variance
    with plt.style.context(("seaborn-paper",)):
        for n, ASPLE_df_var in all_ASPLE_var.items():
            sns.lineplot(data=ASPLE_df_var, x='time', y='x_var', label='%d particles'%n)
            ASPLE_df_var.to_pickle(args.input.replace("out_multi.csv", "%d.pkl"%n))
        plt.xlabel("time [sec]")
        plt.ylabel("particle std [m]")
        plt.legend()
        name = args.input.split('/')
        plt.title(name[1] + " " + name[2])
        plt.savefig(args.input.replace("out_multi.csv", "variance.png"))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ParticleFilter")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)