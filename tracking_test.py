from ParticleFilter import Map, ParticleFilter
from config.mapInfo import map_info

import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def main(args):
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


    _map = Map(map_info[env])
    pf = ParticleFilter(100, _map, args.epsilon)

    f = open(args.input.replace(".csv", "_result.txt"), 'w')
    f.write("time, x, y\n")

    for i in tqdm(range(times.shape[0])):
        if i != times.shape[0] - 1:
            dt = times[i+1]-times[i]
            pf.propagate(dt[0])

        pf.update(features[i])
        f.write("%f, %f, %f\n"%(pf.time, pf.prediction.x, pf.prediction.y))
        pf.resample()

        pf.render("tmp", args.ray)
    
    # output_gif = args.input.replace("/","_").replace("trck_features_", "results/tracking/").replace(".csv", ".gif")
    output_gif = args.input.replace(".csv", "_result.gif")    
    # if not os.path.isdir("results/tracking"):
    #     os.mkdir("results/tracking")
    os.system("convert -delay {} -loop 0 ./tmp/*.png {}".format(int(dt*100),output_gif))
    os.system("rm -rf tmp")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ParticleFilter")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.1)
    args = parser.parse_args()
    main(args)