from ParticleFilter import Map, ParticleFilter_FIX
from config.mapInfo import map_info

import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utils.metric import jaccard


def main(args):
    if not os.path.isdir(args.output) and args.render:
        os.mkdir(args.output)

    _map = Map(map_info[args.env])
    pf = ParticleFilter_FIX(100, _map, args.epsilon)
    pf.init_fixed_particles(*(map_info[args.env]["init_particle"]))

    features = pd.read_csv(args.input)
    i_max = 1
    for i, col in enumerate(features.columns):
        if 'feat' in col:
            i_max = i + 1 # feat0 부터 있으니까 +1 
    features, times = np.split(features.to_numpy(), [i_max], axis=1)
    features = np.array(features, dtype=np.float32)
    label = times[:,0]
    data_id = times[:,1]
    env = times[:,2]


    ###############################
    # Likelihood shaping function #
    ###############################
    features = np.power(features,4)
    features = features - np.quantile(features, 0.75, axis=1).reshape(-1,1)
    

    result = {'left': {'left':0, 'front':0, 'right':0},
            'front':{'left':0, 'front' :0, 'right':0},
            'right':{'left':0, 'front':0, 'right':0}}
    for i in tqdm(range(times.shape[0])):
        if env[i] == args.env and label[i] != 'none':
            pf.init_fixed_particles(*(map_info[args.env]["init_particle"]))
            pf.set_weight([1,0.5,0.5,0.3])
            feature = features[i]
            featlen = feature.shape[0]//2
            feature_L = [feature[:featlen], feature[:featlen]]

            feat = feature_L[0]
            if label[i] == 'front':
                feat = feature_L[1]

            pf.update(feat)
            if args.render:
                pf.render(args.output, env[i]+label[i]+data_id[i])
            pred = pf.predict()
                
            # try
            if label[i] == pred:
                result[label[i]][label[i]] += 1
            else:
                result[label[i]][pred] += 1

    jac = jaccard(result)
    print(args.env, "result")
    print("label:\tN\tJ\tTP\tFP\tFN")
    for key in ['left','front','right']:
        print("%s:\t%d\t%.3f\t%d\t%d\t%d"%(key, jac[key]['N'], jac[key]['J'], jac[key]['TP'], jac[key]['FP'], jac[key]['FN']))
    print('Total accuracy :', jac['accuracy'])

    if not os.path.isdir("results/classification"):
        os.mkdir("results/classification")
    with open("results/classification/result_{}.txt".format(args.output), "a") as f:
        f.write(os.path.join('results/metric', args.env + " result\n"))
        f.write("label:\tN\tJ\tTP\tFP\tFN\n")
        for key in ['left','front','right']:
            f.write("%s:\t%d\t%.3f\t%d\t%d\t%d\n"%(key, jac[key]['N'], jac[key]['J'], jac[key]['TP'], jac[key]['FP'], jac[key]['FN']))
        f.write('Total accuracy : %.3f\n'%jac['accuracy'])
        f.write('\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ParticleFilter")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='image')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--env', type=str, required=True, choices=['SA1','SA2','SB1','SB2','SB3'])
    args = parser.parse_args()
    main(args)