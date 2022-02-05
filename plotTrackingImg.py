import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# plt.style.use(['science','ieee'])
title={
    0 : "SA2_left",
    1 : "SA2_right",
    2 : "SB1_left",
    3 : "SB1_right",
    4 : "SB2_left",
    5 : "SB2_right"
}

data = pd.read_csv('data/trackingData.csv', header=None).to_numpy()
for i in range(6):
    pred = data[3*i]
    gt = data[3*i+1]
    time = data[3*i+2]
    plt.figure()
    plt.plot(time, pred, 'r', label='particle filter prediction')
    plt.plot(time, gt, 'k', linestyle='dotted', label='ground truth regression')
    plt.plot(time[20:], gt[20:], 'k', label='ground truth')

    doa = pd.read_csv('data/%s.csv'%title[i], header=None).to_numpy()[:,:180]
    doa = (np.argmax(doa, axis=1)-90) * math.pi / 180
    doa_f = 12 * np.tan(doa)
    plt.plot(time[:-1], doa_f, 'b', label='Max DoA prediction')

    plt.legend()
    # plt.xticks([-2,-1,0,1,2])
    plt.xlabel('t [sec]')
    plt.ylabel('x position [m]')
    plt.title(title[i])
    plt.savefig('images/%s.png'%title[i])
