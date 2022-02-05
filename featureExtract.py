#-*- coding:utf-8 -*-
import os
import numpy as np
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
from scipy import signal
import scipy.io.wavfile as wavf

import pandas as pd
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt

import time

N_sep = 8

def loadMicarray():
    ar_x = []
    ar_y = []
    
    # iterrate through the xml to get all locations
    # root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/config/ourmicarray_56.xml').getroot()
    # root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/config/respeaker_4.xml').getroot()
    root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/config/ourmicarray_56.xml').getroot()
    for type_tag in root.findall('pos'):
        ar_x.append(type_tag.get('x'))
        ar_y.append(type_tag.get('y'))

    # set up the array vector
    micArray = np.zeros([len(ar_x)//N_sep, 3])
    micArray[:,1] = ar_x[::N_sep]
    micArray[:,2] = ar_y[::N_sep]

    micArrayConfig = """
  _______________________________________________________________
   Loading microphone Array with {} microphones.  
                                            -O  |
                                -O              |
                    -O                          |
        -O               |Z                     |            ┌ ┐
                         |    _Y            -O  |            |X|
                         |___/  -O              | micArray = |Y|
                    -O    \                     |            |Z|
        -O                 \X                   |            └ ┘
                                            -O  |
                                -O              |
                    -O                          |
        -O                                      | 
  _______________________________________________________________\n\n
        """.format(micArray.shape[0])

    return micArray

def extractSRPFeature(dataIn, sampleRate, micArray, resolution, freqRange=[10,1200], nfft=2*256, L=2):
    # generate fft lengths and filter mics and create doa algorithm
    doaProcessor = pra.doa.algorithms['SRP'](micArray.transpose(), sampleRate, nfft, azimuth=np.linspace(-90.,90., resolution)*np.pi/180, max_four=4)
    # max_four 가 뭘 뜻하는지 알 수가 없다

    # extract the stft from parameters
    container = []
    for i in range(dataIn.shape[1]): # 채널 별로 수행하는 것
        _, _, stft = signal.stft(dataIn[:,i], sampleRate, nperseg=nfft)
        container.append(stft)
    container = np.stack(container)
    # channel, frequency bin, length
    
    # split the stft into L segments
    segments = []
    delta_t = container.shape[-1] // L 
    for i in range(L):
        segments.append(container[:, :, i*delta_t:(i+1)*delta_t])
    # pdb.set_trace()
    # container = [container[:, :, 0:94], container[:, :, 94:94+94]]

    # apply the doa algorithm for each specified segment according to parameters
    feature = []
    for i in range(L):
        doaProcessor.locate_sources(segments[i], freq_range=freqRange)
        feature.append(doaProcessor.grid.values)
        # normalized 된 값이 저장됨

    return np.mean(feature, axis=0)

def main(args):
    if not os.path.isdir("image"):
        os.mkdir("image")
    
    sr, data = wavf.read(args.input)
    data = data[:,::N_sep]
    total_time = data.shape[0] / sr

    extracted_data = None

    fig = plt.figure()
    mic_array = loadMicarray()

    dt = 0.1

    start = time.time()
    count = 0

    for t in tqdm(np.arange(0,total_time-dt,dt).tolist()):
        # data_t = data[int(sr*t):int(sr*(t+dt)),1:-1]
        data_t = data[int(sr*t):int(sr*(t+dt))]
        feature = extractSRPFeature(data_t, sr, mic_array, resolution=180, freqRange=[50,2000], nfft=512, L=1)
        count += 1
        
        x = [(i-len(feature)/2)*180/len(feature) for i in range(len(feature))]
        plt.plot(x, feature)
        plt.xlim([-90,90])
        plt.ylim([0,1])
        plt.title("%.1f sec"%t)
        plt.savefig(os.path.join('image','test_%06d.png'%(int(t/dt))))
        plt.close()

        if extracted_data is None:
            # first set number of columns according to the feature shape
            data_columns = ['feat' + str(x) for x in range(feature.shape[0])]

            #then append the rest of the columns from the label_data
            data_columns.extend(['time'])

            # now create the dataframe and add the feature and label details
            extracted_data = pd.DataFrame(columns=data_columns)
            extracted_data = extracted_data.append(pd.DataFrame([np.concatenate((feature, np.array([t])))], columns=data_columns), ignore_index=True)
        else:
            extracted_data = extracted_data.append(pd.DataFrame([np.concatenate((feature, np.array([t])))], columns=data_columns), ignore_index=True)

    print("TIME : (ms) %f"%((time.time()-start)/count*1000))

    new_file = args.input.replace("data","feature").replace(".wav",".csv")
    extracted_data.to_csv(new_file, index=False)

    output_gif = args.input.replace("data","gif").replace(".wav",".gif")
    os.system("convert -delay {} -loop 0 ./image/*.png {}".format(int(dt*100), output_gif))
    os.system("rm -rf image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)