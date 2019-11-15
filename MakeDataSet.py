import librosa
import scipy.io.wavfile as wavfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys
import os


def elev_extract(path,filename,full_path):
    df = pd.read_csv(full_path, skiprows=[i for i in range(0, 3000009)], names=['time', 'data', 'nan'])
    del df['nan']
    df['time'] = df['time'].astype(float)

    times = df['time'].values
    n_measurements = len(times)
    timespan_seconds = times[-1] - times[0]

    sample_rate_hz = int(n_measurements / timespan_seconds)

    data = df['data'].values
    sf.write('{0}/{1}.wav'.format(path,filename), data, sample_rate_hz)

def extract(path,filename,full_path):
    df = pd.read_csv(full_path, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=['time', 'data', 'nan'])
    del df['nan']
    df['time'] = df['time'].astype(float)

    times = df['time'].values
    n_measurements = len(times)
    timespan_seconds = times[-1] - times[0]

    sample_rate_hz = int(n_measurements / timespan_seconds)

    data = df['data'].values
    sf.write('{0}/{1}.wav'.format(path,filename), data, sample_rate_hz)


def make_image(save_path,input_wav, orgin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=orgin_sr)
    data = librosa.resample(y, sr, resample_sr)

    #csv read, write
    # data_time=data.shape[0]/float(sr) #총 시간
    # sampling_interval=1.0/float(sr) #데이터의 시간
    # time=[i for i in np.arange(0,data_time,sampling_interval)]
    # time_data=list(zip(time,data))
    # print(data[0:10])

    count=0
    count_arr=[]

    for i in range(10000,len(data),10000):
        count_arr.append(data[count:i])
        count=i

    for i,j in enumerate(count_arr):
        plt.specgram(j, Fs=sr)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.savefig("{0}/{1}.png".format(save_path, i+1), bbox_inces='tight', pad_inches=0)



try:
    l_path=sys.argv[1] #"./raw_data"
    s_path=sys.argv[2] #"./dataset/train"
except:
    print("python JMakeDataSet.py ./raw_data ./dataset/train/")

path_dir=l_path
file_list=os.listdir(path_dir) # [0,1,2,..,18]
file_list.sort()

for i in file_list:
    csv_filename=os.listdir(path_dir+"/"+i)  # 10min.csv
    r_path=path_dir+"/"+i #./raw_data/0
    save_path=s_path+"/"+i #"./dataset/train"
    full_path=path_dir+"/"+i+"/"+csv_filename[0] #./data/0/10min.csv
    if csv_filename[0][0:2] == "11":
        elev_extract(r_path,csv_filename[0][0:5],full_path)
    else:
        extract(r_path,csv_filename[0][0:5],full_path)

    wav_filename = "{0}/{1}.wav".format(r_path,csv_filename[0][0:5])
    make_image(save_path,wav_filename,50000,20000) #저장경로, 파일명, 리샘플링할 샘플레이트
