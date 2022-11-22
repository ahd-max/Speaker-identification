# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:58:16 2022

@author: Hangdong AN
"""

from pydub import AudioSegment
import scipy.io.wavfile as wav
import python_speech_features
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
import numpy as np

main_wav_path='D:/master/中科院/2-3/OriginalData/7'
path_list = os.listdir(main_wav_path)
path_list.sort()
part_wav_path='D:/master/中科院/2-3/OriginalData/6'
def get_ms_part_wav(main_wav_path, start_time,cut_time,part_wav_path,number):
    audio = AudioSegment.from_file(main_wav_path)
    audio_time = len(audio)
    start_time = int(start_time)
    cut_time=int(cut_time)
    cut_parameters = np.arange(cut_time,audio_time/1000,cut_time)
    i=number
    for t in cut_parameters:
        stop_time = int(t*1000)
        audio_chunk = audio[start_time:stop_time]
        i=i+1
        audio_chunk.export(part_wav_path+'/'+'6'+'_'+str(i)+'.wav',format="wav")
        start_time = int(stop_time)
        print('finish')
number = 0
for j in range(len(path_list)):
    file=path_list[j]
    audio = AudioSegment.from_file(main_wav_path+"/"+str(file))
    get_ms_part_wav(main_wav_path+"/"+str(file),0,2,part_wav_path,number)
    number=number+int(len(audio)/2000)



part_wav_path='D:/master/中科院/2-3/OriginalData/8'   
def get_ms_part_wav1(main_wav_path, start_time,cut_time,part_wav_path,number):
    audio = AudioSegment.from_file(main_wav_path)
    audio_time = len(audio)
    start_time = int(start_time)
    cut_time=int(cut_time)
    cut_parameters = np.arange(cut_time,audio_time/1000,cut_time)
    i=number
    for t in cut_parameters:
        stop_time = int(t*1000)
        audio_chunk = audio[start_time:stop_time]
        i=i+1
        audio_chunk.export(part_wav_path+'/'+'8'+'_'+str(i)+'.wav',format="wav")
        start_time = int(stop_time)
        print('finish')
number = 0
for j in range(len(path_list)):
    file=path_list[j]
    audio = AudioSegment.from_file(main_wav_path+"/"+str(file))
    get_ms_part_wav1(main_wav_path+"/"+str(file),0,4,part_wav_path,number)
    number=number+int(len(audio)/4000)

part_wav_path='D:/master/中科院/2-3/OriginalData/9'   



def get_ms_part_wav1(main_wav_path, start_time,cut_time,part_wav_path,number):
    audio = AudioSegment.from_file(main_wav_path)
    audio_time = len(audio)
    start_time = int(start_time)
    cut_time=int(cut_time)
    cut_parameters = np.arange(cut_time,audio_time/1000,cut_time)
    i=number
    for t in cut_parameters:
        stop_time = int(t*1000)
        audio_chunk = audio[start_time:stop_time]
        i=i+1
        audio_chunk.export(part_wav_path+'/'+'8'+'_'+str(i)+'.wav',format="wav")
        start_time = int(stop_time)
        print('finish')
number = 0
for j in range(len(path_list)):
    file=path_list[j]
    audio = AudioSegment.from_file(main_wav_path+"/"+str(file))
    get_ms_part_wav1(main_wav_path+"/"+str(file),0,6,part_wav_path,number)
    number=number+int(len(audio)/6000)







