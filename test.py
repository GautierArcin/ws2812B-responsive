import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from rpi_ws281x import *

np.set_printoptions(suppress=True) # don't use scientific notation

CHUNK = 4096 # number of data points to read at a time
RATE = 44100*2 # time resolution of the recording device (Hz)

p=pyaudio.PyAudio() # start the PyAudio class
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

LED_COUNT      = 30      # Number of LED pixels.
#LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 55     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Create NeoPixel object with appropriate configuration.
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
strip.begin()

minFreq = 20
maxFreq = 4500
scale =  np.logspace(np.log10(20), np.log10(20000.0),strip.numPixels()+1)
# create a numpy array holding a single read of audio data
while True: #to it a few times just to see
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    data = data * np.hanning(len(data)) # smooth the FFT by windowing data

    fft = abs(np.fft.fft(data).real)
    fft = fft[:int(len(fft)/2)] # keep only first half

    freq = np.fft.fftfreq(CHUNK,1.0/RATE)
    freq = freq[:int(len(freq)/2)] # keep only first half

    #freqPeak = freq[np.where(fft==np.max(fft))[0][0]]+1
    #freqPeak2 = freq[np.where(fft==np.max(fft))][0]

    #print(str(freqPeak2))
    #print("peak frequency: %d Hz"%freqPeak)
    # print("")
    # print("")
    print("")
    print("")
    # print(fft.shape)
    
    for i in range(strip.numPixels()):
        # print(i," : ",scale[i], " - ",scale[i+1])
        idxBottom = (np.abs(freq-scale[i])).argmin() 
        idxUp= (np.abs(freq-scale[i+1])).argmin() 

        print(idxBottom, " -> ", idxUp)
        mean = np.sum(fft[idxBottom:idxUp]) / (idxUp - idxBottom) / 200000
        print(mean)
        # print(np.where(freq==bottomFreq)[0])
        # upFreq = (find_nearest(freq, scale[i+1]))
        # mean = np.mean(fft[):int(scale[i+1])])
        if(np.isnan(mean)): mean=0.1
        # mean = mean / 1000.0
        strip.setPixelColor(i,Color(int(mean/50.0*255),0,int(mean/50.0*255)))
    strip.show()

    # time.sleep(0.20)



    # uncomment this if you want to see what the freq vs FFT looks like
    # plt.plot(freq,fft)
    # plt.axis([0,4000,None,None])
    # plt.show()
    # plt.close()

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()