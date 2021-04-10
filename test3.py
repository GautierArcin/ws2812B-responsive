
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from rpi_ws281x import *

np.set_printoptions(suppress=True) # don't use scientific notation

p=pyaudio.PyAudio() # start the PyAudio class
 
RESPEAKER_RATE = 44100
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 0  # refer to input device id
CHUNK = 2048

RATE = RESPEAKER_RATE
stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=1,
            input=True,
            input_device_index=RESPEAKER_INDEX)



LED_COUNT      = 20      # Number of LED pixels.
LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53


# Create NeoPixel object with appropriate configuration.
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# Intialize the library (must be called once before other functions).
strip.begin()

gamma8 = [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255 ]



minFreq = 20
maxFreq = 20000
chevauchement = 0
scale =  np.logspace(np.log10(20), np.log10(20000.0),strip.numPixels()+chevauchement+1)

# base = 1.000001
#scale =  np.logspace( np.log(minFreq) / np.log(base),  np.log(maxFreq) / np.log(base),strip.numPixels()+1, True, base)
# create a numpy array holding a single read of audio data
while True: #to it a few times just to see
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    data = data * np.hanning(len(data)) # smooth the FFT by windowing data

    fft = abs(np.fft.fft(data).real)
    fft = fft[:int(len(fft)/2)] # keep only first half

    freq = np.fft.fftfreq(CHUNK,1.0/RATE)
    freq = freq[:int(len(freq)/2)] # keep only first half

    for i in range(strip.numPixels()):
        print(i," : ",scale[i], " - ",scale[i+1])
        idxBottom = (np.abs(freq-scale[i])).argmin() 
        idxUp= (np.abs(freq-scale[i+chevauchement+1])).argmin() 

        print(idxBottom, " -> ", idxUp)
        mean = np.sum(fft[idxBottom:idxUp]) / (idxUp - idxBottom) / 20000
        
        print(mean)
        # print(np.where(freq==bottomFreq)[0])
        # upFreq = (find_nearest(freq, scale[i+1]))
        # mean = np.mean(fft[):int(scale[i+1])])
        if(np.isnan(mean)): mean=0.001
        colorIntensity = mean/5.0*255
        if(colorIntensity > 250): colorIntensity=255
        color1 = gamma8[int(colorIntensity * (i/strip.numPixels()))]
        color2 = gamma8[int(colorIntensity * ((strip.numPixels() - i) /strip.numPixels()))]
        # mean = mean / 1000.0
        strip.setPixelColor(i,(Color(color1 ,0, color2)))
    strip.show()
    # exit()

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
