
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from rpi_ws281x import *

np.set_printoptions(suppress=True) # don't use scientific notation

class Masque:
    def __init__(self, rate=44100, channel=1, width=2, chunk=2048, index=0, ledCount = 111, ledPin = 10, ledFreqHZ = 800000, ledDMA = 10, ledBright = 255, ledInvert=False, ledChannel=0):

        # Stream
        self.rate       = rate        
        self.channel    = channel
        self.index      = index
        self.chunk      = chunk

        self.channel    = channel
        self.width      = width
        
        self.p=pyaudio.PyAudio() # start the PyAudio class
        self.stream = self.p.open(
            rate=self.rate,
            format=self.p.get_format_from_width(self.width),
            channels=self.channel,
            frames_per_buffer=self.chunk,
            input=True,
            input_device_index=self.index)


        # Leds
        self.ledCount       = ledCount          # Number of LED pixels.
        self.ledPin         = ledPin            # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
        self.ledFreqHZ      = ledFreqHZ         # LED signal frequency in hertz (usually 800khz)
        self.ledDMA         = ledDMA            # DMA channel to use for generating signal (try 10)
        self.ledBright      = ledBright         # Set to 0 for darkest and 255 for brightest
        self.ledInvert      = ledInvert         # True to invert the signal (when using NPN transistor level shift)
        self.ledChannel     = ledChannel        # set to '1' for GPIOs 13, 19, 41, 45 or 53

        # Create NeoPixel object with appropriate configuration.
        self.strip = Adafruit_NeoPixel(self.ledCount, self.ledPin, self.ledFreqHZ, self.ledDMA, self.ledInvert, self.ledBright, self.ledChannel)
        
        # Intialize the library (must be called once before other functions).
        self.strip.begin()
        
        #Gamma correction approximation
        self.gamma8 = [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255 ]
    
        #Scale
        minFreq = 20
        maxFreq = 20000
        self.chevauchement = 10
        self.nbPixel = int((self.strip.numPixels() - 2 ) / 2)
        #self.nbPixel = 10
        self.scale =  np.logspace(np.log10(minFreq), np.log10(maxFreq),self.nbPixel+self.chevauchement+1)

    def calculateFFT(self):
        data = np.fromstring(self.stream.read(self.chunk),dtype=np.int16)
        data = data * np.hanning(len(data)) # smooth the FFT by windowing data

        fft = abs(np.fft.fft(data).real)
        self.fft = fft[:int(len(fft)/2)] # keep only first half

        freq = np.fft.fftfreq(self.chunk,1.0/self.rate)
        self.freq = freq[:int(len(freq)/2)] # keep only first half

        self.freqPeak = self.freq[np.where(self.fft==np.max(self.fft))[0][0]]+1
        print("peak frequency: %d Hz"%self.freqPeak)
        #print(np.sum(self.fft)/22311964.0)

    def displayLed(self, dividor = 300000):
        for i in range(self.nbPixel):
            #print("Normal : ",i, ", Reverse : ", self.nbPixel+(self.nbPixel-i))
            idxBottom = (np.abs(self.freq-self.scale[i])).argmin() 
            idxUp= (np.abs(self.freq-self.scale[i+self.chevauchement+1])).argmin() 

            mean = np.sum(self.fft[idxBottom:idxUp]) / (idxUp - idxBottom) / dividor
            #print(mean)
            if(np.isnan(mean)): mean=0.001
            colorIntensity = int(mean*255)
            if(colorIntensity > 255): colorIntensity=255
            #print(colorIntensity)

            self.strip.setPixelColor(i, Color( colorIntensity ,0,0))
            self.strip.setPixelColor( self.nbPixel+(self.nbPixel-i), Color( 0,0,colorIntensity ))

        #for i in range(self.strip.numPixels()):
            # print(i," : ",scale[i], " - ",scale[i+1])
            # idxBottom = (np.abs(freq-scale[i])).argmin() 
            # idxUp= (np.abs(freq-scale[i+chevauchement+1])).argmin() 
            # print(idxBottom, " -> ", idxUp)
            # mean = np.sum(fft[idxBottom:idxUp]) / (idxUp - idxBottom) / 20000
            # print(mean)
            # print(np.where(freq==bottomFreq)[0])
            # upFreq = (find_nearest(freq, scale[i+1]))
            # mean = np.mean(fft[):int(scale[i+1])])
            # if(np.isnan(mean)): mean=0.001
            # colorIntensity = mean/5.0*255
            # if(colorIntensity > 250): colorIntensity=255
            # color1 = gamma8[int(colorIntensity * (i/strip.numPixels()))]
            # color2 = gamma8[int(colorIntensity * ((strip.numPixels() - i) /strip.numPixels()))]
            # mean = mean / 1000.0
            # strip.setPixelColor(i,(Color(color1 ,0, color2)))
            #self.strip.setPixelColor(i, Color( int(255 * i/ self.strip.numPixels()) ,0,0))
            
            #print("i : ",i," color calculated : ", int(255 * i/ self.strip.numPixels()) )

        colorIntensity = int(self.freqPeak / 2000 * 255)
        if(colorIntensity > 255): colorIntensity=255
        print(colorIntensity)
        self.strip.setPixelColor(109, self.wheel(colorIntensity))
        self.strip.setPixelColor(110, self.wheel(colorIntensity))

        self.strip.show()

    
    def terminate(self):
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, Color(0,0,0))
        self.strip.show()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    # not Mine, from strandtest
    # 0 +> 255 To get a position
    def wheel(self,pos):
        """Generate rainbow colors across 0-255 positions."""
        if pos < 85:
            return Color(pos * 3, 255 - pos * 3, 0)
        elif pos < 170:
            pos -= 85
            return Color(255 - pos * 3, 0, pos * 3)
        else:
            pos -= 170
            return Color(0, pos * 3, 255 - pos * 3)


if __name__ == "__main__":
    instance = Masque()
    try: 
        while True:
            instance.calculateFFT()
            instance.displayLed()
            #print("new line")
    except KeyboardInterrupt:
        instance.terminate()