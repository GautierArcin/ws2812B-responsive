
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from rpi_ws281x import *

np.set_printoptions(suppress=True) # don't use scientific notation

class Masque:
    def __init__(self, rate=44100, channel=1, width=2, chunk=256*2*2, index=2, ledCount = 111, ledPin = 10, ledFreqHZ = 800000, ledDMA = 10, ledBright = 255, ledInvert=False, ledChannel=0):

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
        self.chevauchement = 2
        self.nbPixel = int((self.strip.numPixels() - 2  ) / 2)
        #self.nbPixel = 10
        self.scale =  np.logspace(np.log10(minFreq), np.log10(maxFreq),self.nbPixel+self.chevauchement+1)

        # FreqPeak
        self.freqPeak = 10
        self.freqPeakList = [1]*80

        # Volume
        self.rmsList = [0] * 250
        self.rmsListEyes = [0] * 12
        self.rmsThreeshold = 250
        self.rmsMedian = 0
        self.rmsMedianEyes = 0

    def calculateFFT(self):
        data = np.fromstring(self.stream.read(self.chunk, exception_on_overflow = False),dtype=np.int16)
        data = data * np.hanning(len(data)) # smooth the FFT by windowing data

        fft = abs(np.fft.fft(data).real)
        self.fft = fft[:int(len(fft)/2)] # keep only first half

        freq = np.fft.fftfreq(self.chunk,1.0/self.rate)
        self.freq = freq[:int(len(freq)/2)] # keep only first half

        rms = np.sqrt(np.mean(np.square(data)))
        self.rmsList.append(rms)
        self.rmsList.pop(0)
        self.rmsListEyes.append(rms)
        self.rmsListEyes.pop(0)
        self.rmsMedian= np.median(self.rmsList)
        self.rmsMedianEyes= np.mean(self.rmsListEyes) - self.rmsMedian # Substracting the median of the 120 last to 20 last, in order to see variation
        #print("rms eyes : ", self.rmsMedianEyes)

        freqPk = self.freq[np.where(self.fft[10:]==np.max(self.fft[10:]))[0][0]]+1 # We get the 20 first sample out, in order to not have the basses parasite the fondamental frequency 
        #print("where is max : ", np.where(self.fft[20:]==np.max(self.fft[20:]))[0][0])
        #print("where is max 2 : ", np.where(self.fft==np.max(self.fft))[0][0])
        if(freqPk>1):
            self.freqPeakList.append(freqPk)
            self.freqPeakList.pop(0)   
            self.freqPeakValue = np.log10(np.mean(self.freqPeakList))

    def displayLed(self, dividor = 500000, debug=False):
        for i in range(self.nbPixel):
            idxBottom = (np.abs(self.freq-self.scale[i])).argmin() 
            idxUp= (np.abs(self.freq-self.scale[i+1])).argmin() +self.chevauchement

            mean = np.sum(self.fft[idxBottom:idxUp]) / (idxUp - idxBottom) / dividor

            if(self.rmsMedian > self.rmsThreeshold):
                colorIntensity = int(mean*255)
                if(colorIntensity > 255): colorIntensity=255
            else:
                colorIntensity = 0
                self.strip.setPixelColor(109, Color(0,0,0))
                self.strip.setPixelColor(110, Color(0,0,0))
                
            self.strip.setPixelColor(i, Color( colorIntensity ,0,0))
            self.strip.setPixelColor( self.nbPixel+(self.nbPixel-i), Color( 0,0,colorIntensity ))

            if(debug):
                print("\n\n\n")
                print("setting %d and %d" % (i,  self.nbPixel+(self.nbPixel-i)))
                print("self scale %f", self.scale)
                print("indx up %f, indx down %f" % (idxUp, idxBottom))
                print("i :", i, ", mean: ", mean, ", color intensity : ", colorIntensity)
                
        if(self.rmsMedian > self.rmsThreeshold):
            self.strip.setPixelColor(109, self.eyes(self.freqPeakValue,self.rmsMedianEyes))
            self.strip.setPixelColor(110, self.eyes(self.freqPeakValue,self.rmsMedianEyes))
        else:
            self.strip.setPixelColor(109, Color(0,0,0))
            self.strip.setPixelColor(110, Color(0,0,0))

        self.strip.show()

    
    def terminate(self):
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, Color(0,0,0))
        self.strip.show()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    
    def eyes(self, pitch, energy):
        """Generate rainbow colors for eyes"""
        """Define intensity through fft variation"""
        """Varies color through pitch"""

        print("pitch %f, energy %f" % (pitch, energy))

        # Pitch varies between 2 and 4
        # We wants its to varies between 0 and 255
        pitch -= 1.5
        color = pitch / 2.0 * 255
        if(color < 0): color = 0
        if(color > 255): color = 255

        # Energy varies between -1000 and 1000
        # We wants it to varies between 0 and 1
        energy += 1000
        energy -= 500
        variation = energy / 2000.0 
        if(variation > 1): variation = 1
        if(variation < 0.05): variation = 0.05

        #print("color %f, variation %f" % (color, variation))
        #print("color eyes : ", Color(int((255-color)*variation), int(color*variation), 0))
        
        return Color(int((255-color)*variation), int(color*variation), 0)

if __name__ == "__main__":
    instance = Masque()
    try: 
        while True:
            start = time.time()
            instance.calculateFFT()
            instance.displayLed()
            end = time.time()
            print("time elpased : ", end - start)
            #print("new line")
    except KeyboardInterrupt:
        instance.terminate()
