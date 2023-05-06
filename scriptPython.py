
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from rpi_ws281x import *

np.set_printoptions(suppress=True) # don't use scientific notation

class Masque:
    def __init__(self, rate=44100, channel=1, width=2, chunk=2048, index=2, ledCount = 111, ledPin = 10, ledFreqHZ = 800000, ledDMA = 10, ledBright = 255, ledInvert=False, ledChannel=0):

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
        self.freqPeakList = [1]*10

        # Volume
        self.rmsList = [0] * 120
        self.rmsListEyes = [0] * 10
        self.rmsThreeshold = 300
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
        print("rms eyes : ", self.rmsMedianEyes)
        

        freqPk = self.freq[np.where(self.fft[2:]==np.max(self.fft[2:]))[0][0]]+1 
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
                #print("self color intensity : ", colorIntensityPeak, ", frequ : ", np.log10(self.freqPeakValue ))
                self.strip.setPixelColor(109, self.eyes4(self.rmsMedianEyes))
                self.strip.setPixelColor(110, self.eyes4(self.rmsMedianEyes))
            else:
                colorIntensity = 0
                colorIntensityPeak = 0
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
                print("color intensity Peak: ", colorIntensityPeak)

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
            
    def wheel2(self,pos):
        """Generate rainbow colors across 0-255 positions."""
        if pos < 255:
            return Color(pos, 255 - pos , 0)
        elif pos < 510:
            pos -= 255
            return Color(255 - pos , 0, pos )
        else:
            pos -= 510
            return Color(0, pos, 255 - pos)

    def eyes(self,color):
        """Generate rainbow colors across 0-255 positions."""
        return Color(255-color , 0, color)
    
    def eyes2(self,color):
        """Generate rainbow colors for eyes, with value between 1 and 3"""
        #print("color : ", color)
        color -= 1 
        color -= 0.5 # Correction because we wants more green
        if(color < 0): color = 0
        if(color > 2): color = 2
        color = int(color*255)
        if(color > 255):
            color-= 255
            return Color(0, 255-color, color)
        else:
            return Color(255-color, color, 0)

    def eyes3(self,color):
        """Generate rainbow colors for eyes, with value between 1000 and -1000"""
        #print("color : ", color)
        color += 1000
        if(color < 0): color = 0
        if(color > 2000): color = 2000
        color = int(color*255*2/2000)
        if(color > 255):
            color-= 255
            return Color(0, 255-color, color)
        else:
            return Color(255-color, color, 0)


    def eyes4(self,color):
        """Generate rainbow colors for eyes, with value between 1000 and -1000"""
        print("color : ", color)
        color += 1000
        if(color < 0): color = 0
        if(color > 2000): color = 2000
        color = int(color*255/2000)
        return Color(255-color, color, 0)

if __name__ == "__main__":
    instance = Masque()
    try: 
        while True:
            instance.calculateFFT()
            instance.displayLed()
            #print("new line")
    except KeyboardInterrupt:
        instance.terminate()