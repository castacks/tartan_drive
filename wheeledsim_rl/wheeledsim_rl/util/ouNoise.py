import numpy as np
import matplotlib.pyplot as plt

#If you do sin(ouNoise), dont saturate the input to the sine (Also try the region-based thing.)

class sinOuNoise(object):
    """
    OU noise, but clamp by passing through a sine function.
    """
    def __init__(self,offset=np.array([0.2,0]),\
        var=np.array([0.025,0.05]),\
        ):
        self.offset = np.array(offset)
        self.var = np.array(var)
        self.noise = np.copy(self.offset)
        self.noiseFilt = np.copy(self.noise)

    def genNoise(self):
        dNoise = np.random.normal(0,self.var)
        self.noise = self.noise + dNoise
#        self.noise = np.sin(self.noise * np.pi)
        return np.copy(np.sin(np.pi * self.noise))

    def multiGenNoise(self,numGen,returnAll = False):
        allNoises = []
        for i in range(numGen):
            allNoises.append(self.genNoise())
        if returnAll:
            return allNoises
        else:
            return allNoises[-1]

    def flipOffset(self):
        self.offset = -self.offset

class ouNoise(object):
    def __init__(self,offset=np.array([0.2,0]),\
        damp = np.array([0.0125,0.001]),\
        var=np.array([0.025,0.05]),\
        filtAlpha=np.array([0.8,0.99]),\
        lowerlimit=-np.ones(2),\
        upperlimit=np.ones(2)):
        self.offset = np.array(offset)
        self.damp = np.array(damp)
        self.var = np.array(var)
        self.filtAlpha = filtAlpha
        self.lowerlimit = lowerlimit
        self.upperlimit = upperlimit
        self.noise = np.copy(self.offset)
        self.noiseFilt = np.copy(self.noise)
    def genNoise(self):
        dNoise = self.damp*(self.offset-self.noise) + np.random.normal(0,self.var)
        self.noise = self.noise + dNoise
        self.noiseFilt = self.noiseFilt*self.filtAlpha+(1-self.filtAlpha)*self.noise
        self.noiseFilt = np.maximum(self.lowerlimit,np.minimum(self.upperlimit,self.noiseFilt))
        return np.copy(self.noiseFilt)
    def multiGenNoise(self,numGen,returnAll = False):
        allNoises = []
        for i in range(numGen):
            allNoises.append(self.genNoise())
        if returnAll:
            return allNoises
        else:
            return allNoises[-1]
    def flipOffset(self):
        self.offset = -self.offset

if __name__ == '__main__':
    noise = ouNoise(var = np.array([0.01, 0.01]), offset=np.zeros([2, ]), damp=np.ones([2, ])*1e-4)
    sin_noise = sinOuNoise(var = np.array([0.01, 0.01]))
    
    noise1 = []
    noise2 = []
    noise3 = []
    noise4 = []

    for i in range(10000):
        generatedNoise = noise.multiGenNoise(20)
        generatedNoise2 = sin_noise.multiGenNoise(20)
        noise1.append(generatedNoise[0])
        noise2.append(generatedNoise[1])
        noise3.append(generatedNoise2[0])
        noise4.append(generatedNoise2[1])

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(noise1, label='OU')
    axs[0].plot(noise2, label='OU')
    axs[1].plot(noise3, label='sine')
    axs[1].plot(noise4, label='sine')
    plt.show()
