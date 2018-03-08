
# coding: utf-8

# # implementação do HIO puro
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# In[13]:


# Implementando o HIO puro

iterations = 1000
beta = 1.0

#Assign random phases


imagSize = np.shape(DifPad);
np.random.seed(1) #definning randon seed for randon phase values to be the same betwwen reconstructions
phase_angle = np.random.rand(imagSize[0],imagSize[1]).astype(np.float32);
phase_angle = phase_angle * 2 * np.pi

#Define initial k, r space
initial_k = np.fft.ifftshift(DifPad)
ampKSpace = np.fft.ifftshift(DifPad)
initial_k[ampKSpace == -1] = 0
k_space = initial_k * np.exp(1j*phase_angle);


buffer_r_space = np.real(np.fft.ifftn(k_space)).astype(np.float32);
r_space = scipy.real(ifftn(k_space));


#Preallocate error arrays
RfacF = np.zeros((iterations,1)).astype(np.float32);  
counter1=0; 
errorF=1;


#HIO iterations
iter = 1
while (iter <= iterations):

    #HIO with Support & Positivity constraint
    #r_space = np.real(np.fft.ifftn(k_space));
    
    sample = r_space * Mask;
    
    r_space = buffer_r_space - beta*r_space;
    sample[sample<0] = r_space[sample<0];

    r_space[Mask==1] = sample[Mask==1];
    
    buffer_r_space = r_space;
    
    #k_space = np.fft.fftn(r_space);
    k_space = fftn(r_space)
    
    
    #phase_angle = np.angle(k_space);
    #k_space = ampKSpace*np.exp(1j*phase_angle); #funciona também, mas na forma de baixo é pra ser mais universal
    notthe_angle = np.divide(k_space,np.absolute(k_space))
    notthe_angle[np.absolute(k_space)==0] = 1
    
    k_space[ampKSpace!=-1] = ampKSpace[ampKSpace!=-1] * notthe_angle[ampKSpace!=-1]  
    r_space = scipy.real(ifftn(k_space));
    
    
    #Calculate errors
        
        #Calculate error in reciprocal space
    Ktemp = sample
    #Ktemp = np.absolute(np.fft.fftshift(np.fft.fft2(Ktemp)));
    Ktemp = np.absolute(fftn(Ktemp));
    errorF = np.sum((np.absolute(Ktemp[ampKSpace!=-1]-ampKSpace[ampKSpace!=-1]))) / np.sum(ampKSpace[ampKSpace!=-1]);
    RfacF[counter1] = errorF;
    counter1+=1;

    iter = iter + 1


plt.subplot(2, 2, 1)
imgplot = plt.imshow(Lena)
plt.subplot(2, 2, 2)
imgplot = plt.imshow(np.log10(np.absolute(np.fft.fftshift(k_space))))
plt.subplot(2, 2, 3)
imgplot = plt.imshow(r_space)
plt.subplot(2, 2, 4)
plt.plot(RfacF)
plt.ylabel('Fourier R-factor')
plt.xlabel('Iteration')
plt.title('Final R-factor: %s' %(errorF*100))
plt.tight_layout()



#while True:
#    plt.pause(0.5)
    

