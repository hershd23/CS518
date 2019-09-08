#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from skimage import img_as_ubyte
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


plt.close('all')
clear = lambda: os.system('clear')
clear()
colors = [[1,0,0],[0,1,0],[0,0,1],[0,0.5,0.5],[0.5,0,0.5]]


# In[15]:


np.random.seed(110) #for reproducability of results
imgNames = ['water_coins','jump','tiger']
#{'balloons', 'mountains', 'nature','ocean', 'polarlights'};
segmentCounts = [2,3,4,5]
req_img = np.ones((12), dtype = int)


# In[16]:


# In[20]:


ctr_img = 0
for imgName in imgNames:
    for SegCount in segmentCounts:
        # Load the imageusing OpenCV        
        img = mpimg.imread("Input/"+ imgName + ".png");""" Read Image using mplib library-- 2 points """ 
        print('Using Matplotlib Image Library: Image is of datatype ',img.dtype,'and size ',img.shape) # Image is of type float 

        # Load the Pillow-- the Python Imaging Library
        img = Image.open("Input/"+ imgName + ".png"); """ Read Image using PILLOW-- 3 points"""
        #print('Using Pillow (Python Image Library): Image is of datatype ',img.dtype,'and size ',img.size) # Image is of type uint8  
        print('Using Pillow (Python Image Library): Image is of datatype', np.array(img).dtype,'and size ',img.size) # Image is of type uint8  
        
        #%% %Define Parameters
        nSegments = SegCount   # of color clusters in image
        nPixels = img.size[0]*img.size[1];""" Compute number of image pixels from image dimensions-- 2 points""";    # Image can be represented by a matrix of size nPixels*nColors
        maxIterations = 20; #maximum number of iterations allowed for EM algorithm.
        nColors = 3;
        #%% Determine the output path for writing images to files
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/']));
        if not(os.path.exists(outputPath)):
            os.makedirs(outputPath)
            
        mpimg.imsave(outputPath + "0.png", np.array(img))
        """ save input image as *0.png* under outputPath-- 3 points""" #save using Matplotlib image library
        #%% Vectorizing image for easier loops- done as im(:) in Matlab
        pixels = img
        pixels = np.reshape(np.array(img), (nPixels, nColors, 1));""" Reshape pixels as a nPixels X nColors X 1 matrix-- 5 points"""
        
        #%%
        """ Initialize pi (mixture proportion) vector and mu matrix (containing means of each distribution)
            Vector of probabilities for segments... 1 value for each segment.
            Best to think of it like this...
            When the image was generated, color was determined for each pixel by selecting
            a value from one of "n" normal distributions. Each value in this vector 
            corresponds to the probability that a given normal distribution was chosen."""
        
        
        """ Initial guess for pi's is 1/nSegments. Small amount of noise added to slightly perturb 
           GMM coefficients from the initial guess"""
           
        pi = 1/nSegments*(np.ones((nSegments, 1),dtype='float'))
        increment = np.random.normal(0,.0001,1)
        for seg_ctr in range(len(pi)):
            if(seg_ctr%2==1):
                pi[seg_ctr] = pi[seg_ctr] + increment
            else:
                pi[seg_ctr] = pi[seg_ctr] - increment
                if pi[seg_ctr] < 0:
                    pi[seg_ctr] = 0
                

         #%% 


        """Similarly, the initial guess for the segment color means would be a perturbed version of [mu_R, mu_G, mu_B],
           where mu_R, mu_G, mu_B respectively denote the means of the R,G,B color channels in the image.
           mu is a nSegments X nColors matrrix,(seglabels*255).np.asarray(int) where each matrix row denotes mean RGB color for a particcular segment"""
           
        mu = 1/nSegments*(np.ones((nSegments, nColors),dtype='float'));"""Initialize mu to 1/nSegments*['ones' matrix (whose elements are all 1) of size nSegments X nColors] -- 5 points"""  #for even start
        #add noise to the initialization (but keep it unit)
        for seg_ctr in range(nSegments):
            if(seg_ctr%2==0):
                increment = np.random.normal(0, 0.0001,1)
            for col_ctr in range(nColors):
                if(seg_ctr%2==1):
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) + increment
                else:
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) - increment;              
        

        #%% EM-iterations begin here. Start with the initial (pi, mu) guesses        
        
        mu_last_iter = mu;
        pi_last_iter = pi;
        
        for iteration in range(maxIterations):
            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % -----------------   E-step  -----estimating likelihoods and membership weights (Ws)
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration: ',str(iteration+1), ' E-step']))
            # Weights that describe the likelihood that pixel denoted by "pix_import scipy.miscctr" belongs to a color cluster "seg_ctr"
            Ws = np.ones((nPixels,nSegments),dtype='float')  # temporarily reinitialize all weights to 1, before they are recomputed

            """ logarithmic form of the E step."""
            
            for pix_ctr in range(nPixels):
                # Calculate Ajs
                logAjVec = np.zeros((nSegments,1),dtype='float')
                for seg_ctr in range(nSegments):
                    x_minus_mu_T  = np.transpose(pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T)
                    x_minus_mu    = ((pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T))
                    logAjVec[seg_ctr] = np.log(pi[seg_ctr]) - .5*(np.dot(x_minus_mu_T,x_minus_mu))
                
                # Note the max
                logAmax = max(logAjVec.tolist()) 
                
                # Calculate the third term from the final eqn in the above link
                thirdTerm = 0;
                for seg_ctr in range(nSegments):
                    thirdTerm = thirdTerm + np.exp(logAjVec[seg_ctr]-logAmax)
                
                # Here Ws are the relative membership weights(p_i/sum(p_i)), but computed in a round-about way 
                for seg_ctr in range(nSegments):
                    logY = logAjVec[seg_ctr] - logAmax - np.log(thirdTerm)
                    Ws[pix_ctr][seg_ctr] = np.exp(logY)
                

            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % -----------------   M-step  --------------------
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
            
            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration: ',str(iteration+1), ' M-step: Mixture coefficients']))
            #%% temporarily reinitialize mu and pi to 0, before they are recomputed
            mu = np.zeros((nSegments,nColors),dtype='float') # mean color for each segment
            pi = np.zeros((nSegments,1),dtype='float') #mixture coefficients
            mu1 = np.zeros((nSegments,nColors),dtype='float')
            pi1 = np.zeros((nSegments,1),dtype='float')
            
            for seg_ctr in range(nSegments):

                denominatorSum = 0;
                '''
                for pix_ctr in range(nPixels):
                    p = np.reshape(np.copy(pixels[pix_ctr,:]), (3))
                    mu[seg_ctr, :] = mu[seg_ctr, :] + p*Ws[pix_ctr][seg_ctr];"""Update RGB color vector of mu[seg_ctr] as current mu[seg_ctr] + pixels[pix_ctr,:] times Ws[pix_ctr,seg_ctr] -- 5 points"""
                    denominatorSum = denominatorSum + Ws[pix_ctr][seg_ctr]'''
                
                """Compute mu[seg_ctr] and denominatorSum directly without the 'for loop'-- 10 points.
                   If you find the replacement instruction, comment out the for loop with your solution"
                   Hint: Use functions squeeze, tile and reshape along with sum"""
                
                #soln for the upper part, without the loop
                denominatorSum = np.sum(Ws[:,seg_ctr])
                npix = np.transpose(np.reshape(np.copy(pixels), (-1, 3)))
                nWs = np.transpose((np.transpose(np.copy(Ws)))[seg_ctr])
                nWs = np.reshape(nWs, (-1, 1))
                add_mat = np.reshape(np.matmul(npix, nWs), (1,3))
                
                ## Update mu
                mu[seg_ctr,:] = mu[seg_ctr, :] + add_mat
                mu[seg_ctr,:] /= denominatorSum
                ## Update pi
                pi[seg_ctr] = denominatorSum / nPixels; #sum of weights (each weight is a probability) for given segment/total num of pixels   
        

            print(np.transpose(pi))

            muDiffSq = np.sum(np.multiply((mu - mu_last_iter),(mu - mu_last_iter)))
            piDiffSq = np.sum(np.multiply((pi - pi_last_iter),(pi - pi_last_iter)))
            
            if (muDiffSq < .0000001 and piDiffSq < .0000001 and iteration!=0): #sign of convergence
                req_img[ctr_img] = iteration
                ctr_img += 1
                print('Convergence Criteria Met at Iteration: ',iteration, '-- Exiting code')
                break;
            
            if(iteration == 19):
                req_img[ctr_img] = 20
                ctr_img +=1

            mu_last_iter = mu;
            pi_last_iter = pi;


            ##Draw the segmented image using the mean of the color cluster as the 
            ## RGB value for all pixels in that cluster.
            segpixels = np.array(pixels)
            cluster = 0
            for pix_ctr in range(nPixels):
                cluster = np.where(Ws[pix_ctr,:] == max(Ws[pix_ctr,:]))
                vec     = np.squeeze(np.transpose(mu[cluster,:])) 
                segpixels[pix_ctr,:] =  vec.reshape(vec.shape[0],1)
            
            """ Save segmented image at each iteration. For displaying consistent image clusters, it would be useful to blur/smoothen the segpixels image using a Gaussian filter.  
                Prior to smoothing, convert segpixels to a Grayscale image, and convert the grayscale image into clusters based on pixel intensities"""
            
            segpixels = np.reshape(segpixels,(img.size[1],img.size[0],nColors)) ## reshape segpixels to obtain R,G, B image
            """convert segpixels to uint8 gray scale image and convert to grayscale-- 5 points""" #convert to grayscale
            segpixels = img_as_ubyte(segpixels) #Custom function built to convert the image to uint8 type
            segpixels = rgb2gray(segpixels)
            kmeans = KMeans(n_clusters = SegCount).fit(np.reshape(segpixels,(img.size[0]*img.size[1], 1)));""" Use kmeans from sci-kit learn library to cluster pixels in gray scale segpixels image to *nSegments* clusters-- 10 points"""
            seglabels = np.reshape(kmeans.labels_, (img.size[1], img.size[0]));""" reshape kmeans.labels_ output by kmeans to have the same size as segpixels -- 5 points"""
            seglabels = gaussian(np.clip(label2rgb(seglabels, colors = colors), 0,1), sigma = 2, multichannel = False);"Use np.clip, Gaussian smoothing with sigma =2 and label2rgb functions to smoothen the seglabels image, and output a float RGB image with pixel values between [0--1]-- 20 points"""
            mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabels) #save the segmented output

            """ Display the 20th iteration (or final output in case of convergence) segmentation images with nSegments = 2,3,4,5 for the three images-- this will be a 3 row X 4 column image matrix-- 15 points"""
            
            """ Comment on the results obtained, and discuss your understanding of the Image Segmentation problem in general-- 10 points """


# In[18]:

# In[19]:


i=0
fig = plt.figure(figsize = (40,40))
for imgName in imgNames:
    for SegCount in segmentCounts:
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/']))
        img = mpimg.imread(outputPath + str(req_img[i]) + ".png")
        i+=1
        a = fig.add_subplot(3,4,i)
        a.set_title(imgName + " " + str(SegCount))
        plt.imshow(img)

plt.show()

# In[ ]:




