import numpy as np


def graph_based_segmentation(img):
    len, wid, _  = img.shape
    numPixels = len * wid
    
    W = np.zeros((numPixels, numPixels))
    D = np.zeros((numPixels, numPixels))
    
    
    #all pixels p
    for p in range(numPixels):
        #get the 2D coordinates of p
        i,j = convertTo2D(p,wid)
        
        #all pixels q
        for q in range(numPixels):
            #get the 2D coordinates of q
            k,l = convertTo2D(q,wid)
            
            #skips if i=k and j=l
            if i == k and j == l:
                continue
            
            #if abs(i-k) <= 20 and abs(j-l) <= 20
            if abs(i-k) <= 20 and abs(j-l) <= 20:
                
                #sum of squared differences of color
                diff = (img[i,j,0] - img[k,l,0])**2 + (img[i,j,1] - img[k,l,1])**2 + (img[i,j,2] - img[k,l,2])**2
                
                #e^-100diff
                val = np.exp(-100*diff)
                W[p,q] = val
    
    #loop for D
    for p in range(numPixels):
        #running sum
        sum = 0
        for q in range(numPixels):
            sum += W[p,q]
        D[p,p] = sum
    
    #gets the A matrix
    A = np.eye(numPixels) - np.matmul(np.linalg.inv(D), W)
    
    
    #get the eigenvalues and eigenvectors of A
    eigvals, eigvecs = np.linalg.eig(A)
    
    #gets the vector index of the second smallest eigenvalue
    indexesSorted = np.argsort(np.abs(eigvals))
    index = indexesSorted[1]
    
    #gets the second smallest eigenvector
    vector = eigvecs[:,index]
    
    #reshape the vector into a 2D array
    vector = np.reshape(vector, (len, wid))
    return vector


def convertTo2D(pixel, n):
    i = pixel // n
    j = pixel % n
    return i,j