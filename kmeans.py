import numpy as np

def kmeans(feats, k, max_iter=1000):
    #init
    center_idx = np.random.choice(feats.shape[0], k)
    centers = feats[center_idx,:]
    yprev = None
    for i in range(max_iter):
        #compute distances
        squared_distance = np.sum(feats*feats, axis=1, keepdims=True) + np.sum(centers*centers, axis=1, keepdims=True).T - 2*np.matmul(feats, centers.T)
        #update assignments
        y = np.argmin(squared_distance, axis=1)
        #squared distance of each point to its center
        d = np.min(squared_distance, axis=1)
        #print objective: sum of squared distance of each point to its assigned center
        obj = np.sum(d)
        print('Iteration {:d}: {:f}'.format(i, obj))
        if np.all(y==yprev):
            break


        #update centers
        for j in range(k):
            if np.sum(y==j) == 0:
                #no data points assigned to the j-th cluster
                #assign data point farthest from its assigned cluster
                idx = np.argmin(d)
                d[idx]=0
                y[idx]=j

            #new center is mean of all assigned datapoints
            centers[j,:] = np.mean(feats[y==j,:], axis=0)
        
        yprev=y
    return y

def kmeans_with_color(img, k):
    #reshape img as a 2D feature vector
    nPixels = img.shape[0]*img.shape[1]
    feats = np.reshape(img, (nPixels, img.shape[2]))
    
    #run kmeans with these color features
    y = kmeans(feats, k)
    
    #reshape y into a 2D array
    y = np.reshape(y, (img.shape[0], img.shape[1]))
    return y

def kmeans_with_color_posn(img, k, alpha):
    #reshape img as a 2D feature vector
    nPixels = img.shape[0]*img.shape[1]
    colorfeats = np.reshape(img, (nPixels, img.shape[2]))
    
    #get position of a pixel as a feature
    posfeats = np.zeros((nPixels, 2))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #loop to set the x value of the position feature
            posfeats[i*img.shape[1]+j,0] = j
            #loop to set the y value of the position feature
            posfeats[i*img.shape[1]+j,1] = i
            
            
    #applies alpha to the position features
    posfeats = alpha*posfeats
    
    #concatenates both sets of features
    feats = np.concatenate((colorfeats, posfeats), axis=1)
    
    #run kmeans with these color features
    y = kmeans(feats, k)
    
    #reshape y into a 2D array
    y = np.reshape(y, (img.shape[0], img.shape[1]))
    return y