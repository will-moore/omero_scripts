import warnings
import numpy as np
import math
from math import pi
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
import scipy.ndimage.filters as filters
from scipy import stats
from scipy.fftpack import fft2,ifft2,fftshift

def create_distance_matrix(pointsA,pointsB):   
    return np.sort(cdist(np.array(pointsA),np.array(pointsB),'euclidean'))

def nearest_neighbour(distMat,col):
    nnDist = distMat[:,col]
    return nnDist
    
def range_search(distMat,dataSize,rangeProps):
    rmax = rangeProps[1]
    rangeMat = np.zeros((rmax,dataSize))
    i = 0
    for idist in range(rmax):
        j = 0
        for idata in range(dataSize):
            idxA = np.where(distMat[i,:] < idist)
            rangeMat[i,j] = len(idxA[0].nonzero())
            j += 1
        i += 1
    return rangeMat
    

"""
    Filter localisations by either intensity or distance
"""
def calc_stats(data,loc_max_pos_x,loc_max_pos_y,loc_max_amp):
    data_unif,data_std = spatial_ave(data)
    bgMeanMax = data_unif[loc_max_pos_x,loc_max_pos_y]
    bgMeanStd = data_std[loc_max_pos_x,loc_max_pos_y]
    pValue = 1 - stats.norm.cdf(loc_max_amp,bgMeanMax,bgMeanStd)
    return bgMeanMax,bgMeanStd,pValue
    
def filter_coords(data,filter_type,thresh,loc_max_pos_x,loc_max_pos_y,loc_max_amp,bg_mean,bg_std,p_value):
    if filter_type == 'intensity':
        keepMax = np.where(p_value < thresh)[0]
    elif filter_type == 'distance':
        dist = create_distance_matrix([loc_max_pos_x,loc_max_pos_y],[loc_max_pos_x,loc_max_pos_y])
        nnDist = nearest_neighbour(dist,1)
        keepMax = np.where(nnDist > thresh)[0]
    
    loc_max_pos_x = loc_max_pos_x[keepMax,:]
    loc_max_pos_y =  loc_max_pos_y[keepMax,:]
    loc_max_amp = loc_max_amp[keepMax,:]
    bg_mean = bg_mean[keepMax,:]
    bg_std = bg_std[keepMax,:]
    p_value = p_value[keepMax,:]
    #numLocalMax = keepMax.shape[0] 
    return loc_max_pos_x,loc_max_pos_y,loc_max_amp,bg_mean,bg_std,p_value 

def spatial_ave(data):
    #use a uniform filter to get the background
    data_unif = filters.uniform_filter(data,size=11)
    data_std = filters.generic_filter(data,np.std,size=11)
    return data_unif,data_std  
    
def get_coords(fit_data,num_frames,nm_per_pixel):
    if type(fit_data[0]).__name__ == 'MoleculeList':
        x = []
        y = []
        for f in range(num_frames):
            for m in range(fit_data[f].num_peaks):
                x.append(fit_data[f].molecules[m].x*nm_per_pixel)
                y.append(fit_data[f].molecules[m].y*nm_per_pixel)
    else:
        x = fit_data[:,1]*nm_per_pixel
        y = fit_data[:,2]*nm_per_pixel
    return np.array([x,y]).T

def get_coords_in_roi(data,roi,num_frames=None,nm_per_pixel=None):
    all_coords = get_coords(data,num_frames,nm_per_pixel)  
    return all_coords[(all_coords[:,0] > roi[0]) & (all_coords[:,0] < roi[1])
                      & (all_coords[:,1] > roi[2]) & (all_coords[:,1] < roi[3])]

def ripleykfunction(dataXY,dist_scale,box,method):
    N,cols = dataXY.shape
    print box    
    rbox = np.min([ dataXY[:,0] - box[0], box[1] - dataXY[:,0], dataXY[:,1] - box[2], box[3] - dataXY[:,1] ])
    A = (box[1]-box[0])*(box[3]-box[2])
    dist = create_distance_matrix(dataXY,dataXY).T
    print 'dist shape = ',dist.shape
    if method == 0: # no correction...
        L = np.zeros((dist_scale.shape[0],1))
        for i in range(dist_scale.shape[0]):
            K = A*np.sum(np.sum(dist[1:,:]<dist_scale[i],axis=0))/N**2
            L[i] = np.sqrt(K/pi) - dist_scale[i]  
    elif method == 1: # edge correction
        L = np.zeros((dist_scale.shape[0],1))
        Nk = dist_scale.shape[0]
        for i in range(Nk):
            index = np.where(rbox > dist_scale[i])[0]
            if index.any():
                K = A*np.sum(np.sum(dist[1:,index]<dist_scale[i],axis=0))/(index.shape[0]*N)
                L[i] = np.sqrt(K/pi) - dist_scale[i]
    return L

def cart2pol(x,y,z):
    theta = np.arctan2(x,y)
    r = np.sqrt(x**2+y**2)
    v = z
    return theta,r,v

def pc_corr(image1=None,image2=None,region=None,rmax=100):
    if image1 is None:
        raise 'Input at least one image to calculate correlation', image1
    
    if region is None:
        raise 'must create an roi'
    else:
        # roi will be [xmin,xmax,ymin,ymax]
        mask = np.zeros(image1.shape,dtype='float64')
        mask[region[0]:region[1],region[2]:region[3]] = 1.0

    N1 = np.sum(image1*mask) # number of particles within mask

    if image2 is None:
        N2 = sum(sum(image1*mask)) # number of particles within mask
        I1 = image1.astype('float64')       # convert to double
        I2 = image1.astype('float64')
        L1 = I1.shape[0]+rmax # size of fft2 (for zero padding)
        L2 = I1.shape[1]+rmax # size of fft2 (for zero padding)
    else:
        N2 = sum(sum(image2*mask))
        I1 = image1.astype('float64')       # convert to double
        I2 = image1.astype('float64')
        L1 = I1.shape[0]+rmax # size of fft2 (for zero padding)
        L2 = I2.shape[1]+rmax # size of fft2 (for zero padding)

    A = float(np.sum(mask))      # area of mask  
    
    NP = np.real(fftshift(ifft2(np.abs(fft2(mask, (L1, L2)))**2)))# Normalization for correct boundary conditions
    #G1 = A**2/N1**2*np.real(fftshift(ifft2(np.abs(fft2(I1*mask,(L1, L2)))**2)))#/NP
    G1 = (A**2/N1/N2)*np.real(fftshift(ifft2(fft2(I1*mask,(L1, L2))*np.conj(fft2(I2*mask, (L1, L2))))))/NP
    xmin = math.floor(L2/2+1)-rmax
    ymin = math.floor(L1/2+1)-rmax
    w = h = 2*rmax+1
    G = G1[xmin:xmin+w,ymin:ymin+h]  #only return valid part of G

    xvals = np.ones((1, 2*rmax+1)).T*np.linspace(-rmax,rmax,2*rmax+1)    #map to x positions with center x=0
    yvals = np.outer(np.linspace(-rmax,rmax,2*rmax+1),np.ones((1, 2*rmax+1)))    #map to y positions with center y=0
    zvals = G
    theta,r,v = cart2pol(xvals,yvals,zvals)  # convert x, y to polar coordinates
    Ar = np.reshape(r,(1, (2*rmax+1)**2))
    Avals = np.reshape(v,(1, (2*rmax+1)**2))
    ind = np.argsort(Ar,axis=1)
    rr = np.sort(Ar,axis=1)
    vv = Avals[:,ind]
    rbins = [i-0.5 for i in range(int(math.floor(np.max(rr)))+1)]      # the radii you want to extract
    bin = np.digitize(rr[0,:], rbins)      # bin by radius
    g = np.ones((1,rmax+1))
    dg = np.ones((1,rmax+1)) 
    for j in range(1,rmax+1):
        m = bin == j
        n2 = np.sum(m)                 # the number of pixels in that bin
        if n2>0:
            warnings.simplefilter("error", RuntimeWarning)
            try:
                g[:,j-1] = np.sum(m*vv)/n2   # the average G values in this bin
                dg[:,j-1] = np.sqrt(np.sum(m*(vv-g[:,j-1])**2))/n2 # the variance of the mean
            except RuntimeWarning:
                pass
        else:
            g[:,j-1] = 0.0
            dg[:,j-1] = 0.0
            
    r = range(rmax+1)

    G[rmax+1, rmax+1] = 0
    return G,r,g,dg,mask
