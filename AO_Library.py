import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import argrelmax, argrelextrema
from scipy.optimize import curve_fit
import Function_Library as VUW


def find_neighbourhood_lines(Nneigh, z, x):
    '''
    for each hit in an event, consider the Nneigh nearest points
    and fit a straight line - x = mz + c
    return m, c
    '''
    
    zdiff = np.add.outer(-z, z)                          #(nhits,nhits)
    xdiff = np.add.outer(-x, x)                          #(nhits,nhits)
    r = ( zdiff**2 + xdiff**2 ) ** 0.5                   #(nhits,nhits)
    
    
    #in each row find indices of Nneigh nearest neighbours (+1 discounts itself)
    idx = np.argsort(r, axis=1)[:,1:Nneigh+1]            #(nhits,Nneigh)
    
    #select the z, x coordinates of points nearest to each point
    zneigh = z[idx]                                      #(nhits,Nneigh)
    xneigh = x[idx]                                      #(nhits,Nneigh)

    #get mean z and x values surrounding each point
    zmean = np.mean(zneigh, axis=1)[:,None]              #(nhits,1)
    xmean = np.mean(xneigh, axis=1)[:,None]              #(nhits,1)
    
    #gradient and y-intercept of lines
    zx_sum = np.sum((zneigh - zmean) * (xneigh - xmean), axis=1) #(nhits)
    zz_sum = np.sum((zneigh - zmean) ** 2, axis=1)               #(nhits)
    m = np.true_divide(zx_sum, zz_sum)
    c = np.squeeze(xmean) - m * np.squeeze(zmean)
    
    return m, c



def all_lines_intersects(m, c, plot=False):
    '''
    For a set of N straight lines, there are N(N-1)/2 intersections in the plane
    return the z,x coordinates of all of these

    for 2 lines: x = m1 z + c1; x = m2 z + c2
    they intersect at zint = (c2-c1)/(m1-m2)c 
    
    
    from above requirement, clearly require the difference between
    all possible pairs of m1, m2
    np.ufunc.outer will double count since it considers both mi-mj and mj-mi
    this is corrected by the masking step
    '''
    
    #first eliminate the case where there is only 1 line
    if len(m)==1:
        return (m, c), False
    
    
    mdiff = np.add.outer(-m, m)                                    #(nhits, nhits)
    cdiff = np.add.outer(-c, c)                                    #(nhits, nhits)
    
    z_intercept = -np.true_divide(cdiff, mdiff, where=(mdiff!=0))  #(nhits, nhits)
    x_intercept = m * z_intercept + c                              #(nhits, nhits)
    
    #only want the upper triangle
    mask = np.tri(z_intercept.shape[0], dtype=bool)
    z_intercept = np.ma.array(z_intercept, mask=mask)
    x_intercept = np.ma.array(x_intercept, mask=mask)
    
    intercepts = np.column_stack((z_intercept.compressed(), x_intercept.compressed()))
    return intercepts, True


def gradient_histogram(m, hits_per_bin, plot=True):
    '''
    Simply plot a histogram of some data, m
    and return all useful information about
    '''
    
    #number of data points -> number of bins
    nhits = len(m)
    nbins = int(nhits/hits_per_bin)
    #plot the histogram
    mfreq, bins = np.histogram(m[~np.isnan(m)], nbins)
    
    #which bin is each m in?
    m_which_bins = np.digitize(m, bins) - 1
    m_which_bins[m_which_bins>=nbins] = nbins-1
    
    #other bin information
    bin_centres = 0.5 * np.add(bins[:-1], bins[1:])
    binsize = (bins[-1] - bins[0])/(nbins-1)
    
    if plot:
        mfreq, bins,_ = plt.hist(m[~np.isnan(m)], nbins)
    
    return mfreq, bins, bin_centres, nbins, binsize, m_which_bins


def multi_gauss(x, *params):
    '''
    Define the function for a superposition of some number of gaussians
    params looks like [A0,mu0,sigma0, A1,mu1,sigma1, ...]
    where Ai give the height of each gaussian
    mui give the centres and sigmai, the standard deviation
    '''
    x = x[:,None]
    params = np.array(params)
    mu = params[::3]
    A = params[1::3]
    sigma = params[2::3]
    #print(mu, A, sigma)
    
    a = -0.5*((x-mu)/sigma)**2 #exponent
    y = A * np.exp(a)
    y = np.sum(y, axis=1)
    return y


def histogram_gaussian_fit(mfreq, bin_centres, order=10, Asig=0.1, plot=True):
    '''
    Plot a histogram of all the gradients of local lines
    Expect each particle to be near(est) hits in the same particle
    so the gradient gives the trajectory of the particle
    This means that particles should be given by clumped gradient peaks
    Fit some number of gaussians to the histogram to find the best
    '''
    
    x_data = bin_centres
    y_data = mfreq
    nhits = len(y_data)
    
    #get some estimate of the number of peaks from local maxima
    localmaxs = argrelextrema(y_data, np.greater, order=order)[0]
    maxsort = np.argsort(y_data[localmaxs])
    sortedpeaks = localmaxs[maxsort]
    npeaks = len(localmaxs)
    
    if npeaks==0:
        return None, None
    

    
    attempts = 0
    p0, lowerbound, upperbound = [], [], []
    successes = []
    while attempts < 6:
        p0 += [ x_data[localmaxs[attempts]], y_data[localmaxs[attempts]],  0.5 * y_data[localmaxs[attempts]]]
        lowerbound += [-np.inf, 0, 0]
        upperbound += [np.inf, np.inf, np.inf]
        try:
            params, _ = curve_fit(multi_gauss, x_data, y_data, p0, bounds=(lowerbound,upperbound))
            successes.append(params)
            #if successful plot
            if plot:
                plt.plot(x_data, y_data, '.', label='mfreq')
                yfit = multi_gauss(x_data, *params)
                plt.plot(x_data, yfit, label='Gaussian Fit')
                plt.title('peaks={}'.format(attempts + 1))
                plt.legend()
                plt.show()
            
            attempts+=1
            
        except RuntimeError:
            attempts += 1
    
    error = []
    for success in successes:
        fit = multi_gauss(x_data, *success)
        error.append(np.sum((fit-y_data)**2) )
    
    params = successes[np.argmin(error)].reshape(-1,3)
    A = params[:,1]
    max_A = np.amax(A)
    Asizes = A / max_A
    params = params[np.logical_not(Asizes<Asig)] #0.1 in need of testing
    
    
    npeaks = len(params)

    return params, npeaks


def gaussian_to_bin_influence(params, npeaks, nbins, bins, binsize, m_which_bins):
    '''
    Under the above conception, each gaussian should correspond to a particle track
    so need to work out the probability that each particle belongs to each gaussian
    
    Will take a discrete approach where each bin is assigned an influence for each gaussian
    then points are given the influences of the bins into which they fall
    
    This function gives the 'influence' of each gaussian on each bin by the approximate
    fraction of the gaussian's area which lies in said bin (trapesium rule)
    '''

    
    #fit at each bin edge
    mu, A, sigma = params[:,0], params[:,1], params[:,2] #(npeaks)
    bins = bins[:,None]                                  #(nbins+1,1)
    binsfit = A * np.exp(-0.5*((bins-mu)/sigma)**2)      #(nbins+1,npeaks)
    
    #A = 0.5h(a+b)
    areas = 0.5*binsize * np.add(binsfit[:-1,:], binsfit[1:,:]) #(nbins,npeaks)
    
    #Total area of gaussian = A/(sigma*sqrt(2pi))
    area_T = A/(sigma*np.sqrt(2*np.pi))                         #(npeaks)
    bin_influence = areas / area_T                              #(nbins,npeaks)
    bin_influence = (bin_influence / np.sum(bin_influence, axis=1)[:,None]) #(nbins, npeaks)
    most_bin_influence = np.argmax(bin_influence, axis=1)       #(nbins)
    point_influence = bin_influence[m_which_bins]               #(nhits,npeaks)
    most_point_influence = most_bin_influence[m_which_bins]     #(nhits)
    return point_influence, most_point_influence, bin_influence, most_bin_influence



def color_points(point_influence):
    '''
    Colours showers differently
    '''
    color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown', 'gray',
          'lightblue', 'lightgreen', 'purple', 'cyan', 'magenta', 'lime', 'teal', 'olive', 'maroon', 'navy']
    # Create a dictionary to map color values to colors
    colors = {i: color_list[i] for i in range(20)}
    col = [colors[c] for c in point_influence]
    return col

def shower_separater(most_point_influence, plot=False, z=None, x=None):
    npeaks = np.amax(most_point_influence)+1
    showers = []
    for i in range(npeaks):
        showers.append(np.where(most_point_influence==i)[0])
    
    if plot:
        col = color_points(most_point_influence)
        plt.scatter(z, x, c=col)
        plt.show()
    return showers




def shower_average_line(point_influence, m, c, plot=False, event_data=None):
    '''
    To turn these showers into a vertex classification
    must allocate some directionality
    Do this by fitting averaging the m and c
    for all points for all showers weighted by that
    shower's influence on that point
    
    m.shape, c.shape = (nhits)
    point_influence.shape = (nhits, npeaks)
    '''

    m_showers = np.sum(m[:,None]*point_influence, axis=0) / np.sum(point_influence, axis=0)
    c_showers = np.sum(c[:,None]*point_influence, axis=0) / np.sum(point_influence, axis=0)
    
    if plot:
        VUW.plotter(event_data, save=False, flush=False)
        EV, zx, adc, true = event_data
        z, x = zx
        zz = np.array([np.amin(z),np.amax(z)])
        for i in range(len(m_showers)):
            xx = m_showers[i]*zz + c_showers[i]
            plt.plot(zz, xx)
        plt.show()
    
    
    return m_showers, c_showers



def remove_similar_gradients(m, c, a=0.01):
    '''
    If two lines are almost parallel, their intersect will be very far from
    the relevant zone and they likely correspond to the same thing.
    Such artifacts must be removed
    '''
    
    # Sort the gradients
    sorted_gradients = np.sort(m)
    # Find the indices where the difference between consecutive elements is greater than or equal to a
    split_indices = np.flatnonzero(np.abs(np.diff(sorted_gradients)) >= a) + 1
    
    #split into groups of similar gradients
    split_indices = np.hstack([0, split_indices])
    groups = np.split(np.argsort(m), split_indices)
    groups = [group[1:] for group in groups]
    groups = np.concatenate(groups)
    m_reduced = np.delete(m, groups)
    c_reduced = np.delete(c, groups)
    return m_reduced, c_reduced


def AO(event_data, neigh_fraction=0.1, hits_per_bin=5, Asig=0.1, order=10, a=0.01, plot=False):
    EV, zx, adc, true = event_data
    z, x = zx
    nhits = len(z)
    Nneigh = int(nhits * neigh_fraction)
    
    m, c = find_neighbourhood_lines(Nneigh, z, x)
    
    intercepts, VALIDATE = all_lines_intersects(m, c)
    
    mfreq, bins, bin_centres, nbins, binsize, m_which_bins = gradient_histogram(m, hits_per_bin, plot=plot)  
    
    
    params, npeaks = histogram_gaussian_fit(mfreq, bin_centres, plot=plot)
    
    if npeaks==None:
        print("UNABLE TO FIT GAUSSIANS")
        return None, None
    

    point_influence, most_point_influence, bin_influence, most_bin_influence = gaussian_to_bin_influence(params, npeaks, nbins, bins, binsize, m_which_bins)
    showers = shower_separater(most_point_influence, plot=plot, z=z, x=x)
    
    m_showers, c_showers = shower_average_line(point_influence, m, c, plot=plot, event_data=event_data)
    m_showers_red, c_showers_red = remove_similar_gradients(m_showers, c_showers, a=a)
    
    shower_intercepts, VALIDATE = all_lines_intersects(m_showers_red, c_showers_red)
    if not VALIDATE:
        return None, None
    vtx = np.mean(shower_intercepts, axis=0)
    
    return vtx, showers
