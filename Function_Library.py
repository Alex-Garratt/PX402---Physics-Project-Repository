#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#PACKAGES

from uproot_io import Events, View
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import random
import numba
from numba import jit
from sklearn.cluster import KMeans
import itertools
import heapq
import matplotlib.collections as col
from concurrent.futures import ThreadPoolExecutor

# In[ ]:CONSTANTS
thr_std=0.2 #some constants
n_events=9310

# In[ ]:
def load_data():
    print("Running...")
    toc = time.perf_counter() #start timer

    E = Events("CheatedRecoFile_1.root") #import data

    view_u = View(E, "u") #unpack to view
    view_v = View(E, "v")
    view_w = View(E, "w")

    global u_x, u_z, u_adc, u_truevtx_z, u_truevtx_x
    global v_x, v_z, v_adc, v_truevtx_z, v_truevtx_x
    global w_x, w_z, w_adc, w_truevtx_z, w_truevtx_x

    u_x = view_u.x #unpack each plane
    u_z = view_u.z
    u_adc = view_u.adc
    u_truevtx_z = view_u.true_vtx_z
    u_truevtx_x = view_u.true_vtx_x

    v_x = view_v.x #unpack each plane
    v_z = view_v.z
    v_adc = view_v.adc
    v_truevtx_z = view_v.true_vtx_z
    v_truevtx_x = view_v.true_vtx_x

    w_x = view_w.x #unpack each plane
    w_z = view_w.z
    w_adc = view_w.adc
    w_truevtx_z = view_w.true_vtx_z
    w_truevtx_x = view_w.true_vtx_x

    tic = time.perf_counter() #stop timer
    tt = tic-toc
    print("All data loaded in", int(tt/60),"minutes and", tt%60,"seconds")
    return



def unpack_event(plane, event_number, thresholding=True, no_std=thr_std, ordering=True):
    #given plane and event number, extract required information and threshold if required
    if plane=="u": #unpack data from required plane
        x = u_x[event_number]
        z = u_z[event_number]
        adc = u_adc[event_number]
        truevtxz = u_truevtx_z[event_number]
        truevtxx = u_truevtx_x[event_number]
    elif plane=="v": #unpack data from required plane
        x = v_x[event_number]
        z = v_z[event_number]
        adc = v_adc[event_number]
        truevtxz = v_truevtx_z[event_number]
        truevtxx = v_truevtx_x[event_number]
    elif plane=="w": #unpack data from required plane
        x = w_x[event_number]
        z = w_z[event_number]
        adc = w_adc[event_number]   
        truevtxz = w_truevtx_z[event_number]
        truevtxx = w_truevtx_x[event_number]
    else:
        print("PLane not u,v,w")
        
    if thresholding: #remove data points with energies far away from the mean
        mean = np.average(adc)
        std = np.std(adc)
        x = x[(mean-no_std*std<adc)]
        z = z[(mean-no_std*std<adc)]
        adc = adc[(mean-no_std*std<adc)]
        
    if ordering: #order data along z such that AoI[0] is min_z and AoI[-1] is max_z
        order = np.argsort(z)
        z = np.take_along_axis(z, order, axis=0)
        x = np.take_along_axis(x, order, axis=0)
        adc = np.take_along_axis(adc, order, axis=0)
    
    EV = (plane, event_number)
    zx = (z, x)
    true = (truevtxz, truevtxx)
    return EV, zx, adc, true
def plotter(event_data, save=False, flush=False):
    EV, zx, adc, true = event_data
    plane, event = EV
    print(str(event))
    z, x = zx
    
    plt.scatter(z, x, c=adc, cmap="viridis", zorder=2)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$x$')
    if save:
        plt.savefig(str(plane)+'-plane_'+str(event)+'.png')
    if flush:
        plt.show()
    return 0


def event(event_number):
    Aoi1, _ = unpack_event('u', event_number)
    Aoi2, _ = unpack_event('v', event_number)
    Aoi3, _ = unpack_event('w', event_number)
    
    return Aoi1, Aoi2, Aoi3
def event_plotter(event_number):
    u, v, w = event(event_number)
    zu, xu, adcu = u
    zv, xv, adcv = v
    zw, xw, adcw = w
    fig, axs = plt.subplots(3)
    axs[0].scatter(zu, xu, c=adcu, cmap="viridis", s=5)
    axs[1].scatter(zv, xv, c=adcv, cmap="viridis", s=5)
    axs[2].scatter(zw, xw, c=adcw, cmap="viridis", s=5)
    
    return 0



def leastz_scores(AoI):
    z, x, adc = AoI
    if z[-1]==z[0]:
        return [1]
    scores = (z[-1]-z)/(z[-1]-z[0])
    return scores




#rphi stuff
def rfunc(r, epsilon, n):
    #functional r dependence - optimal at inv_sqrt(r)
    return 1/(r+epsilon)**n
rfunc = np.vectorize(rfunc)
def rphi_scores(event_data, rdep=rfunc, no_sectors=12, epsilon=0.01, n=0.5):
    #take the point number to make a centre, divide into radial sectors and plot a histogram of where points lie
    
    EV, zx, adc, true = event_data
    z, x = zx
    nhits = len(z)
    scores = np.zeros(nhits)

    
    for i in range(nhits):
        cen_x = x - x[i] #centre data around i
        cen_z = z - z[i]

        theta = np.arctan2(cen_x, cen_z) #get all polar angles
        r2 = ( cen_x**2 + cen_z**2 ) ** 0.5 #and squared polar radial coordinate

        rweight = rdep(r2, epsilon, n)
        hist_data, hist_bins = np.histogram(theta, no_sectors, (-np.pi,np.pi), weights=rweight) #turn into histogram data
        hist_data = (hist_data - np.roll(hist_data, shift=int(0.5*no_sectors))) #subract mirrored data

        score = np.sum(hist_data**2)/nhits**2 # score for this point as centre
        scores[i] = score**0.5 #normalise by size of event

            
    return scores



#quadvertex stuff
def quadvertex(event_data, no_samples=2000, binwidth=(100,100)):
    
    EV, zx, adc, true = event_data
    z, x = zx
    nhits = len(z) #number of data points in event
    
    offset = 100
    
    zrange = np.array([z[0]-offset, z[-1]+offset])
    xrange = np.array([np.amin(x)-offset, np.amax(x)+offset])
    
    seed = 46576877
    np.random.seed(seed)
    points = np.random.randint(0, nhits, size=(no_samples, 4)) #generate 4 random point indices for each sample
    #should really disallow duplicates in inner dimension heere but don't know how so ignore
    sample_z = z[points] #(no_samples,4) array of z co-ords
    sample_x = x[points] # and their corresponding xs
    sample_zx = np.concatenate((sample_z[...,None], sample_x[...,None]), axis=2) #combine into 1 array shape=(no_samples,4,2)
    
    combinations = np.array([[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]])
    np.seterr(divide='ignore')
    #(nosamples,4,2)-> (nosamples,3,2,2,2) # for each 4 points, get 3 sets of 2 lots of 2 points, (z,x)
    z_x = sample_zx[:,combinations]
    delz_x = np.subtract(z_x[...,1,:],z_x[...,0,:]) #subtract zs and xs (nosamples,3,2,2)
    m = np.true_divide(delz_x[...,1], delz_x[...,0], where=(delz_x[...,0]!=0)) #slope of lines (nosamples,3,2)
    c = z_x[...,0,1] - z_x[...,0,0] * m #x-axis intercept (no_samples,3,2)

    
    #now compare pairs of lines to see where they intersect
    m_c = np.concatenate((m[...,None], c[...,None]), axis=3) #combine into one array (no_samples,3,2,2)
    delm_c = np.subtract(m_c[...,1,:],m_c[...,0,:]) #subtract ms and cs (nosamples,3,2)
    zints = np.true_divide(delm_c[...,1], -delm_c[...,0], where=(delm_c[...,0]!=0)) #z-intersects (nosamples,3)
    xints = m_c[...,0,0] * zints + m_c[...,0,1] #x-intersects (nosamples,3)
    
    intersections = np.concatenate((zints[...,None], xints[...,None]), axis=2).reshape(no_samples*3, 2)
    filt = (zrange[0]<intersections[:,0]) & (intersections[:,0]<zrange[1]) & (xrange[0]<intersections[:,1]) & (intersections[:,1]<xrange[1])
    intersections = intersections[filt] #filter values outside of range
    hist = plt.hist2d(intersections[:,0], intersections[:,1], bins=binwidth, cmap=plt.cm.jet) #2d hist data
    plt.clf()
    return hist
def quadvertex_scores(event_data, no_samples=2000, binwidth=(100,100), binradius=3):
    EV, zx, adc, true = event_data
    z, x = zx
    h, zedges, xedges, image = quadvertex(event_data, no_samples, binwidth) #import heavy lifting from qv main function
    zbins = np.digitize(z, zedges) -1 #for each z,x work out its bin coords
    xbins = np.digitize(x, xedges) -1
    zxbins = np.concatenate((zbins[...,None],xbins[...,None]), axis=1) #combine z,x bin coords (L,2)
    
    
    
    zbins[zbins<binradius] =  binradius #catch edge cases
    zbins[zbins>(binwidth[0]-2*binradius)] = binwidth[0]-2*binradius
    xbins[xbins<binradius] =  binradius
    xbins[xbins>(binwidth[1]-2*binradius)] = binwidth[1]-2*binradius
    
    
    window = (2*binradius+1, 2*binradius+1)
    contribution = np.lib.stride_tricks.sliding_window_view(h, window)[zbins-binradius, xbins-binradius]
    
    scores = np.sum(contribution, axis=(1,2))
    #this implements the window approach - for each point, take a square window with input binradius out of h
    #the total score is the sum over this square window
    
    scores = scores/np.amax(scores) #normalise
    return scores


# In[ ]:

def stuff():
    theta_u = 0.623257100582
    theta_v = -0.623257100582
    theta_w = 0.
    
    cos_u = np.cos(theta_u)
    cos_v = np.cos(theta_v)
    cos_w = np.cos(theta_w)
    sin_u = np.sin(theta_u)
    sin_v = np.sin(theta_v)
    sin_w = np.sin(theta_w)
    
    sin_dvu = np.sin(theta_v - theta_u)
    sin_dwv = np.sin(theta_w - theta_v)
    sin_duw = np.sin(theta_u - theta_w)
    
    # wire to 3D
    def uv_to_y(u, v):
        return ((u * cos_v - v * cos_u) / sin_dvu)
    
    def uv_to_z(u, v):
        return ((u * sin_v - v * sin_u) / sin_dvu)
    
    def uw_to_y(u, w):
        return ((w * cos_u - u * cos_w) / sin_duw)
    
    def uw_to_z(u, w):
        return ((w * sin_u - u * sin_w) / sin_duw)
    
    def vw_to_y(v, w):
        return ((v * cos_w - w * cos_v) / sin_dwv)
    
    def vw_to_z(v, w):
        return ((v * sin_w - w * sin_v) / sin_dwv)
    
    # 3D to wire
    def yz_to_u(y, z):
        return z * cos_u - y * sin_u
    
    def yz_to_v(y, z):
        return z * cos_v - y * sin_v
    
    def yz_to_w(y, z):
        return z * cos_w - y * sin_w
    return



def pdf(plane, events, limit, algorithm_scores, add_args=(), no_pdfbins=30, plot=True):
    #leastz add_args = (dependence=np.exp, plot=False)
    #rphi add_args = (rdep=rfunc, no_sectors=12, centre=0)
    #quadvertex add_args = (no_samples=2000, binwdith=(100,100), binradius=3)
    
    signal = []
    noise = []
    
    for ev in tqdm(events): #ev is an index number
        #print(ev)
        event_data = unpack_event(plane, ev)
        EV, zx, adc, true = event_data
        z, x = zx
        ztrue, xtrue = true
        nhits = len(z)

        scores = algorithm_scores(event_data, *add_args)

        r = ( (z-ztrue)**2 + (x-xtrue)**2 )**0.5 #distances from true vertex
        truth = r<limit

        sig = [i for (i, v) in zip(scores, truth) if v]
        noi = [i for (i, v) in zip(scores, truth) if not v]
        signal.extend(sig)
        noise.extend(noi)
    
    '''
    sigdata, bins = np.histogram(signal)
    noidata, bins = np.histogram(noise)
    sigdata = np.ones(len(signal))*np.sum(sigdata)
    noidata = np.ones(len(noise))*np.sum(noidata)
    '''
    noiweight = np.ones(len(noise)) / len(noise)
    sigweight = np.ones(len(signal)) / len(signal)
    
    plt.clf() #flush previous figure data without printing
    signalpdf, bins, stuff = plt.hist(signal,no_pdfbins, weights=sigweight)
    plt.title('signal')
    plt.savefig('pdf_signal.png')
    if plot:
        plt.show()
    
    noisepdf, bins, stuff = plt.hist(noise,no_pdfbins, weights=noiweight)
    plt.title('noise')
    plt.savefig('pdf_noise.png')
    if plot:
        plt.show()

    np.savetxt('pdfsignal.txt', signalpdf)
    np.savetxt('pdfnoise.txt', noisepdf)
    return signalpdf, noisepdf

'''
no_events = 300
rng = np.random.default_rng()
events = rng.choice(9310, size=no_events, replace=False)


sigpdf1, noipdf1 =  pdf('u', events, 4, linear_least_z_scores, add_args=(), no_pdfbins=30, plot=False)
sigpdf2, noipdf2 =  pdf('u', events, 4, rphi_scores, add_args=(), no_pdfbins=30, plot=False)
sigpdf3, noipdf3 =  pdf('u', events, 4, quadvertex_scores, add_args=(), no_pdfbins=30, plot=False)
np.savetxt('sigpdf1.txt', sigpdf1)
np.savetxt('noipdf1.txt', noipdf1)
np.savetxt('sigpdf2.txt', sigpdf2)
np.savetxt('noipdf2.txt', noipdf2)
np.savetxt('sigpdf3.txt', sigpdf3)
np.savetxt('noipdf3.txt', noipdf3)
'''

# In[ ]:


def likelihood(AoI, signalpdf, noisepdf, no_pdfbins, algorithm_scores, add_args=()):
    #each pdf array consists of no_pdfbins bins covering scores from 0-1
    
    scores = algorithm_scores(AoI, *add_args)
    pdfbins = np.linspace(0,1,no_pdfbins)
    
    scorebins = np.digitize(scores, pdfbins) - 1 #which bin is each score in?
    P_signal = signalpdf[scorebins]
    P_noise = noisepdf[scorebins]
    
    return P_signal, P_noise

def all_likelihoods(AoI, sigpdfs, noipdfs, no_pdfbins, als_scores):
    al1_scores, al2_scores, al3_scores , *alx_scores = als_scores
    sig1pdf, sig2pdf, sig3pdf , *sigxpdf = sigpdfs
    noi1pdf, noi2pdf, noi3pdf , *noixpdf = noipdfs
    
    P_signal1, P_noise1 = likelihood(AoI, sig1pdf, noi1pdf, no_pdfbins, al1_scores)
    P_signal2, P_noise2 = likelihood(AoI, sig2pdf, noi2pdf, no_pdfbins, al2_scores)
    P_signal3, P_noise3 = likelihood(AoI, sig3pdf, noi3pdf, no_pdfbins, al3_scores)
    
    L_signal = P_signal1 * P_signal2 * P_signal3
    L_noise = P_noise1 * P_noise2 * P_noise3
    
    curlyL = L_signal/(L_signal+L_noise)
    
    return curlyL



def analytics(algorithm, no_events, add_args=(), dr68_gr50 = (1,1), seed=677963, plot=True):
    np.random.default_rng(seed=seed)
    events = np.random.choice(n_events, no_events, replace=False)
    
    dr = np.zeros(no_events)
    i=0
    for event in events:
        event_data = unpack_event('u', event)
        EV, zx, adc, true = event_data
        scores = algorithm(Aoi, *add_args)
        if scores == None:
            continue
        alg_vtx_ind = np.argmax(scores)
        alg_vtx = z[alg_vtx_ind], x[alg_vtx_ind]
        
        dr[i] = (  (alg_vtx[0] - true[0]) ** 2 + (alg_vtx[1] - true[1]) ** 2  ) ** 0.5
        i+=1
    

    
    
    sort = np.sort(dr)
    dr68, gr50 = None, None
    
    if dr68_gr50[0]:
        percentile68 = int(no_events * 0.68)
        dr68 = sort[percentile68]
    
    if dr68_gr50[1]:
        ind_gr50 = np.where(sort>50)[0][0]
        gr50 = 100 * (1 - ind_gr50/no_events)
    
    if plot:
        nbins = 40#int(no_events/20)+1
        plt.hist(dr, nbins)
        
        llc = [[dr68,0],[dr68, 1]]
        lc = col.LineCollection(llc, color='k')
        plt.gca().add_collection(lc)
        plt.show()
    
    
    return dr68, gr50




def rphi_tester(no_events):
    n_set = [-10]#, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 10]
    epsilon_set = [1000000, 100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    no_sectors_set = [2]#,3,4,5,6,8,10,15,20,30,40,50,100]
    
    A, B, C = np.meshgrid(no_sectors_set, epsilon_set, n_set, indexing='ij')
    params = np.column_stack((A.ravel(), B.ravel(), C.ravel()))
    
    with ThreadPoolExecutor() as executor:
        results = [executor.submit(analytics, rphi_scores, no_events, param) for param in params]
    
    arr = []
    for result in results:
        arr.append(result.result())
    
    return arr

def vertex3D(vertex_u, vertex_v, vertex_w, naive=True):

    theta_u = 0.623257100582
    theta_v = -0.623257100582
    theta_w = 0.

    cos_u = np.cos(theta_u)
    cos_v = np.cos(theta_v)
    cos_w = np.cos(theta_w)
    sin_u = np.sin(theta_u)
    sin_v = np.sin(theta_v)
    sin_w = np.sin(theta_w)

    sin_dvu = np.sin(theta_v - theta_u)
    sin_dwv = np.sin(theta_w - theta_v)
    sin_duw = np.sin(theta_u - theta_w)

    
    u, x_u = vertex_u
    v, x_v = vertex_v
    w, x_w = vertex_w
    
    x = np.array( [ (x_u), (x_v), (x_w) ] )
    y = np.array( [ ((u * cos_v - v * cos_u) / sin_dvu), ((w * cos_u - u * cos_w) / sin_duw), ((v * cos_w - w * cos_v) / sin_dwv) ] )
    z = np.array( [ ((u * sin_v - v * sin_u) / sin_dvu), ((w * sin_u - u * sin_w) / sin_duw), ((v * sin_w - w * sin_v) / sin_dwv) ] )

    
    if naive:
        xvtx, yvtx, zvtx = np.mean(x), np.mean(y), np.mean(z)
    return xvtx, yvtx, zvtx





def create_feature_space(plane, events, killring):
    no_events = len(events)
    
    #each event will have less than 1000 hits
    #allocate memory for feature space coordinates and signal/background classification
    feature_space = np.zeros((1000*no_events,3))
    classification = np.zeros(1000*no_events)

    #index for adding to these arrays
    i = 0
    
    for event in tqdm(events):
        Aoi, true = VUW.AoI(plane, event)

        z_scores = VUW.leastz_scores(Aoi)
        rphi_scores = VUW.rphi_scores(Aoi)
        quadvertex_scores = VUW.quadvertex_scores(Aoi)

        z, x, adc = Aoi
        truez, truex = true
        n_hits = len(z)

        #distance to true vertex
        deltaz = z - truez
        deltax = x - truex
        deltar = ( deltaz**2 + deltax**2 ) ** 0.5
    
        deltar[deltar>killring] = 0 #points outside killring are background (0)
        deltar[np.nonzero(deltar)] = 1 #points inside killring are signal (1)
        
        #add to feature space
        feature_space[i:i+n_hits] = np.concatenate([z_scores[:,None], rphi_scores[:,None], quadvertex_scores[:,None]], axis=1)
        classification[i:i+n_hits] = deltar
        
        #iterate i
        i += n_hits
    
    #remove unused rows
    feature_space = feature_space[:i]
    classification = classification[:i]
    return feature_space, classification


def BDT(seed, n_sampled, plane, killring, breakdown=(0.6, 0.2, 0.2)):
    np.random.seed(seed)
    events = np.random.choice(n_events, n_sampled, replace=False)
    
    #get number of events in training/validation/testing sections
    n_training = int(breakdown[0]*n_sampled)
    n_validation = int(breakdown[1]*n_sampled)
    n_testing = int(breakdown[2]*n_sampled)

    #allocate each section some events
    training = events[:n_training]
    validation = events[n_training:n_training+n_validation]
    testing = events[n_validation:n_validation+n_testing]
    
    #from these events get the feature space and classification
    fs_training, c_training = create_feature_space(plane, training, killring)
    fs_validation, c_validation = create_feature_space(plane, validation, killring)
    fs_testing, c_testing = create_feature_space(plane, testing, killring)
    
    #TRAINING
    base = DecisionTreeClassifier(max_depth=1)
    bdt = AdaBoostClassifier(base_estimator=base, n_estimators=100, random_state=0)
    bdt.fit(fs_training, c_training)
    print(bdt.score(fs_training, c_training))
    
    #VALIDATION
    print(bdt.score(fs_validation, c_validation))
    
    #TESTING
    print(bdt.score(fs_testing, c_testing))
    return 0


no = 1000
#print(VUW.analytics(VUW.leastz_scores, no))
#print(VUW.analytics(VUW.rphi_scores, no))
#print(VUW.analytics(VUW.quadvertex_scores, no))



