from uproot_io import Events, View
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

print("Running...")


E = Events("CheatedRecoFile_1.root")


view_u = View(E, "u")
view_v = View(E, "v")
view_w = View(E, "w")


u_x = view_u.x
u_z = view_u.z
u_adc = view_u.adc
u_truevtx_z = view_u.true_vtx_z
u_truevtx_x = view_u.true_vtx_x


v_x = view_v.x
v_z = view_v.z
v_adc = view_v.adc
v_truevtx_z = view_v.true_vtx_z
v_truevtx_x = view_v.true_vtx_x

w_x = view_w.x
w_z = view_w.z
w_adc = view_w.adc
w_truevtx_z = view_w.true_vtx_z
w_truevtx_x = view_w.true_vtx_x






def AoI(plane, event_number, thresholding=True, no_std=0.2):
    if plane=="u":
        AoI_x = u_x[event_number]
        AoI_z = u_z[event_number]
        AoI_adc = u_adc[event_number]
        AoI_truevtxz = u_truevtx_z
        AoI_truevtxx = u_truevtx_x
    elif plane=="v":
        AoI_x = v_x[event_number]
        AoI_z = v_z[event_number]
        AoI_adc = v_adc[event_number]
        AoI_truevtxz = v_truevtx_z
        AoI_truevtxx = v_truevtx_x
    elif plane=="w":
        AoI_x = w_x[event_number]
        AoI_z = w_z[event_number]
        AoI_adc = w_adc[event_number]   
        AoI_truevtxz = w_truevtx_z
        AoI_truevtxx = w_truevtx_x
    else:
        print("PLane not u,v,w")
        
    if thresholding:
        mean = np.average(AoI_adc)
        std = np.std(AoI_adc)
        AoI_x = AoI_x[(mean-no_std*std<AoI_adc)]
        AoI_z = AoI_z[(mean-no_std*std<AoI_adc)]
        AoI_adc = AoI_adc[(mean-no_std*std<AoI_adc)]
    
    return AoI_x, AoI_z, AoI_adc, AoI_truevtxz, AoI_truevtxx






def least_z(plane, event_number, thresholding=True, no_std=0.2):
    
    AoI_x, AoI_z, AoI_adc, AoI_truevtxz, AoI_truevtxx = AoI(plane, event_number, thresholding, no_std)
    
    min_z = np.amin(AoI_z)
    ind = np.where(min_z)[0]
    min_x = AoI_x[ind]
    #print("Least z prediction for vertex location: (x-position, z-position, plane)=", min_x, min_z, plane)
    return min_z, min_x









def phi_hist(plane, event_number, thresholding=True, no_std=0.2, leastz=False, centre_coord=(0,0), no_sectors=20, centre=0, printhist=True, optimise=True):
    AoI_x, AoI_z, AoI_adc, AoI_truevtxz, AoI_truevtxx = AoI(plane, event_number, thresholding, no_std)
    
    if leastz:
        centre_coord = least_z(plane, event_number, thresholding, no_std)
    
    
    centred_x = AoI_x - centre_coord[1]
    centred_z = AoI_z - centre_coord[0]

    theta = np.arctan2(centred_x, centred_z)
    hist_data, hist_bins = np.histogram(theta, no_sectors, (-np.pi+centre,np.pi+centre))
    
    hist_sqr = hist_data**2
    score = np.sum(hist_sqr)
    norm_score = (score/len(theta)**2)**0.5
    
    if printhist:
        plt.hist(hist_data, hist_bins)
        plt.show()
    
    
    
    
    
    return score, norm_score


















def accuracy(plane, function, no_events, thresholding=True, no_std=0.2):
    
    Zvert = np.zeros(2, no_events)
    Xvert = np.zeros(2, no_events)
    
    
    for i in tqdm(range(no_events)):
        AoI_x, AoI_z, AoI_adc, Zvert[1,i], Xvert[1,i] = AoI(plane, i, thresholding, no_std)
        Zvert[0,i], Xvert[0,i] = function(plane, i)
        
    delta_z = Zvert[1] - Zvert[0]
    delta_x = Xvert[1] - Xvert[0]
    delta_r2 = delta_z**2 + delta_x**2
    
    data, bins = np.histogram(delta_r2, 30)
    
    plt.hist(data, bins)
    plt.show()













'''
def collate_lz(plane, thresholding=True, no_std=1):
    #no_events = len(u_x)
    no_events = 9310
    
    
    zx_array = np.zeros((2, no_events))
    zx_array_true = np.zeros((2, no_events))
    for i in tqdm(range(no_events)):
        zx_array[0,i], zx_array[1,i] = least_z(plane, i, thresholding, no_std)
        zx_array_true[0,i], zx_array_true[1,i] = true_vtx(plane, i)
    
    return zx_array
'''








print("Finished! :)")   