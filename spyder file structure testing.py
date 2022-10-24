# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:08:17 2022

@author: alexg
"""

from uproot_io import Events, View
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

interaction_dictionary = {}
with open('interactions.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = int(row.pop('Idx'))
        interaction = row.pop('Interaction')
        interaction_dictionary[key] = interaction
        
print(interaction_dictionary[3])
print(interaction_dictionary[6])

events = Events("CheatedRecoFile_1.root")
events_testing = events.reco_hits_x_u

print(events_testing)

def view_creator(event_orientation, event_number, thresholding_on_off=True):
    adc_new = []
    z_new = []
    x_new = []
    if event_orientation == "u":
        view_u = View(events, "u")
        z1_view_u = view_u.z[event_number]
        x1_view_u = view_u.x[event_number]
        adc1_view_u = view_u.adc[event_number]
        if thresholding_on_off == True:
            threshold_value = np.mean(adc1_view_u) - 0.3 * np.std(adc1_view_u)
            for i in range(len(adc1_view_u)):
                if adc1_view_u[i] > threshold_value:
                    adc_new.append(adc1_view_u[i])
                    z_new.append(z1_view_u[i])
                    x_new.append(x1_view_u[i])
            plt.scatter(z_new, x_new, c=adc_new, cmap='jet', s=15)
            plt.title(event_orientation+str(event_number))
            plt.show()
        else:  
            plt.scatter(z1_view_u, x1_view_u, c=adc1_view_u, cmap='jet', s=15)
            plt.title(event_orientation+str(event_number))
            plt.show()
        
    if event_orientation == "v":
        view_v = View(events, "v")
        z1_view_v = view_v.z[event_number]
        x1_view_v = view_v.x[event_number]
        adc1_view_v = view_v.adc[event_number]
        if thresholding_on_off == True:
            threshold_value = np.mean(adc1_view_v) - 0.3 * np.std(adc1_view_v)
            for i in range(len(adc1_view_v)):
                if adc1_view_v[i] > threshold_value:
                    adc_new.append(adc1_view_v[i])
                    z_new.append(z1_view_v[i])
                    x_new.append(x1_view_v[i])
                    
            plt.scatter(z_new, x_new, c=adc_new, cmap='jet', s=15)
            plt.title(event_orientation+str(event_number))
            plt.show()
        else:
            plt.scatter(z1_view_v, x1_view_v, c=adc1_view_v, cmap='jet', s=15)
            plt.title(event_orientation+str(event_number))
            plt.show()
    
    if event_orientation == "w":
        view_w = View(events, "w")
        z1_view_w = view_w.z[event_number]
        x1_view_w = view_w.x[event_number]
        adc1_view_w = view_w.adc[event_number]
        if thresholding_on_off == True:  
            threshold_value = np.mean(adc1_view_w) - 0.3 * np.std(adc1_view_w)
            for i in range(len(adc1_view_w)):
                if adc1_view_w[i] > threshold_value:
                    adc_new.append(adc1_view_w[i])
                    z_new.append(z1_view_w[i])
                    x_new.append(x1_view_w[i])
                
            plt.scatter(z_new, x_new, c=adc_new, cmap='jet', s=15)
            plt.title(event_orientation+str(event_number))
            plt.show()
        else:
            plt.scatter(z1_view_w, x1_view_w, c=adc1_view_w, cmap='jet', s=15)
            plt.title(event_orientation+str(event_number))
            plt.show()
    return

def scatter_points(plane, event_number):
    if plane=="u":
        view_u = View(events, "u")
        true_x = view_u.true_vtx_x[event_number]
        z_view = view_u.z[event_number]
        x_view = view_u.x[event_number]
        adc_view = view_u.adc[event_number]
        true_z = view_u.true_vtx_z[event_number]
    elif plane=="v":
        view_v = View(events, "v")
        true_x = view_v.true_vtx_x[event_number]
        z_view = view_v.z[event_number]
        x_view = view_v.x[event_number]
        adc_view = view_v.adc[event_number]
        true_z = view_v.true_vtx_z[event_number]
    elif plane=="w":
        view_w = View(events, "w")
        true_x = view_w.true_vtx_x[event_number]
        z_view = view_w.z[event_number]
        x_view = view_w.x[event_number]
        adc_view = view_w.adc[event_number]
        true_z = view_w.true_vtx_z[event_number]
    else:
        print("PLane not u,v,w")
    return x_view, z_view, adc_view, true_z, true_x

varia = scatter_points("u", 10)

'''             
view_u = View(events, "u")
view_v = View(events, "v")
view_w = View(events, "w")

z_view_w = view_w.z

z1_view_w = view_w.z[67]
x1_view_w = view_w.x[67]
adc1_view_w = view_w.adc[67]

x1_view_v = view_v.x[1]
z1_view_v = view_v.z[1]
adc1_view_v = view_v.adc[1]

print(np.std(adc1_view_w))
print(np.mean(adc1_view_w))

adc_new = []
z_new = []
x_new = []

threshold_value = np.mean(adc1_view_w) - 0.5 * np.std(adc1_view_w)

for i in range(len(adc1_view_w)):
    if adc1_view_w[i] > threshold_value:
        adc_new.append(adc1_view_w[i])
        z_new.append(z1_view_w[i])
        x_new.append(x1_view_w[i])

plt.scatter(z1_view_v, x1_view_v, c=adc1_view_v, cmap='winter', s=20)
plt.show()

plt.scatter(z_new, x_new, c=adc_new, cmap='summer', s=15)
plt.show()

'''