#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:17:38 2018

@author: Guigui
"""

#Define the default folder
import os
os.chdir('/Users/Guigui/Dropbox/Work_LPSC/Sabine_project/Machine_Learning/IDM_xsec')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from scipy.interpolate import griddata
import numpy as np

import pandas as pd # panda library is to import & manage datatsets

#Importing the dataset
dataset = pd.read_csv('IDM_xsecs_13TeV.csv')

#listing the variables
print(dataset.info())

#Separating the feature set from the target variables
#.values converts the pandadataframe into an array
X = dataset.iloc[:,0:5].values

y_3535 = dataset.iloc[:,5].values


#Creating the MH0, MA0 and xsec_3535 array for future plotting
"""
Reminder:
    35 : H0
    36 : A0
    37 : H+
    
    So xsec_3535 is the production cross section of two H0 particles,
       xsec_3636       "    "  "        "       "of two A0 particles,
       xsec_3537       "    "  "        "       "of one H0 and one H+ particle,
       and so on
"""

MH0 = dataset.iloc[:,0].values
MA0 = dataset.iloc[:,1].values
MHC = dataset.iloc[:,2].values
lam2 = dataset.iloc[:,3].values
lamL = dataset.iloc[:,4].values
xsec_3535 = dataset.iloc[:,5].values
xsec_3636 = dataset.iloc[:,6].values
xsec_3737 = dataset.iloc[:,7].values
xsec_3537 = dataset.iloc[:,8].values
xsec_3637 = dataset.iloc[:,9].values
xsec_3735 = dataset.iloc[:,10].values
xsec_3736 = dataset.iloc[:,11].values
xsec_3536 = dataset.iloc[:,12].values
"""
minimum = 999999
maximum = 0

for i in range(0,len(MH0)):
#    print(MH0[i])
    if MH0[i] < minimum:
        minimum = MH0[i]
        imin = i
        
    if MH0[i] > maximum:
        maximum = MH0[i]
        imax = i
        

print(imin)
print(minimum)
print(imax)
print(maximum)
"""


# Convert from pandas dataframes to numpy arrays
MH0_dat, MA0_dat, xsec_3535_dat, = np.array([]), np.array([]), np.array([])
xsec_3636_dat, xsec_3737_dat, xsec_3537_dat = np.array([]), np.array([]), np.array([])
xsec_3637_dat, xsec_3735_dat, xsec_3736_dat = np.array([]), np.array([]), np.array([])
xsec_3536_dat, MHC_dat = np.array([]), np.array([]) 
lamL_dat, lam2_dat = np.array([]), np.array([])

for i in range(len(MH0)):
        MH0_dat = np.append(MH0_dat,MH0[i])
        MA0_dat = np.append(MA0_dat,MA0[i])
        MHC_dat = np.append(MHC_dat,MHC[i])
        lamL_dat = np.append(lamL_dat,lamL[i])
        lam2_dat = np.append(lam2_dat,lam2[i])
        xsec_3535_dat = np.append(xsec_3535_dat,xsec_3535[i])
        xsec_3636_dat = np.append(xsec_3636_dat,xsec_3636[i])
        xsec_3737_dat = np.append(xsec_3737_dat,xsec_3737[i])
        xsec_3537_dat = np.append(xsec_3537_dat,xsec_3537[i])
        xsec_3637_dat = np.append(xsec_3637_dat,xsec_3637[i])
        xsec_3735_dat = np.append(xsec_3735_dat,xsec_3735[i])
        xsec_3736_dat = np.append(xsec_3736_dat,xsec_3736[i])
        xsec_3536_dat = np.append(xsec_3536_dat,xsec_3536[i])
"""        
MH0_dat.max()
MA0_dat.min()
xsec_3535_dat.max()
"""        
# create x-y points to be used in heatmap
xi = np.linspace(MA0_dat.min(),MA0_dat.max(),1000)
yi = np.linspace(MHC_dat.min(),MHC_dat.max(),1000)

xi_2 = np.linspace(lamL_dat.min(),lamL_dat.max(),1000)
yi_2 = np.linspace(MA0_dat.min(),MA0_dat.max(),1000)

# Z is a matrix of x-y values
zi = griddata((MA0_dat, MHC_dat), xsec_3637_dat, (xi[None,:], yi[:,None]), method='nearest'
              ,rescale=True)

zi_2 = griddata((lamL_dat, MA0_dat), xsec_3636_dat, (xi_2[None,:], yi_2[:,None]), method='nearest'
              ,rescale=True)

# I control the range of my colorbar by removing data 
# outside of my range of interest
zmin = zi.min()
zmax = zi.max()
zi[(zi<zmin) | (zi>zmax)] = None

zmin2 = zi_2.min()
zmax2 = zi_2.max()
zi_2[(zi_2<zmin2) | (zi_2>zmax2)] = None

# Create the contour plot
#CS = plt.contourf(xi, yi, zi, cmap=plt.cm.rainbow)
#plt.colorbar()  
#plt.show()


fig, ax = plt.subplots()
#Palette PuBu_r
#cs = ax.contourf(xi, yi, zi, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
#Palette Rainbow
cs = ax.contourf(xi, yi, zi, locator=ticker.LogLocator(), cmap=plt.cm.rainbow)


cbar = fig.colorbar(cs)
#cbar.set_label('xsec [pb]',rotation=270)
cbar.set_label('xsec [pb]')

plt.title('p p -> A0 H+')
plt.xlabel('MA0 [GeV]')
plt.ylabel('MH+ [GeV]')



fig2, ax2 = plt.subplots()
#Palette PuBu_r
#cs = ax.contourf(xi, yi, zi, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
#Palette Rainbow
cs2 = ax2.contourf(xi_2, yi_2, zi_2, locator=ticker.LogLocator(), cmap=plt.cm.rainbow)

cbar2 = fig2.colorbar(cs2)
#cbar.set_label('xsec [pb]',rotation=270)
cbar2.set_label('xsec [pb]')

plt.title('p p -> A0 A0')
plt.xlabel('MA0 [GeV]')
plt.ylabel('MH+ [GeV]')


plt.show()