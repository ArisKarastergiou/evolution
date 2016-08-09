#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import constants
import argparse
import sys
from scipy.stats import gaussian_kde

# read in the files
ppdot_diagram = np.loadtxt("ppdot.dat")
observed_pulsars = np.loadtxt('simulated_ppdot.txt')

#ppdot for the known pulsars
known_pulsars = ppdot_diagram.shape[0]
x = np.zeros(known_pulsars)
y = np.zeros(known_pulsars)
for i in range(known_pulsars):
    x[i] = np.log10(ppdot_diagram[i,0])
    y[i] = np.log10(ppdot_diagram[i,1])
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# ppdot for simulated pulsars
xobs = np.log10(observed_pulsars[:,0])
yobs = np.log10(observed_pulsars[:,1])
xyobs = np.vstack([xobs,yobs])
zobs = gaussian_kde(xyobs)(xyobs)


plt.subplot(223)
plt.plot(x,y,'bo')
plt.plot(xobs,yobs,'go')
plt.axis([-2.0,1.0,-17.0,-11.0])

plt.subplot(221)
plt.hist(x,20,range=(-2.0,1.0),normed=True,histtype='step')
plt.hist(xobs,20,range=(-2.0,1.0),normed=True,histtype='step')

plt.subplot(222)
plt.hist(y,20,range=(-17.0,-11.0),normed=True,histtype='step')
plt.hist(yobs,20,range=(-17.0,-11.0),normed=True,histtype='step')

plt.subplot(224)
#fig, ax = plt.subplots()
plt.scatter(x, y, c=z, s=10, edgecolor='')
plt.scatter(xobs, yobs, c=zobs, s=10, edgecolor='')
plt.axis([-2.0,1.0,-17.0,-11.0])
plt.show()
