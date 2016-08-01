#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import constants
import sys
from scipy.stats import gaussian_kde

# Lorimer 2003 radial distribution
const = 64.6
index = 2.35
sigma = -1.528
rmax = 25
dstep = 0.25
nstep = rmax/dstep
nstep = int(nstep)
dmax = 50.0
histbins = 200

gcradius = dstep * (np.arange(nstep) + 1.0)
gcarea = 2.0 * np.pi * np.power(dstep,2.0) * (np.arange(nstep) + 1.0)

# rho is the density per kpc^2
exp_part = np.exp(gcradius/sigma)
rho = const * np.power(gcradius,index)
rho = rho * exp_part

# multiply by area to get number of pulsars and normalise
# this is the PDF
np_radius = np.multiply(rho,gcarea)
np_radius = np_radius / np.sum(np_radius)

# pick a radial distance from this PDF
nmax = 10000
zscale = 0.1
r = np.random.choice(gcradius,size=nmax,p=np_radius)
# angle theta
theta = np.random.uniform(0.0,2.0*np.pi,nmax)
# compute x,x^2,y,y^2
x = np.multiply(r,np.cos(theta))
x2 = np.power(x,2.0)
y = np.multiply(r,np.sin(theta))
y2 = np.power(y,2.0)
# height z
z = zscale * np.random.randn(nmax)
z2 = np.power(z,2.0)
# gc total distance
dgc = np.sqrt(x2 + y2 + z2)
# earth x distance
xearth = 8.0 + x
x2earth = np.power(xearth,2.0)
# earth total distance
dearth = np.sqrt(x2earth + y2 + z2)

# create PDF
# NB earthbin has one more element than earthist so delete the first one
earthhist,earthbin = np.histogram(dearth,bins=histbins,range=(0.0,dmax))
earthbin = np.delete(earthbin,0,0)
earthpdf = earthhist.astype(float) / nmax

fluxconst = 1.0*1.0/np.power(10,16.0)
dist = np.random.choice(earthbin,size=100000,p=earthpdf)
print " less than 1 kpc", np.sum(dist < 1.0)
print " less than 3 kpc", np.sum(dist < 3.0)
print " less than 5 kpc", np.sum(dist < 5.0)
print " less than 8 kpc", np.sum(dist < 8.0)
print " less than 20kpc", np.sum(dist < 20.0)
dist = np.power(dist,2.0)
lumin = np.zeros(100000,dtype=float)
lumin.fill(np.power(10,30.0))
flux = fluxconst * np.sqrt(lumin) / dist
bool = flux > 1.0
print np.sum(bool)

#plot
xyobs = np.vstack([xearth,y])
zobs = gaussian_kde(xyobs)(xyobs)
fig, ax = plt.subplots()
ax.scatter(xearth, y, c=zobs, s=10, edgecolor='')
plt.show()
