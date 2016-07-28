#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import constants
import argparse
import sys
from scipy.stats import gaussian_kde

def bsurf(p0,p1):
    bsurf = np.sqrt(p0*p1) * bk10
    return bsurf

def age(p0,p1):
    age = p0/2.0/p1/secondsPerYear/1.e6
    return age

def edot(p0,p1):
    edot = 45.0 + np.log10((4.0*np.pi*np.pi*p1)/(np.power(p0,3)))
    return edot

def generatePSRs(npsr, muP0, muP1, sigmaP0, sigmaP1):
    p0_array = np.power(10.,np.random.normal(np.log10(muP0), sigmaP0, npsr))
    p1_array = np.power(10.,np.random.normal(muP1, sigmaP1, npsr))
    this_bsurf = np.zeros(npsr)
    this_a = np.arccos(np.random.uniform(0.0, 1.0, npsr))
    this_zeta = np.arccos(np.random.uniform(0.0, 1.0, npsr))
    this_index = np.arange(npsr)
    for i in range(npsr):
        this_bsurf[i] = bsurf(p0_array[i], p1_array[i])

    p0p1baz = np.column_stack((p0_array, p1_array, this_bsurf, this_a, this_zeta, this_index))
    return p0p1baz
     

# Parse arguments

parser = argparse.ArgumentParser(description='Evolve pulsars on the P-Pdot diagram')
parser.add_argument('-alpha', metavar="<alpha>", type=float, default='45', help='inclination angle in degrees (default = 45)')
parser.add_argument('-p0', metavar="<p0>", type=float, default='0.020', help='period in s (default = 0.02 s)')
parser.add_argument('-p1', metavar="<p1>", type=float, default='-12.0', help='log p-dot in s/s (default = -12 )')
parser.add_argument('-s0', metavar="<s0>", type=float, default='0.0001', help='log sigma of p0 distribution (default = 0.0001)')
parser.add_argument('-s1', metavar="<s1>", type=float, default='0.0001', help='log sigma of p1 (default = 0.0001 )')
parser.add_argument('-maxage', metavar="<maxage>", type=float, default='8', help='log max age in years (default = 8 )')
parser.add_argument('-birthrate', metavar="<birthrate>", type=float, default='100', help='birth rate in years (default = 100 )')
parser.add_argument('-stepsize', metavar="<stepsize>", type=float, default='1', help='update step in units of birthrate (default = 1 )')
parser.add_argument('-npsr', metavar="<npsr>", type=int, default='100000', help='total number of pulsars to generate (default = 100000 )')
parser.add_argument('-iseed', metavar="<iseed>", type=int, default='4', help='integer seed for a pseudo-random number generator (default = 4)')
parser.add_argument('-file', metavar="<file>", default='ppdot.dat', help='ppdot file (default = psr.list)')


args = parser.parse_args()

alpha = args.alpha
p0_0 = args.p0
p1_0 = args.p1
s0_0 = args.s0
s1_0 = args.s1
maxage = args.maxage
birthrate = args.birthrate
stepsize = args.stepsize
npsrs = args.npsr
iseed = args.iseed
file = args.file

ppdot_diagram = np.loadtxt(file)
known_pulsars = ppdot_diagram.shape[0]
x = np.zeros(known_pulsars)
y = np.zeros(known_pulsars)
for i in range(known_pulsars):
    x[i] = np.log10(ppdot_diagram[i,0])
    y[i] = np.log10(ppdot_diagram[i,1])
#data = ppdot_diagram
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=10, edgecolor='')

# bk is the log constant in the B~sqrt(p*pdot) function
bk = 19.505
bk10 = np.power(10.,bk)
secondsPerYear = 365.25*86400.0
alpha_decay = np.power(10.,7) 
alpha_update = np.exp(birthrate / alpha_decay)

stepSec = birthrate * secondsPerYear
print "step in seconds ", stepSec
print p0_0, p1_0
psr_array = generatePSRs(npsrs, p0_0, p1_0, s0_0, s1_0)
print psr_array.shape
current_pulsars = npsrs
dropped_pulsars = np.zeros(psr_array.shape)
print dropped_pulsars.shape
total_steps = npsrs
dead_pulsars = 0
detected = 0
tot_dead_pulsars = 0
for i in range(total_steps):
    print "Loop info: ", i, "killed: ", tot_dead_pulsars," remaining: ", current_pulsars, "detected: ", detected
    time = (i+1) * birthrate

#    braking_noise = np.full(current_pulsars, 1.0)
# update p0
    psr_array[:,0] += psr_array[:,1] * stepSec

# update p1 only in user-defined multiples of birthrate
    if np.mod(time, stepsize * birthrate) == 0:
        braking_sigma = 5. / (np.sqrt(time))
        braking_noise = braking_sigma * np.random.randn(current_pulsars) + 1.0
        psr_array[:,1] = np.power(psr_array[:,2],2)/np.power(bk10,2)/psr_array[:,0] * np.abs(braking_noise)
# update b
        psr_array[:,2] = np.sqrt(np.multiply(psr_array[:,0],psr_array[:,1])) * bk10


# update alpha
    psr_array[:,3] = psr_array[:,3]/alpha_update
    rho = 3.0 * np.sqrt(np.pi * 300 / 2.0 / psr_array[:,0] / 3.e5)
    beta = psr_array[:,4] - psr_array[:,3]
#    print "beta, rho, alpha ", 180./np.pi * beta,180./np.pi * rho, psr_array[:,3]*180./np.pi
# Check which pulsars are already not observable
    tot_dead_pulsars = 0
    dead_pulsars = 0
    top = 0
    dead_index = []
    for j in range(current_pulsars):
        if np.abs(beta[j]) > rho[j]:
            dead_pulsars +=1
            dead_index.append(j)
    psr_array = np.delete(psr_array,dead_index,0)
#    if len(dead_index) > 0 and dead_index[0] == 0:
#        top = 1
    current_pulsars -= dead_pulsars
    tot_dead_pulsars += dead_pulsars
    dead_pulsars = 0
    dead_index = []
    if np.mod(time, 10000) == 0:
        for j in range(current_pulsars):
# death line test
            if psr_array[j,2]/np.power(psr_array[j,0],2) < 0.17e12:
                dead_index.append(j)
                dead_pulsars +=1
# luminosity test          
            else:
                L_edot = np.sqrt(4. * np.pi * np.pi * psr_array[j,1] / np.power(psr_array[j,0],3) * 1e11)
                random_number = np.random.uniform(0.,1.,1)
                if  random_number > L_edot:
                    dead_index.append(j)
                    dead_pulsars +=1

#    if len(dead_index) > 0 and dead_index[0] == 0:
#        top = 1
    psr_array = np.delete(psr_array,dead_index,0)
    current_pulsars -= dead_pulsars
    tot_dead_pulsars += dead_pulsars


    if current_pulsars == 0:
        break
    L_edot = np.sqrt(4. * np.pi * np.pi * psr_array[0,1] / np.power(psr_array[0,0],3) * 1e11)
    random_number = np.random.uniform(0.,1.,1)
    
#   add to observed array, only if it hasn't already been removed or it hasn't crossed the death line
    if  random_number < L_edot and psr_array[0,2]/np.power(psr_array[0,0],2) > 0.17e12 and psr_array[0,5] == i:
        dropped_pulsars[i] = psr_array[0,:]
        detected += 1
        psr_array = np.delete(psr_array,0,0)
        current_pulsars-=1

#    print "new shape :", psr_array.shape
#    print psr_array
dropped_index = []
for i in range(npsrs):
    if dropped_pulsars[i,0] == 0:
        dropped_index.append(i)

dropped_pulsars = np.delete(dropped_pulsars, dropped_index, 0)

xobs = np.log10(dropped_pulsars[:,0])
yobs = np.log10(dropped_pulsars[:,1])
xyobs = np.vstack([xobs,yobs])
zobs = gaussian_kde(xyobs)(xyobs)
#fig, ax = plt.subplots()
ax.scatter(xobs, yobs, c=zobs, s=10, edgecolor='')
plt.show()
    
#plt.plot(np.log10(dropped_pulsars[:,0]),np.log10(dropped_pulsars[:,1]),'.')
#plt.show()


