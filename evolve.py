#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import constants
import argparse
import sys
from scipy.stats import gaussian_kde

# return magnetic field given P and Pdot
# bk is the constant in the B field equation
bk = 19.505
bk10 = np.power(10.,bk)
def bsurf(p0,p1):
    bsurf = np.sqrt(np.multiply(p0,p1)) * bk10
    return bsurf

# return log age in years given P and Pdot
def age(p0,p1):
    age = np.log10(p0/2.0/p1/secondsPerYear)
    return age

# return log edot given P and Pdot
def edot(p0,p1):
    edot = 45.0 + np.log10((4.0*np.pi*np.pi*p1)/(np.power(p0,3.0)))
    return edot

# Deathline test
# Is the edot less than 10^30 ? If so then the pulsar has crossed the death line
def deathlinetest(p0,p1):
    return edot(p0,p1) < 30.0

# Flux test
# The luminosity is the sqrt(Edot) 
# The flux is luminosity/distance^2
# empirically it seems as if at d=2kpc, S=1mJy for Edot = 10^32
# returns true if not detected
def luminositytest(p0,p1,dist):
    fluxconst = 2.0*2.0/np.power(10,16.0)
    L_edot = np.power(10, 0.5 * (edot(p0,p1)))
    flux = fluxconst * L_edot / np.power(dist,2.0)
    return flux < 0.25

#---------------------------------------------------------------------
def distfill(this_dist):
# create a PDF of distances from earth
# the PDF has nbin values from 0 to dmax
# assume Lorimer 2003 distribution
    drl_const = 64.6
    drl_index = 2.35
    drl_sigma = -1.528
    drl_rmax = 25
    drl_dstep = 0.25
    drl_nstep = drl_rmax/drl_dstep
    drl_nstep = int(drl_nstep)
    drl_dmax = 50.0
    drl_histbins = 200
    drl_zscale = 0.1

# set up the radius bins in step of drl_dstep
# set up the area between bins constant
    gcradius = drl_dstep * (np.arange(drl_nstep) + 1.0)
    gcarea = 2.0 * np.pi * np.power(drl_dstep,2.0) * (np.arange(drl_nstep) + 1.0)

# rho is the density per kpc^2
# see the equation in Lorimer 2003
    drl_exp_part = np.exp(gcradius/drl_sigma)
    drl_rho = drl_const * np.power(gcradius,drl_index)
    drl_rho = drl_rho * drl_exp_part

# multiply by area to get number of pulsars and normalise
# this is the PDF
    npsr_radius = np.multiply(drl_rho,gcarea)
    npsr_radius = npsr_radius / np.sum(npsr_radius)

# radial distance r
    r = np.random.choice(gcradius, size=npsrs, p=npsr_radius)
# angle theta
    theta = np.random.uniform(0.0,2.0*np.pi,npsrs)
# compute x,x^2,y,y^2
    x = np.multiply(r,np.cos(theta))
    x2 = np.power(x,2.0)
    y = np.multiply(r,np.sin(theta))
    y2 = np.power(y,2.0)
# height z
    z = drl_zscale * np.random.randn(npsrs)
    z2 = np.power(z,2.0)
# earth x distance
    xearth = x + 8.0
    x2earth = np.power(xearth,2.0)
# earth total distance
    dearth = np.sqrt(x2earth + y2 + z2)

# create PDF
# NB earthbin has one more element than earthist so delete the first one
    earthhist,earthbin = np.histogram(dearth,bins=drl_histbins,range=(0.0,drl_dmax))
    earthbin = np.delete(earthbin,0,0)
    earthpdf = earthhist.astype(float) / npsrs

    this_dist = np.random.choice(earthbin,size=npsrs,p=earthpdf)
    return this_dist
# end of distfill function
#---------------------------------------------------------------------
# the dither update function
def dither_update(psr_array):
# update p0:  p0 = p1 * delta T
    psr_array[:,0] += psr_array[:,1] * update_rate_sec

# update p1, change braking index only in user-defined multiples of birthrate
# new p1 = B^2 / P0 * dither
    if np.mod(time, stepsize * birthrate) == 0:
        braking_sigma = 5. / (np.sqrt(time))
        psr_array[:,7] = braking_sigma * np.random.randn(current_pulsars) + 1.0
    psr_array[:,1] = np.power(psr_array[:,2],2)/np.power(bk10,2)/psr_array[:,0] * np.abs(psr_array[:,7])

# update magnetic field
    psr_array[:,2] = bsurf(psr_array[:,0],psr_array[:,1])

# update alpha by dividing through by the alpha_update constant
    psr_array[:,3] = psr_array[:,3]/alpha_update
    return psr_array
# end of update function
#---------------------------------------------------------------------
# the brake update function
def brake_update(psr_array,deltaT):
    number_processing = psr_array.shape[0]
    brake_sigma = 0.25
    brake_mean = np.zeros(number_processing)
# this line pulls the braking index back towards 3.0
#    brake_mean = (3.0 - psr_array[:,7])/100.0
# record the freq and freq derivative before update
    v0_old = 1. / psr_array[:,0]
    v1_old = -1.0 * psr_array[:,1] * np.power(v0_old,2)

# update p0:  p0 = p1 * deltaT
    psr_array[:,0] += psr_array[:,1] * deltaT

# update the braking index -- must be between 2.3 and 7.0
# then update p1 - formula from Johnston&Galloway
#    print "Updating ", number_processing, " source(s)"
    extra = (brake_sigma * np.random.randn(number_processing)) + brake_mean
    psr_array[:,7] = psr_array[:,7] + extra
    psr_array[:,7] = np.clip(psr_array[:,7],0.7,8.5)
    v0_new = 1. / psr_array[:,0]
    tempthing = deltaT * (psr_array[:,7] - 1.0)
    v1_new = -1.0 * v0_new/(tempthing - v0_old[:number_processing]/v1_old[:number_processing])
    psr_array[:,1] =  -1.0 * v1_new * np.power(psr_array[:,0],2)

# update magnetic field
    psr_array[:,2] = bsurf(psr_array[:,0],psr_array[:,1])

# update alpha by dividing through by the alpha_update constant
    psr_array[:,3] = psr_array[:,3]/alpha_update
    return psr_array
# end of update function
#---------------------------------------------------------------------
# generate the pulsar array function
# the array has the following columns
# 0: period 
# 1: period derivative 
# 2: magnetic field 
# 3: alpha
# 4: zeta
# 5: index number
# 6: distance
# 7: braking index
def generatePSRs(npsr, muP0, muP1, sigmaP0, sigmaP1):
    p0_array = np.power(10.,np.random.normal(np.log10(muP0), sigmaP0, npsr))
    p1_array = np.power(10.,np.random.normal(muP1, sigmaP1, npsr))
    this_bsurf = bsurf(p0_array, p1_array)
    this_a = np.arccos(np.random.uniform(0.0, 1.0, npsr))
    this_zeta = np.arccos(np.random.uniform(0.0, 1.0, npsr))
    this_index = np.arange(npsr,dtype=int)
    this_dist = np.zeros(npsr)
    this_dist = distfill(this_dist)
    braking_noise = np.full(npsrs, 3.0)

    p0p1baz = np.column_stack((p0_array, p1_array, this_bsurf, this_a, this_zeta, this_index, this_dist, braking_noise))
    return p0p1baz
# end of generate function
#---------------------------------------------------------------------
# MAIN CODE STARTS HERE
# Parse arguments
parser = argparse.ArgumentParser(description='Evolve pulsars on the P-Pdot diagram')
parser.add_argument('-alpha', metavar="<alpha>", type=float, default='45', help='inclination angle in degrees (default = 45)')
parser.add_argument('-p0', metavar="<p0>", type=float, default='0.020', help='period in s (default = 0.02 s)')
parser.add_argument('-p1', metavar="<p1>", type=float, default='-12.0', help='log p-dot in s/s (default = -12 )')
#parser.add_argument('-edotmax', metavar="<edotmax>", type=float, default='34.0', help='log of max edot detected (default = 34 )')
parser.add_argument('-s0', metavar="<s0>", type=float, default='0.0001', help='log sigma of p0 distribution (default = 0.0001)')
parser.add_argument('-s1', metavar="<s1>", type=float, default='0.0001', help='log sigma of p1 (default = 0.0001 )')
parser.add_argument('-timestep', metavar="<timestep>", type=int, default='10', help='time step in years (default = 10 )')
parser.add_argument('-maxtime', metavar="<maxtime>", type=int, default='8', help='log max age in years (default = 8 )')
parser.add_argument('-birthrate', metavar="<birthrate>", type=int, default='100', help='birth rate in years (default = 100 )')
parser.add_argument('-update', metavar="<update_rate>", type=int, default='100', help='update step in years (default = 100 )')
#parser.add_argument('-npsr', metavar="<npsr>", type=int, default='100000', help='total number of pulsars to generate (default = 100000 )')
parser.add_argument('-iseed', metavar="<iseed>", type=int, default='4', help='integer seed for a pseudo-random number generator (default = 4)')
parser.add_argument('-file', metavar="<file>", default='ppdot.dat', help='ppdot file (default = psr.list)')

secondsPerYear = 365.25*86400.0
args = parser.parse_args()
alpha = args.alpha
p0_0 = args.p0
p1_0 = args.p1
s0_0 = args.s0
s1_0 = args.s1
maxtime = args.maxtime
timestep = args.timestep
birthrate = args.birthrate
update_rate = args.update
update_rate_sec = update_rate * secondsPerYear
#npsrs = args.npsr
npsrs = np.power(10,maxtime)/birthrate
total_steps = np.power(10,maxtime)/timestep
iseed = args.iseed
file = args.file
print "maxtime ",maxtime," birthrate ",birthrate," update rate ",update_rate," npsrs ",npsrs
if np.mod(birthrate,timestep) != 0:
   sys.exit("Error: timestep must divide evenly into birthrate !!")
if np.mod(update_rate,timestep) != 0:
   sys.exit("Error: timestep must divide evenly into update rate !!")
# the remaining time for the "observe" step in seconds
leftover_step = np.remainder(birthrate,update_rate) * secondsPerYear

# the observed p-pdot diagram read in from file generated by psrcat and plotted
ppdot_diagram = np.loadtxt(file)
known_pulsars = ppdot_diagram.shape[0]
x = np.zeros(known_pulsars)
y = np.zeros(known_pulsars)
for i in range(known_pulsars):
    x[i] = np.log10(ppdot_diagram[i,0])
    y[i] = np.log10(ppdot_diagram[i,1])
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=10, edgecolor='')

# USEFUL VALUES
# em_height is the emission height in km
# rho_const because rho = 3.0 * sqrt((pi^2 * height)/(2 P0 c))
# alpha_decay is the alpha decay time in years
# taken from Weltevrede & Johnston (7e7 years)
# alpha update is the fractional change in alpha per birthrate
lightspeed = 299792.0
em_height = 300.0
rho_const = 3.0 * np.sqrt(np.pi * em_height / 2.0 / lightspeed)
alpha_decay = np.power(10.,7.845) 
alpha_update = np.exp(update_rate / alpha_decay)

#generate the pulsar array
psr_array = generatePSRs(npsrs, p0_0, p1_0, s0_0, s1_0)
observed_pulsars = np.zeros(psr_array.shape)
current_pulsars = npsrs

# set stuff to zero before commencing on loop
dead_pulsars = 0
detected = 0
not_beaming = 0
death_liners = 0
weaks = 0
time = 0
psrcount = 0
# START MAIN LOOP
for i in range(total_steps):
    tot_dead_pulsars = 0
    time = (i+1) * timestep

# check to see if it is time for an update
    if np.mod(time, update_rate) == 0:
#    psr_array = dither_update(psr_array)
        psr_array = brake_update(psr_array,update_rate_sec)

# rho is the cone opening angle = 3 * sqrt (pi/2 * height / P0 / c)
# beta = abs(zeta - alpha) - absolute value to check for +/- rho
        rho = rho_const / np.sqrt(psr_array[:,0])
        beta = np.abs(psr_array[:,4] - psr_array[:,3])

# Check which pulsars are already not beaming towards you
# this is the case if beta > rho
        dead_pulsars = 0
        dead_index = []
        for j in range(current_pulsars):
            if beta[j] > rho[j]:
                dead_pulsars +=1
                not_beaming +=1
                dead_index.append(j)
# delete these pulsars from the array
        psr_array = np.delete(psr_array,dead_index,0)
        current_pulsars -= dead_pulsars
        tot_dead_pulsars += dead_pulsars

# Scythe through the array looking for pulsars below the death or under-luminous
    if np.mod(time, 100*update_rate) == 0:
        dead_pulsars = 0
        dead_index = []
        for j in range(current_pulsars):
# death line test
            if deathlinetest(psr_array[j,0],psr_array[j,1]):
                dead_index.append(j)
                dead_pulsars +=1
                death_liners +=1
# luminosity test          
            elif luminositytest(psr_array[j,0],psr_array[j,1],psr_array[j,6]):
                dead_index.append(j)
                dead_pulsars +=1
                weaks +=1
# delete these pulsars from the array
        psr_array = np.delete(psr_array,dead_index,0)
        current_pulsars -= dead_pulsars
        tot_dead_pulsars += dead_pulsars

    if current_pulsars == 0:
        break

# Now test the pulsar we are about to observe (or not)
# once every birthrate years
    if np.mod(time, birthrate) == 0:
# it must have the same array index as the pulsar number !!    
        if psr_array[0,5] == psrcount:
# must also update (nb leftover_step can be zero)
            to_observe = psr_array[0][np.newaxis,:]
            brake_update(to_observe, leftover_step)
            to_observe = to_observe.reshape(psr_array.shape[1])
#            rho = rho_const / np.sqrt(psr_array[0,0])
            rho = rho_const / np.sqrt(to_observe[0])
#            beta = np.abs(psr_array[0,4] - psr_array[0,3])
            beta = np.abs(to_observe[4] - to_observe[3])
# beta test
            if beta > rho:
                dead_pulsars +=1
                not_beaming +=1
                tot_dead_pulsars += 1
# death line test
            elif deathlinetest(to_observe[0],to_observe[1]):
                dead_pulsars += 1
                death_liners += 1
                tot_dead_pulsars += 1
# luminosity test          
            elif luminositytest(to_observe[0],to_observe[1],to_observe[6]):
                dead_pulsars += 1
                weaks += 1
                tot_dead_pulsars += 1
# Detection!            
            else:
                observed_pulsars[psrcount] = to_observe
                detected += 1
# Top one has either been observed or executed so delete it from the pulsar array
            psr_array = np.delete(psr_array,0,0)
            current_pulsars -= 1
            print "Time: ", time, " beaming: ", not_beaming," weak ", weaks, " dead ",death_liners, " remaining: ", current_pulsars, "detected: ", detected
# update the pulsar count and print loop info
        psrcount += 1
# END OF MAIN LOOP
# -----------------------
print "Maxtime (yr): ",maxtime, "Birthrate (yr): ", birthrate, "Update rate (yr): ", update_rate," npsrs: ", npsrs
print "Dead info: beaming: ", not_beaming," --- death-liners: ", death_liners, "--- too weak: ", weaks

# observed_pulsars array was same size as original psr_array, but is
# only populated with detections, so remove blank rows
dropped_index = []
for i in range(npsrs):
    if observed_pulsars[i,0] == 0:
        dropped_index.append(i)
    else:
        observed_pulsars[i,3] = edot(observed_pulsars[i,0],observed_pulsars[i,1])
observed_pulsars = np.delete(observed_pulsars, dropped_index, 0)

# for detected ones, overwrite alpha and zeta with edot and age
# write to output file
observed_pulsars[:,4] = age(observed_pulsars[:,0],observed_pulsars[:,1])
np.savetxt('simulated_ppdot.txt', observed_pulsars)

# plot
xobs = np.log10(observed_pulsars[:,0])
yobs = np.log10(observed_pulsars[:,1])
xyobs = np.vstack([xobs,yobs])
zobs = gaussian_kde(xyobs)(xyobs)
#fig, ax = plt.subplots()
ax.scatter(xobs, yobs, c=zobs, s=10, edgecolor='')
plt.show()
