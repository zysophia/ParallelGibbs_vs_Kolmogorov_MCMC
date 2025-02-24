import numpy as np
import numpy.random as rng
from abc import ABC, abstractmethod 
from copy import deepcopy
import os
import logging
import math
from gibbsChains import *
from algorithms import *
from meanEstimator import *
from tpa import *


n=10
b = -0.001
prefix1="./logs/isingoutput_n="
prefix2="_beta="

log_filename = "{0}{1}{2}{3}.log".format(prefix1, n, prefix2,b)

os.makedirs(os.path.dirname(log_filename), exist_ok=True)
file_handler = logging.FileHandler(log_filename, mode="a", encoding=None, delay=False)
logging.basicConfig(handlers=[file_handler], level=logging.DEBUG)

logging.info("----- This is a new run -----")

chain = IsingChainLattice(n = n)
logging.info("Ising model, n = %d", n)

bmin=b
bmax = 0.0
# eps = 0.005

# delta and kappa are calculated so that the probablity of sucess is at least 0.8. Note Kol takes: delta=0.75-kappa
delta = 0.3
kappa = 0.05
dist = 64

chain.beta = bmin

# calculate real_z
# dp = [[]]
# for i in range(n**2):
#     for j in range(len(dp)):
#        d = dp.pop(0)
#        dp.append(d+[0])
#        dp.append(d+[1])
# z = 0
# for d in dp:
#    hh = chain.get_Hamiltonian(d)
#    z += np.exp(-bmin*hh)
# real_z = z/2**(n**2)
# print("real z:", real_z)
# logging.info("real value z = %.20f", real_z)

# TPA for parallel and super Gibbs
tao_dict = {256: 1.260, 128: 1.372, 64:1.539, 32: 1.794, 16: 2.197, 8: 2.86, 4:4.0}

Hmax = chain.get_Hmax()
Hmin = chain.get_Hmin()
gamma = 0.24
tao = tao_dict[dist]
m = tao/2/np.log(1+ gamma) * np.log(Hmax)
k = int(m*dist)
print("k = ", k)
chain.beta = bmin
q = np.log(chain.get_upper_Q())

for eps in [  0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025]:
#for eps in [0.001, 0.0005]:



    
# we first calculated tvd based on Kolmogorov's calculations for the TPA schedule

    etilt = 1 - 1/np.sqrt((1+eps))
    r = int(np.ceil(2/etilt**2))
    tvdkol = kappa/ (k*q + m*q*r + 3*r + 1)

# using this tvd we find the schedule
    res = TPA_k_d(bmin, bmax, k, dist, chain, tvdkol)
    schedule, TPAsteps = res["schedule"], res["steps"]
   
    print(f"m={m}, k={k}, q={q}, r={r}, tvd (calculated using kolmogov work)={tvdkol}")


    logging.info("parameters: bmin = %.3f, bmax = %.3f, eps = %.4f, delta = %.2f, kappa = %.2f", \
        bmin, bmax, eps, delta, kappa)
    # logging.info("real value z = %.20f", real_z)
    logging.info("tpa params: steps = %d", TPAsteps)

    # We not run both algorithm using tvdkol
    compute = False
    print(f"Running Kolmogorov (compute z {compute})...")
    steps1, steps2,  kol1TPAsteps, est_z = kolmogorov(schedule = schedule, TPAsteps = TPAsteps, e = eps, kappa=kappa ,gibbsChain = chain, bmin = bmin, bmax = bmax, d = dist, compute_z = compute)
    print(f"kolmogorov with eps_kol takes {steps1} steps, while z = {est_z}, and TPA takes {kol1TPAsteps} steps")
    print(f"kolmogorov with Hoeffding takes {steps2} steps (Hoeffding bound), while z = {est_z}, and TPA takes {kol1TPAsteps} steps")


    if (compute):
        logging.info("Kolmogorov with eps_kol (compute z) takes %d steps, while z = %.20f, and TPA takes %d steps, ", steps1, est_z, kol1TPAsteps)
        logging.info("Kolmogorov with Hoeffding (compute z) takes %d steps, while z = %.20f, and TPA takes %d steps, ", steps2, est_z, kol1TPAsteps)
    else:
        logging.info("Kolmogorov with eps_kol takes %d steps, and TPA takes %d steps", steps1, kol1TPAsteps)
        logging.info("Kolmogorov with Hoeffding takes %d steps, and TPA takes %d steps", steps2, kol1TPAsteps)


    z, steps = parallelGibbs(schedule = schedule, TPAsteps = TPAsteps, bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = True)
    print("parallelGibbs takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm parallelGibbs, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)

    z, steps = parallelGibbs(schedule = schedule, TPAsteps = TPAsteps, bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = False)
    print("MCMCPro takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    logging.info("RunAlgorithm MCMCPro, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)

    # logging.info("multiplicative error is %.7f", z/real_z - 1)
        
    # z, steps = superGibbs(schedule = schedule, TPAsteps = TPAsteps,bmin = bmin, bmax = bmax, gibbsChain = chain, eps = eps, delta = delta, kappa= kappa, d = dist, trace = True)
    # print("superGibbs takes ", steps, "steps, while z = ", z, "TPA takes ", TPAsteps, "steps")
    # logging.info("RunAlgorithm superGibbs, takes %d steps, while z = %.20f, and TPA takes %d steps", steps, z, TPAsteps)
    # logging.info("multiplicative error is %.7f", z/real_z - 1)

        