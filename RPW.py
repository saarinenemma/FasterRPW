import numpy as np
import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping
import torch
import math
from jpype import JArray, JDouble, JInt
from scipy.optimize import brentq

def RPW_LMR(batch_size, X_array, Y_array, C=None, delta=0.001, k=1, p=2):
    new_delta = math.pow(delta / 2, p)
    
    nz = batch_size
    alphaa = 4.0*np.max(C)/new_delta
    s1 = alphaa * nz
    # s1 = 1

    gtSolver = Mapping(nz, X_array, Y_array, C, new_delta, float(s1), float(1 / alphaa), nz ** 2, False, p)
    APinfo = np.array(gtSolver.getAPinfo()) # augmenting path info

    if APinfo.size == 0:
        return 0

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]
    duals = APinfo_cleaned[:,4] / float(alphaa)
    betas = APinfo_cleaned[:,2] / float(s1)
    
    # print(betas)
    # print(duals)
    # APinfo_cleaned = APinfo

    # cost_AP = duals * betas
    cost_AP = APinfo_cleaned[:,3] / float(alphaa) / float(s1)
    cumCost = np.cumsum(cost_AP)
    # cumCost = np.cumsum(cost_AP)/(alphaa*alphaa*nz)

    # print(cumCost)

    cumFlow = np.cumsum(betas)
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)
    # print(flowProgress)

    d_cost = (k * (1 - flowProgress)) ** p - cumCost
    # print(d_cost[-1], flowProgress[-1], cumCost[-1])
    try:
        d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
    except:
        return 0
    d_ind_b = d_ind_a + 1
    
    # # print("Index ", d_ind_a, d_ind_b)
    
    alpha = find_intersection_point_p(flowProgress[d_ind_a], cumCost[d_ind_a], flowProgress[d_ind_b], cumCost[d_ind_b], p, k=k)

    res = 1 - alpha
    return res, gtSolver.getTimeTaken()


def RPW_ICML(batch_size, X_array, Y_array, C=None, delta=0.001, k=1, p=2):
    nz = batch_size
    s1 = 2 * p * nz / delta
    # s1 = 1
    max_iters = int(((2 + 3 / (2 * p - 1)) * p) / (delta ** 2))
    
    min_val = 1
    guess = delta / 4
    java_time = 0
    while guess <= min_val + delta / 4:
        g = math.pow(guess, p)
        s2 = (g * delta) / (2 * p)
        # print("Guess: ", guess, "s2: ", s2)
        gtSolver = Mapping(nz, X_array, Y_array, C, delta, float(s1), float(s2), max_iters, True, p)
        APinfo = np.array(gtSolver.getAPinfo()) # augmenting path info
        java_time += gtSolver.getTimeTaken()
        # print("Java time: ", java_time)

        if APinfo.size == 0:
            continue

        clean_mask = (APinfo[:,2] >= 1)
        APinfo_cleaned = APinfo[clean_mask]
        duals = APinfo_cleaned[:,4] * float(s2)
        betas = APinfo_cleaned[:,2] / float(s1)
        
        # print(betas)
        # print(duals)
        # APinfo_cleaned = APinfo

        # cost_AP = duals * betas
        # print(cost_AP)
        cost_AP = APinfo_cleaned[:,3] * float(s2) / float(s1)
        cumCost = np.cumsum(cost_AP)
        # cumCost = np.cumsum(cost_AP)/(alphaa*alphaa*nz)

        # print(cumCost)

        cumFlow = np.cumsum(betas)
    
        estimate = (k * (1 - cumFlow[-1])) ** p
        if estimate > cumCost[-1]:
            if min_val > 1 - cumFlow[-1]:
                min_val = 1 - cumFlow[-1]
        else:
            d_cost = (k * (1 - cumFlow)) ** p - cumCost
            # print(d_cost[-1], flowProgress[-1], cumCost[-1])
            try:
                d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
            except:
                guess = guess + delta
                continue
            d_ind_b = d_ind_a + 1
            
            # # print("Index ", d_ind_a, d_ind_b)
            
            alpha = find_intersection_point_p(cumFlow[d_ind_a], cumCost[d_ind_a], cumFlow[d_ind_b], cumCost[d_ind_b], p, k=k)

            res = 1 - alpha
            
            if min_val > res:
                min_val = res
                
        # print(f'Guess: {guess}, min_val: {min_val}')
        
        guess = guess + delta / 4
    return min_val, java_time


def find_intersection_point(x1, y1, x2, y2):
    # x1 < x2
    # y1 > 0
    # y2 < 0
    # y = ax + b
    # find x when y = 0
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    x = -b/a
    return x

def find_intersection_point_p(x0, y0, x1, y1, p, k=0.1):
    # print(x0, y0, x1, y1)
    # Define linear f(x) and h(x) = f(x) - g(x)
    def f_piece(x):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def h(x):
        return f_piece(x) - (k * (1 - x)) ** p

    return brentq(h, x0, x1)  # return only x


def find_intersections(x_points, y_points, p):
    for i in range(len(x_points) - 1):
        x0, x1 = x_points[i], x_points[i + 1]
        y0, y1 = y_points[i], y_points[i + 1]

        # Define the linear piece f(x) over [x0, x1]
        def f_piece(x):
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        # Define the difference h(x) = f(x) - (1 - x)^p
        def h(x):
            return f_piece(x) - (1 - x) ** p

        # Check for a sign change to find root in [x0, x1]
        if h(x0) * h(x1) <= 0:
            try:
                root = brentq(h, x0, x1)
                return root
            except ValueError:
                continue  # Skip if brentq fails

    # print((1 - x_points) ** p - y_points)
    return None


def find_one_intersection(x_points, y_points, p):
    for i in range(len(x_points) - 1):
        x0, x1 = x_points[i], x_points[i + 1]
        y0, y1 = y_points[i], y_points[i + 1]

        # Skip if outside of intersection range
        g0 = (1 - x0) ** p
        g1 = (1 - x1) ** p

        if y0 > g0 and y1 > g1:
            continue
        if y0 < g0 and y1 < g1:
            continue

        # Define linear f(x) and h(x) = f(x) - g(x)
        def f_piece(x):
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        def h(x):
            return f_piece(x) - (1 - x) ** p

        return brentq(h, x0, x1)  # return only x

    return None  # no intersection

