import random
from scipy.stats import truncnorm
import numpy as np
import sys
import time
from datetime import datetime
from jpype import JArray, JDouble, JInt

from RPW import *
from utils import *

IS_NORMAL = False
IS_NORMAL_DIFFERENT = False
IS_RANDOM_PLANE = False
IS_HALF_SPACE = False
N = 3000
P = 5
D = 10

def generate_points(n, d):
    """
    Generates the point sets with the corresponding cost matrix
    :param compute_matrix: a function that computes the cost matrix
    """

    masses_a = [1 for _ in range(n)]
    masses_b = [1 for _ in range(n)]
    if IS_NORMAL_DIFFERENT:
        # Define the parameters
        lower_bound = 0
        upper_bound = 1
        mean = 0.4
        std_dev = 0.2
        
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        
        A = [
            np.array(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d))
            for i in range(n)
        ]
        mean = 0.6
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        B = [
            np.array(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d))
            for i in range(n)
        ] 
    if IS_NORMAL_DIFFERENT:
        # Define the parameters
        lower_bound = 0
        upper_bound = 1
        mean = 0.5
        std_dev = 0.4
        
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        
        A = [
            np.array(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d))
            for i in range(n)
        ]
        mean = 0.5
        na = (lower_bound - mean) / std_dev
        nb = (upper_bound - mean) / std_dev
        B = [
            np.array(truncnorm.rvs(na, nb, loc=mean, scale=std_dev, size=d))
            for i in range(n)
        ]
    elif IS_HALF_SPACE:
        A = [
                np.array([float(random.random()) / 2] + [float(random.random()) for _ in range(d-1)])
                for _ in range(n)
            ]
        B = [
                np.array([float(random.random()) / 2 + 0.5] + [float(random.random()) for _ in range(d-1)])
                for _ in range(n)
            ] 
    else:
        A = [
                np.array([float(random.random()) for _ in range(d)])
                for _ in range(n)
            ]
        B = [
                np.array([float(random.random()) for _ in range(d)])
                for _ in range(n)
            ]
    # before = time.time()
    # C = compute_matrix(np.array([p.coordinates for p in A + B]))
    # C = compute_matrix(A + B)
    # after = time.time()

    return A, B, masses_a, masses_b


def round_masses(A, B, delta):
    total_removed_mass = 0
    for a in A:
        prev_adj_mass = a.adj_mass
        a.adj_mass = math.ceil(a.mass / delta) * delta
        if a.transported_mass > 0 and prev_adj_mass > a.adj_mass:
            edge = a.map[list(a.map.keys())[0]]
            edge.mass -= delta
            a.transported_mass -= delta
            edge.point_b.transported_mass -= delta
            total_removed_mass += delta
    for b in B:
        b.adj_mass = math.floor(b.mass / delta) * delta
    return total_removed_mass


if __name__ == "__main__":
    N_min = N if len(sys.argv) < 4 else int(sys.argv[1])
    N_max = N if len(sys.argv) < 4 else int(sys.argv[2])
    N_step = 1 if len(sys.argv) < 4 else int(sys.argv[3])
    P_min = P if len(sys.argv) < 7 else int(sys.argv[4])
    P_max = P if len(sys.argv) < 7 else int(sys.argv[5])
    P_step = 1 if len(sys.argv) < 7 else int(sys.argv[6])
    D_min = D if len(sys.argv) < 10 else int(sys.argv[7])
    D_max = D if len(sys.argv) < 10 else int(sys.argv[8])
    D_step = 1 if len(sys.argv) < 10 else int(sys.argv[9])
    DELTA_min = 0.1 if len(sys.argv) < 13 else float(sys.argv[10])
    DELTA_max = 0.1 if len(sys.argv) < 13 else float(sys.argv[11])
    DELTA_step = 0.01 if len(sys.argv) < 13 else float(sys.argv[12])
    

    now = datetime.now()
    # Format the datetime object
    formatted_datetime = now.strftime("%Y-%m-%d-%H-%M")

    filename = (
        "N"
        + str(N_min)
        + "-"
        + str(N_max)
        + "_P"
        + str(P_min)
        + "-"
        + str(P_max)
        + "_D"
        + str(D_min)
        + "-"
        + str(D_max)
        # + "_Normal"
        # + str(IS_NORMAL)
        + "_date"
        + str(formatted_datetime)
        + ".txt"
    )
    if IS_NORMAL:
        filename = "Normal_" + filename
    elif IS_RANDOM_PLANE:
        filename = "Plane_" + filename
    elif IS_NORMAL_DIFFERENT:
        filename = "NormalDiff_" + filename
    if P_max > P_min:
        filename = "pchange_" + filename
    if D_max > D_min:
        filename = "dchange_" + filename
    if DELTA_max > DELTA_min:
        filename = "dchange_" + filename
    
    filename = "results/result_detailed_OT_" + filename

    # with cProfile.Profile() as pr:
    for n in range(N_min, N_max + 1, N_step):
        for p in range(P_min, P_max + 1, P_step):
            for d in range(D_min, D_max + 1, D_step):
                delta = DELTA_min
                while delta <= DELTA_max:
                    for _ in range(5):
                        A, B, masses_a, masses_b = generate_points(n, d)
                        
                        X = [1/n for _ in range(n)]
                        # X = [x / sum(X) for x in X]
                        X_array = JArray(JDouble)(X)
                        Y = [1/n for _ in range(n)]
                        # Y = [y / sum(Y) for y in Y]
                        Y_array = JArray(JDouble)(Y)

                        time_cost = time.time()
                        C = compute_euclidean_distances(A, B, p)
                        dist = np.array(C.tolist())
                        # print("max dist:", np.max(dist), "min dist:", np.min(dist))
                        time_cost = time.time() - time_cost
                        # print(time_cost)
                        
                        time_our = time.time()
                        rpw_our, java_our = RPW_ICML(n, X_array, Y_array, dist, delta, p=p)
                        time_our = time.time() - time_our
                        # print("Ours time:", time_our)

                        time_lmr = time.time()
                        rpw_lmr, java_LMR = RPW_LMR(n, X_array, Y_array, dist, delta, p=p)
                        time_lmr = time.time() - time_lmr
                        # print("LMR time:", time_lmr)
                        # time_lmr = 0
                        # rpw_lmr = 0
                        # java_LMR = 0
                        
                        # print("ratio: ", time_our / time_lmr)

                        with open(filename, "a") as f:
                            f.write(
                                "\t".join(
                                    [
                                        "n:",
                                        str(n),
                                        "p:",
                                        str(p),
                                        "d:",
                                        str(d),
                                        "delta:",
                                        str(delta),
                                        "LMR-RPW:",
                                        f"{rpw_lmr:.5f}",
                                        "Time LMR:",
                                        f"{time_lmr:.2f}",
                                        "Java Time LMR:",
                                        f"{java_LMR:.2f}",
                                        "OUR-RPW:",
                                        f"{rpw_our:.5f}",
                                        "Time Ours:",
                                        f"{time_our:.2f}",
                                        "Java Time Ours:",
                                        f"{java_our:.2f}",
                                        "Time cost matrix:",
                                        f"{time_cost:.2f}",
                                        "\n",
                                    ]
                                )
                            )

                        del A
                        del B
                        del C
                    delta += DELTA_step
                        # del decompositions
                        # del real_C

        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
