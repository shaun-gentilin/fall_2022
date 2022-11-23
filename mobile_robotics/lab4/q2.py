import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


'''Take in two pointclouds, X and Y, the guess for the optimal rigid 
registration, (t, R), that aligns X to Y, and the maximum admissable
distance, dmax, for associating two points and return a list of estimated
point correspondences.'''
def estimate_correspondences(X, Y, t, R, dmax):
    C = []
    for i, xi in enumerate(X):
        y_min_idx = np.argmin(cdist([R@xi+t], Y))
        if math.dist(R@xi+t, Y[y_min_idx]) < dmax:
            C.append((i, y_min_idx))
    return np.array(C)

'''Optimally aligning pointclouds with known point correspondences'''
def compute_optimal_rigid_registration(X, Y, C):
    x_cent = pointcloud_centroid(X, C, 0)
    y_cent = pointcloud_centroid(Y, C, 1)
    W = compute_cross_covariance_matrix(x_cent, y_cent, X, Y, C)
    u, s, vh = np.linalg.svd(W)
    det_uv = np.linalg.det(u @ vh)
    diag = np.identity(3)
    diag[2][2] = det_uv
    R = u @ diag @ vh
    t = y_cent - R @ x_cent
    return (t, R)

def compute_cross_covariance_matrix(x_cent, y_cent, X, Y, C):
    for k in range(len(C)):
        x_ik = X[C[k][0]]
        x_ik_prime = x_ik - x_cent
        y_jk = Y[C[k][1]]
        y_jk_prime = y_jk - y_cent
        if k == 0:
            W = np.reshape(y_jk_prime, (3,1)) @ np.reshape(x_ik_prime, (1,3))
        else:
            W = W + np.reshape(y_jk_prime, (3,1)) @ np.reshape(x_ik_prime, (1,3))
    return W / len(C)

#given a pointcloud, a list of point correspondences, and a value specifying which
#index in the correspondence to use for the pointcloud, reutrn the centroid of the
#pointcloud
def pointcloud_centroid(PC, C, idx):
    for k in range(len(C)):
        if k == 0:
            sum = PC[C[k][idx]]
        else:
            sum = sum + PC[C[k][idx]]
    return sum / len(C)

def iterative_closest_point(X, Y, t0, R0, dmax, num_ICP_iters):
    t = t0
    R = R0
    for i in range(num_ICP_iters):
        C = estimate_correspondences(X, Y, t, R, dmax)
        t, R = compute_optimal_rigid_registration(X, Y, C)
    return (t, R, C)

    
def read_pointcloud_from_file(filename):
    with open(filename) as file:
        lines = file.readlines()
    PC = np.empty((len(lines),3))
    for i in range(len(lines)):
        line = lines[i].split()
        for j in range(len(line)):
            PC[i][j] = float(line[j])
    return PC

def root_mean_squared_error(X, Y, t, R, C):
    sum = 0
    for k in range(len(C)):
        i = C[k][0]
        j = C[k][1]
        sum += math.dist(Y[j], R @ X[i] + t)
    return math.sqrt(sum / len(C))

t0 = np.array([0,0,0])
R0 = np.identity(3)
dmax = 0.25
num_icp_iters = 30
X = read_pointcloud_from_file('/home/shaun/college/fall_2022/mobile_robotics/pclX.txt')
Y = read_pointcloud_from_file('/home/shaun/college/fall_2022/mobile_robotics/pclY.txt')
t, R, C = iterative_closest_point(X, Y, t0, R0, dmax, num_icp_iters)
print(f'Position: {t}\nRotation: {R}\nCorrespondences: {C}\n')
RMSE = root_mean_squared_error(X, Y, t, R, C)
print(f'RMSE: {RMSE}\n')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

x_Y = [y[0] for y in Y]
y_Y = [y[1] for y in Y]
z_Y = [y[2] for y in Y]

X_transform = [R @ x + t for x in X]
x_X_transform = [x[0] for x in X_transform]
y_X_transform = [x[1] for x in X_transform]
z_X_transform = [x[2] for x in X_transform]

x_X = [x[0] for x in X]
y_X = [x[1] for x in X]
z_X = [x[2] for x in X]

ax.scatter(x_Y, y_Y, z_Y, color='green')
ax.scatter(x_X_transform, y_X_transform, z_X_transform, color='red')
ax.scatter(x_X, y_X, z_X, color='blue')

plt.show()