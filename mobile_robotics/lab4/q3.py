#################################### Imports ####################################

import random
import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

################################### Functions ###################################

'''Takes in the current time (t1), the particle set describing the robots belief
over its position at time t1 (X_t1), the commanded wheel speeds for the left 
and right wheels (phi_l and phi_r), the time at which to predict the next
pose (t2), the wheel radius (r), the track width (w), and the variances of
the wheel speeds (sig_l and sig_r).  Returns the particle set describing the
robots belief over its pose the time t2 (X_t2).'''
def particle_filter_propagate(t_1, X_t_1, phi_l, phi_r, t_2, r, w, sig_l, sig_r):
    X_t_2 = np.empty(X_t_1.shape)
    for i, x_t_1 in enumerate(X_t_1):
        ep_l = np.random.normal(0, sig_l)
        ep_r = np.random.normal(0, sig_r)
        omega = create_omega(phi_l + ep_l, phi_r + ep_r, r, w)
        x_t_2 = x_t_1 @ expm((t_2-t_1) * omega)
        X_t_2[i] = x_t_2
    return X_t_2

'''Creates the matrix omega which represents a mapping of wheel speeds to the
robots velocity in the lie group SE(2) at I.'''
def create_omega(phi_l, phi_r, r, w):
    omega = np.zeros((3,3))
    omega[0][1] = -r/w * (phi_r - phi_l)
    omega[0][2] = r/2 * (phi_r + phi_l)
    omega[1][0] = r/w * (phi_r - phi_l)
    return omega

'''Takes in the particle set representing its prior belief over its pose (X_t),
a noisy position measurement (z_t), and a magnitude of the measurement noise 
(sig_p) and returns the particle set (X_bar_t) modeling the robots posterior belief
after incorporating the measurement z_t.'''
def particle_filter_update(X_t, z_t, sig_p):
    W = calculate_importance_weights(X_t, z_t, sig_p)
    return importance_weighted_resampling(W, X_t)

def importance_weighted_resampling(W, X_t):
    X_bar_t = np.empty(X_t.shape)
    for i in range(len(X_t)):
        x_bar_t = np.random.choice(X_t, p=W)
        X_bar_t[i] = x_bar_t
    return X_bar_t

def calculate_importance_weights(X_t, z_t, sig_p):
    W = np.empty((len(X_t), 1))
    for i, x_i in enumerate(X_t):
        w = multivariate_normal.pdf(z_t, x_i[0], sig_p**2 @ np.identity(2))
        W[i] = w
    return W

def sample_initial_particle_set(l_0, R_0, N):
    x_0 = homogenize_2D_pose(l_0, R_0)
    X_0 = []
    for i in range(N):
        X_0.append(x_0)
    return np.array(X_0)

def homogenize_2D_pose(t, R):
    h_pose = [[R[0][0], R[0][1], t[0]],
              [R[1][0], R[1][1], t[1]],
              [0,       0,       1]]
    return np.array(h_pose)

def extract_positions(X_t):
    xs, ys = [], []
    for pose in X_t:
        xs.append(pose[0][2])
        ys.append(pose[1][2])
    return xs, ys  

'''Apply particle filter propagation at the given times given the parameters
for the filter propagation.'''
def propagate_and_report(times, t_1, X_t_1, phi_l, phi_r, r, w, sig_l, sig_r):
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    fig = plt.figure()
    ax = fig.add_subplot()
    for t in times:
        X_t = particle_filter_propagate(t_1, X_t_1, phi_l, phi_r, t, r, w, sig_l, sig_r)
        xs, ys = extract_positions(X_t)
        print(f'Starting Time: {t_1}\nEnding Time: {t}\nEmpirical mean: ({np.mean(xs)}, {np.mean(ys)})\nCovariance: {np.cov(xs, ys)}\n')
        color = random.choice(colors)
        colors.remove(color)
        ax.scatter(x=xs, y=ys, color=color)
    ax.set_title('Particle Filter Propagation at time t=10 (1000 samples)')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_aspect('equal')
    plt.show()


##################################### Script #####################################

phi_l = 1.5
phi_r = 2
r = 0.25
w = 0.5
sig_l = 0.05
sig_r = 0.05
sig_p = 0.1
N = 1000
l_0 = np.array([0, 0])
R_0 = np.identity(2)
X_0 = sample_initial_particle_set(l_0, R_0, N)

##################################### Part E #####################################

propagate_and_report([10], 0, X_0, phi_l, phi_r, r, w, sig_l, sig_r)

##################################### Part F #####################################

times = [5, 10, 15, 20]
propagate_and_report(times, 0, X_0, phi_l, phi_r, r, w, sig_l, sig_r)