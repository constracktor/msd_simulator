import src.n_linked_msd as msd

import numpy as np
#import os
import matplotlib
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # Number of carts
    n_carts = 4
    # Number samples
    n_samples = 10000
    # Sampling frequency
    sampling_frequency = 0.05
    # Measurement noise standard deviation
    noise = 0.05
    # random seed
    seed = 0
    # initialize model
    msd_model = msd.msd_chain(N=n_carts, noise_sd=noise)
    # generate input
    #u = msd_model.aprbs_signal(T=n_samples, amplitude=2.0, period=300, random_seed=seed)
    u = msd_model.multisine_signal(T= n_samples, amplitude=4.0, period=5, random_seed=None)
    # simulate output
    y = msd_model.simulate(u, f_sample=sampling_frequency)
    # store data
    file_u = open("./data/data_u.txt", "w")
    for value in u:
        file_u.write(str(value) + "\n")
    file_u.close()
    file_y = open("./data/data_y.txt", "w")
    for value in y:
        file_y.write(str(value) + "\n")
    file_y.close()
    # create figures for visualization
    time = np.linspace(0, sampling_frequency*n_samples, n_samples)
    plt.figure(figsize=(10,4))
    plt.plot(time, u, 'k', label='u', linewidth=1)
    plt.savefig('./plot/plot_u.pdf', bbox_inches='tight')

    plt.figure(figsize=(10,4))
    plt.plot(time, y, 'k', label='y', linewidth=1)
    plt.savefig('./plot/plot_y.pdf', bbox_inches='tight')

    print("return 0")
