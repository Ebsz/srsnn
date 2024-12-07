#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import sys
import json

from parse import parse_array1, parse_array2

from network import Network


def plot_3d(arr):
    ax = plt.figure().add_subplot(projection='3d')

    #ax.axis('off')

    T = arr.shape[0]
    print(f"T: {T}")

    #for i in range(0, T-100):
    #    ax.plot3D(arr[i:i+1][0], arr[i:+1][1], arr[i:+1][2], lw=0.5, color=plt.cm.viridis(i/T))

    ax.plot(*arr.T, lw=0.5, color='black')
    #ax.plot(*arr.T, lw=0.5, color='black')

    #ax.plot3D(*arr.T, lw=0.5, color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")


    plt.show()

def plot_system(arr):
    x = arr[:,0]
    y = arr[:,1]
    z = arr[:,2]

    print(y.shape)

    fig, (ax1, ax2,ax3) = plt.subplots(3,1)

    ax1.plot(x)
    ax1.set_ylabel('X')
    ax2.plot(y)
    ax2.set_ylabel('Y')
    ax3.plot(z)
    ax3.set_ylabel('Z')

    plt.show()

def plot_vec(arr):
    plt.plot(arr)
    plt.show()


def plot_time_series(d):
    observed = parse_array1(d[0])
    predicted = parse_array1(d[1])
    deviation = parse_array1(d[2])

    fig, (ax1, ax2,ax3) = plt.subplots(3,1)

    ax1.plot(observed)
    ax1.set_ylabel('Observed')
    ax2.plot(predicted)
    ax2.set_ylabel('Predicted')
    ax3.plot(deviation)
    ax3.set_ylabel('Deviation')

    plt.show()


def pretty_print(x):
    print(json.dumps(x, sort_keys=True, indent=4))

def args():
    if len(sys.argv) < 2:
        print("usage: [FILE.json]\n\n")
        assert(1+1 != 2)

    return sys.argv[1:]


def plot_neuron_models(d):
    series = [parse_array1(x) for x in d]

    fig, ax  = plt.subplots(1, 3, layout='constrained')

    fig.set_figheight(1.7)
    fig.set_figwidth(10)
    #fig.figsize((3,10))

    C = 220
    #C = 1000


    # Lapicque
    #for i in range(3):
    #    a = ax[i]
    #    a.get_xaxis().set_visible(False)
    #    a.get_yaxis().set_visible(False)
    #    ##ax[i].set_title(chr(97+i) + ")", loc='left')
    #    a.set_title("(" + chr(97+i) + ")")

    #    a.plot(series[i][:C], color="black")
    #    a.get_xaxis().set_visible(True)
    #    a.set_xlabel("Time")

    # Izhikevich
    for i in range(3):
        a = ax[i]
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        ##ax[i].set_title(chr(97+i) + ")", loc='left')
        a.set_title("(" + chr(97+i) + ")")

        a.plot(series[i+3][:C], color="black")
        a.get_xaxis().set_visible(True)
        a.set_xlabel("Time")

    #for c in range(2):
    #    for r in range(3):
    #        print((c,r))

    #        a = ax[r][c]

    #        i = r + c * 3

    #        a.get_xaxis().set_visible(False)
    #        a.get_yaxis().set_visible(False)
    #        ##ax[i].set_title(chr(97+i) + ")", loc='left')
    #        a.set_title("(" + chr(97+i) + ")")

    #        a.plot(series[i][:C], color="black")

    #    ax[-1][c].get_xaxis().set_visible(True)
    #    ax[-1][c].set_xlabel("Time")


    filename = "neuron_plot.png"
    plt.savefig(filename)
    print(f"Saved to {filename}")

def run():
    #files = ["results/dynamics/neurons/lapicque_r10.0-c0.001.json",
    #         "results/dynamics/neurons/lapicque_r5.0-c0.001.json",
    #         "results/dynamics/neurons/lapicque_r5.0-c0.005.json",
    #         "results/dynamics/neurons/izhikevich_rs.json",
    #         "results/dynamics/neurons/izhikevich_fs.json",
    #         "results/dynamics/neurons/izhikevich_ch.json"]
    files = args()


    print("loading data")
    d = []
    for path in files:
        with open(path) as f:
            d.append(json.load(f))

    n = len(d)
    print(f"loaded {n} files")

    plot_time_series(d)



    #plot_neuron_models(d)

#    series = [parse_array1(x) for x in d]
#
#    if len(series) == 1:
#        fig, ax  = plt.subplots(n, 1, layout='constrained')
#        ax.get_yaxis().set_visible(False)
#        ax.set_xlabel("Time")
#        ax.plot(series[0][:C], color="black")
#
#    else:
#        #fig, ax  = plt.subplots(n, 1, layout='constrained')
#        fig, ax  = plt.subplots(n, 1)
#
#        for i in range(n):
#            print(i)
#            ax[i].get_xaxis().set_visible(False)
#            ax[i].get_yaxis().set_ticks([])
#            #ax[i].set_title(chr(97+i) + ")", loc='left')
#            ax[i].set_title(chr(97+i))
#
#            ax[i].plot(series[i][:C], color="black")
#
#        ax[-1].get_xaxis().set_visible(True)
#        ax[-1].set_xlabel("Time")
#
#
#    filename = "neuron_plot.png"
#    plt.savefig(filename)
#    print(f"Saved to {filename}")


    #network = Network(d[0])

    #print(network)

    #plot_time_series(d)

    #for i in range(len(arr)):
    #    print(arr[i])

    #spike_times = [[t for t,s in enumerate(i) if s] for i in arr.T]

    #fig, ax = plt.subplots()
    #ax.eventplot(spike_times)
    #ax.set_xlim((0, len(arr)))
    #for i in range(len(spikes)):
    #    ax.vlines(spikes[i], i-0.5,i+0.5, color="black")



    #plot_single_value(arr)
    #plot_system(arr)
    #plot_3d(arr)

    #for (x, y, z) in array:
    #    print(f"{x}\t {y}\t {z}")

    #print(array.shape)


    #pretty_print(data)
    #print(len(data))

if __name__ == "__main__":
    run()
