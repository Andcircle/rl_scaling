import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_sequence(hypes, seqence):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
    fig.subplots_adjust(wspace=0.1)
    fig.suptitle(hypes["name"])

    ax[0,0].plot(seqence.request_history, color='red', label='Request')
    ax[0,0].plot(seqence.capacity_history, color='blue', label='Capacity')
    ax[0,0].legend()
    ax[0,0].set_title("Capcity")

    ax[0,1].plot(seqence.queue_history, color='green', label='Queue')
    ax[0,1].plot(seqence.capacity_gap_history, color='gold', label='Gap')
    ax[0,1].legend()
    ax[0,1].set_title("Queue")

    ax[1,0].plot(seqence.load_ratio_history, color='black', label='Load')
    ax[1,0].legend()
    ax[1,0].set_title("Load")

    plt.show()
