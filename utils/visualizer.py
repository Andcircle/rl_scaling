import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

colors = ['blue', 'red', 'green', 'black', 'gold', 'yellow']


def plot_to_images(actions, values, observations, rewards, dones, states):
    sequence = convert_to_sequence(actions, values, observations, rewards, dones, states)
    return plot_sequence_to_images(sequence)


def plot_to_fig(actions, values, observations, rewards, dones, states):
    sequence = convert_to_sequence(actions, values, observations, rewards, dones, states)
    plot_sequence_to_fig(sequence)


def convert_to_sequence(actions, values, obss, rewards, dones, states):

    sequence = dict()
    # sequence["value"] = []
    # sequence["reward"] = []

    # sequence["total_cost"] = []
    # sequence["capacity_gap"] = []
    # sequence["instance_num"] = []
    # sequence["action"] = []

    # sequence["total_capacity"] = []
    # sequence["request_num"] = []
    # sequence["request_queue_size"] = []

    # sequence["capacity_ratio"] = []
    # sequence["load_ratio"] = []

    # sequence["done"] = []

    for key in states[0].keys():
        sequence[key] = [ state[key] for state in states ]

    sequence['value'] = [ value for value in values ] 
    sequence['reward'] = [ reward for reward in rewards ]
    sequence['done'] = [ int(done) for done in dones ]
    
    return sequence


def plot_sequence_to_fig(sequence):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,9))
    fig.subplots_adjust(wspace=0.1)
    fig.suptitle("Simulator Test")

    ax[0,0].plot(sequence['request_num'], color='red', label='Request')
    ax[0,0].plot(sequence['total_capacity'], color='blue', label='Capacity')
    ax[0,0].legend()
    ax[0,0].set_title("Capcity")

    ax[0,1].plot(sequence['request_queue_size'], color='green', label='Queue')
    ax[0,1].plot(sequence['capacity_gap'], color='gold', label='Gap')
    ax[0,1].legend()
    ax[0,1].set_title("Queue")

    ax[0,2].plot(sequence['load_ratio'], color='black', label='Load Ratio')
    ax[0,2].plot(sequence['capacity_ratio'], color='yellow', label='Capacity Ratio')
    ax[0,2].legend()
    ax[0,2].set_title("Load")

    ax[1,0].plot(sequence['action'], color='green', label='Action')
    ax[1,0].plot(sequence['instance_num'], color='gold', label='Instance')
    ax[1,0].legend()
    ax[1,0].set_title("Action")

    ax[1,1].plot(sequence['value'], color='red', label='Value')
    ax[1,1].plot(sequence['reward'], color='blue', label='Reward')
    ax[1,1].legend()
    ax[1,1].set_title("Value")

    ax[1,2].plot(sequence['done'], color='red', label='Done')
    ax[1,2].legend()
    ax[1,2].set_title("Done")

    plt.show()


def plot_sequence_to_images(sequence):
    images=dict()
    
    title = 'Capacity'
    image = plot_list(title, request=sequence['request_num'], 
                      capacity=sequence['total_capacity'])
    images[title]=image

    title = 'Queue'
    image = plot_list(title, queue=sequence['request_queue_size'], 
                      gap=sequence['capacity_gap'])
    images[title]=image

    title = 'Load'
    image = plot_list(title, load_ratio=sequence['load_ratio'], 
                      capacity_ratio=sequence['capacity_ratio'])
    images[title]=image

    title = 'Value'
    image = plot_list(title, value=sequence['value'], 
                      reward=sequence['reward'])
    images[title]=image

    title = 'Action'
    image = plot_list(title, value=sequence['action'], 
                      reward=sequence['instance_num'])
    images[title]=image

    title = 'Done'
    image = plot_list(title, value=sequence['done'])
    images[title]=image
    
    return images



def plot_list(title, **kwargs):
    fig = plt.gcf()
    ax = plt.gca()
    
    for n, (k, v) in enumerate(kwargs.items()):
        plt.plot(v, color=colors[n], label=k)

    ax.legend()
    ax.set_title(title)

    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.close(fig)
    return image

