import environment
import sequence
import visualizer

from utils import logger
from utils import yaml_loader


def simulate():
    hypes = yaml_loader.load_hypes("./hyper_parameters/hypes.yaml")
    env = environment.Environment(hypes)

    seq = sequence.Sequence()

    for i in range(1000):
        state, action, reward, next_state, _= env.step()
        seq.update(state, action, reward)
    
    visualizer.plot_sequence(hypes,seq)


if __name__ == "__main__":
    simulate()