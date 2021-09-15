import numpy as np
import tensorflow as tf
import logging

from enviroments.environment import Environment
from agent_algorithms.a2c import ActorCritic
from simulator import Simulator

from utils import logger, yaml_loader, visualizer

def train(hypes, tf_writer):

    epoch = hypes['agent']['train']['epoch']
    step = hypes['agent']['train']['step']
    batch = hypes['agent']['train']['batch']
    gamma = hypes['agent']['value']['gamma']


    env = Environment(hypes['environment'])
    agent = ActorCritic(env.get_obs_shape(), hypes['agent'])

    sim_step = step * batch
    sim = Simulator(agent, env, gamma)

    for n in range(epoch):
        do_log = n % hypes['agent']['train']['log_epoch'] == 0
        do_evaluate = n % hypes['agent']['train']['evaluate_epoch'] == 0
        do_save = n % hypes['agent']['train']['save_epoch'] == 0

        actions, values, observations, rewards, dones, states = sim.simulate(sim_step, do_evaluate)
        avg_reward = np.mean(rewards)

        inds = np.arange(sim_step)
        np.random.shuffle(inds)

        pg_losses=[]
        vf_losses=[]
        entropies=[]

        for m in range(step):
            batch_inds = inds[m * batch : m * batch + batch]
            batch_slice = (tf.constant(arr[batch_inds]) for arr in (observations, actions, rewards, values))
            pg_loss, vf_loss, entropy = agent.train(*batch_slice)

            pg_losses.append(pg_loss)
            vf_losses.append(vf_loss)
            entropies.append(entropy)

        pg_loss = np.mean(pg_losses)
        vf_loss = np.mean(vf_losses)
        entropy = np.mean(entropies)

        if do_log:
            with tf_writer.as_default():
                tf.summary.scalar("pg_loss", pg_loss, step=n)
                tf.summary.scalar("vf_loss", vf_loss, step=n)
                tf.summary.scalar("entropy", entropy, step=n)
            tf_writer.flush()

            summary = 'Epoch {}: avg reward: {}, pg_loss: {}, vf_loss: {}, entropy: {}'.format(n, avg_reward, pg_loss, vf_loss, entropy)
            logging.info(summary)

        if do_evaluate:
            evaluate(tf_writer, n, actions[0:360], values[0:360], observations[0:360], rewards[0:360], dones[0:360], states[0:360])

        if do_save:
            agent.save_weights(hypes['output_dir'], n)


def evaluate(tf_writer, epoch, actions, values, observations, rewards, dones, states):
    images = visualizer.plot_to_images(actions, values, observations, rewards, dones, states)
    
    with tf_writer.as_default():
        for k, v in images.items():
            tf.summary.image("simulator/{}".format(k), np.expand_dims(v, axis=0), step=epoch, max_outputs=1)
    tf_writer.flush()


def test(hypes, step=0, sim=None):
    if sim is None:
        # can test with different environment, differnt agent, e.g. random
        env = Environment(hypes['environment'])
        agent = ActorCritic(env.get_obs_shape(), hypes['agent'])
        gamma = hypes['agent']['value']['gamma']
        sim = Simulator(agent, env, gamma)

    if step == 0:
        step = hypes['agent']['test']['step']

    actions, values, observations, rewards, dones, states = sim.simulate(step, True)
    visualizer.plot_to_fig(actions, values, observations, rewards, dones, states)
        

if __name__ == "__main__":
    hypes = yaml_loader.load_hypes("hyper_parameters/hypes.yaml")
    tf_writer = tf.summary.create_file_writer(hypes['output_dir'])
    logger.init_logging(hypes['output_dir'])

    train(hypes, tf_writer)
    # test(hypes, 360)