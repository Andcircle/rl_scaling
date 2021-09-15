import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import os

import importlib

class ActorCritic():
    
    def __init__(self, input_shape, hypes):
        self.hypes = hypes
        self.input_shapt = input_shape

        actor_network = importlib.import_module('networks.' + hypes['actor']['network'])
        self.actor_net = actor_network.build(input_shape, hypes['actor']['network_params'], 'actor')
        print("Actor Network")
        self.actor_net.summary()

        critic_network = importlib.import_module('networks.' + hypes['critic']['network'])
        self.critic_net = critic_network.build(input_shape, hypes['critic']['network_params'], 'critic')
        print("Critic Network")
        self.critic_net.summary()

        if hypes['retrain']:
            self.load_weights()

        self.ent_coef=hypes['actor']['ent_coef']
        self.max_grad_norm=hypes['actor']['max_grad_norm']

        # lr_schedule = InverseLinearTimeDecay(initial_learning_rate=lr, nupdates=nupdates)
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=alpha, epsilon=epsilon)
        
        self.actor_optimizer = getattr(tf.keras.optimizers, hypes['actor']['solver'])\
            (**hypes['actor']['solver_params'])
        self.critic_optimizer = getattr(tf.keras.optimizers, hypes['critic']['solver'])\
            (**hypes['critic']['solver_params'])

        
    def __action_head(self, ret):
        pd = tf.keras.layers.Dense(units=2, name='action_head')(ret)
        mean, std = tf.split(axis=1, num_or_size_splits=2, value=pd)
        std = tf.math.softplus(std) + 1e-5
        mean = tf.squeeze(mean, axis=1)
        std = tf.squeeze(std, axis=1)
        return mean, std

    def __value_head(self, ret):
        return tf.keras.layers.Dense(units=1, name='value_head')(ret)


    def train(self, observations, actions, rewards, values):

        with tf.GradientTape() as actor_tape:
            rets = self.actor_net(observations)
            mean, std = self.__action_head(rets)
            dists = tfd.Normal(mean, std)
            # prob = dists.prob(actions)
            log_prob = dists.log_prob(actions)
            # log_prob = tf.math.log(prob + 1e-5)
            advs = rewards - values
            pg_loss = -tf.reduce_mean(log_prob * advs)
            entropy = tf.reduce_mean(dists.entropy())
            loss = pg_loss - entropy * self.ent_coef

        with tf.GradientTape() as critic_tape:
            vpred = self.__value_head(rets)
            vf_loss = tf.reduce_mean(tf.square(vpred - rewards))
            entropy = tf.reduce_mean(dists.entropy())

        actor_var_list = actor_tape.watched_variables()
        actor_grads = actor_tape.gradient(loss, actor_var_list)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        actor_grads_and_vars = list(zip(actor_grads, actor_var_list))
        self.actor_optimizer.apply_gradients(actor_grads_and_vars)

        critic_var_list = critic_tape.watched_variables()
        critic_grads = critic_tape.gradient(vf_loss, critic_var_list)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        critic_grads_and_vars = list(zip(critic_grads, critic_var_list))
        self.critic_optimizer.apply_gradients(critic_grads_and_vars)

        return pg_loss, vf_loss, entropy

    
    def step(self, observation):
        actor_ret = self.actor_net(observation)
        mean, std = self.__action_head(actor_ret)
        dist = tfd.Normal(mean, std)
        action = dist.sample()
        action = tf.clip_by_value(action, 0., 1.)
        critic_ret = self.critic_net(observation)
        value = self.__value_head(critic_ret)

        # step is called by simulation, no need batch
        return action[0], value[0,0]


    def save_weights(self, dir, step):
        actor_save_path = os.path.join(dir, 'actor_net/actor_{}.ckpt'.format(step))
        self.actor_net.save_weights(actor_save_path)

        critic_save_path = os.path.join(dir, 'critic_net/critic_{}.ckpt'.format(step))
        self.critic_net.save_weights(critic_save_path)


    def load_weights(self, dir):
        actor_load_dir = os.path.join(dir, 'actor_net')
        latest_actor_path = tf.train.latest_checkpoint(actor_load_dir)
        self.actor_net.load_weights(latest_actor_path)

        critic_load_dir = os.path.join(dir, 'critic_net')
        latest_critic_path = tf.train.latest_checkpoint(critic_load_dir)
        self.critic_net.load_weights(latest_critic_path)


