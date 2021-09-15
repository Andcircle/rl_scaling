import numpy as np

class Simulator():

    def __init__(self, agent, environment, gamma):
        self.agent = agent
        self.environment = environment
        self.gamma = gamma

    def simulate(self, step, test=False):
        
        _, state, obs, _ = self.environment.reset()

        actions = []
        values = []
        obss = []
        rewards = []
        dones = []

        states = []

        for i in range(step):
            batch_obs = np.expand_dims(obs, axis=0)
            action, value = self.agent.step(batch_obs)

            actions.append(action)
            values.append(value)
            obss.append(obs)

            if test:
                states.append(state)

            reward, state, obs, done = self.environment.step(action)

            rewards.append(reward)
            dones.append(done)

            if done:
                _, state, obs, _ = self.environment.reset()

        actions = np.asarray(actions, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        obss = np.asarray(obss, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool)

        # Method 1 use true Q-value, which may lead to diverge because of sequence length
        # reward = 0
        # if dones[-1] == 0:
        #     # if the eposide is not done, initialize reward with estimiate value function
        #     batch_obs = np.expand_dims(obs, axis=0)
        #     _, value = self.agent.step(batch_obs)
        #     reward = value

        # for n in reversed(range(step)):
        #     reward = rewards[n] + reward * self.gamma * (1 - dones[n])
        #     rewards[n] = reward

        # Method 2 only use 1 step reward + V-value, and done is always false
        batch_obs = np.expand_dims(obs, axis=0)
        _, value = self.agent.step(batch_obs)

        for n in reversed(range(step)):
            reward = rewards[n] + value * self.gamma
            rewards[n] = reward
            value = values[n]
        
        return actions, values, obss, rewards, dones, states
        # visualizer.plot_sequence(hypes,seq)


# if __name__ == "__main__":
#     simulate()