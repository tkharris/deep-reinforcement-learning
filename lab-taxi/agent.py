import numpy as np
from collections import defaultdict
import gym

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.taxi_env = gym.make('Taxi-v2')
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0
        self.epsilon_min = 0.00001
        self.alpha = 0.75
        self.gamma = 0.999
        self.num_eps = 0
        print("epsilon: {}, epsilon_min: {}, alpha: {}, gamma: {}".format(self.epsilon, self.epsilon_min, self.alpha, self.gamma))

    def _squash_state(self, state):
        row, col, pass_loc, dest_loc = self.taxi_env.decode(state)
        if pass_loc != 4:
            #We haven't picked the passenger up
            #Squash destination state space
            return self.taxi_env.encode(row, col, pass_loc, 0)
        return self.taxi_env.encode(row, col, dest_loc, 1)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        state = self._squash_state(state)
        best = np.argmax(self.Q[state])
        self.probs = np.full(self.nA, self.epsilon/self.nA)
        self.probs[best] += 1 - self.epsilon
        return np.random.choice(np.arange(self.nA), p=self.probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        state = self._squash_state(state)
        next_state = self._squash_state(next_state)

        G = reward + self.gamma*np.dot(self.Q[next_state], self.probs)
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*G
        if done: 
            self.num_eps += 1
            self.epsilon = max(self.epsilon_min, 1.0/self.num_eps)
