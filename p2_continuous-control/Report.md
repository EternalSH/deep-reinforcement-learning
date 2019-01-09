# Summary

This is implementation of my DDPG agent for reacher problem. The implemetation is in Continous_Control_Solution.ipynb_ file (Jupyter Notebook)

## Learning Algorithm

The algorithm that I have used is simple DDPG, as in example suggested in the problem description. The biggest change that I have introduced is adding noise decay - the more episodes have passed, the noise is becoming weaker (its value is multiplied by epsilon, the epsilon decays to zero).

Overall, it is an Actor - Critic solution with both created as simple, two-layered neural networks.

### Neural Network Parameters

#### Actor

1. First hidden layer
  - No. of units: 256
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 128
  - Activation function: ReLU
3. Output layer
  - No. of units: 4 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: 256
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 128
  - Activation function: ReLU
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

### Reinforcement Learning Parameters

1. _n\_episodes_: 500
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 1000
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE_EPSILON_: 1.00, with decay of 1e-6 (_NOISE_EPSILON_DECAY_)
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG_LEARN_TIMES_: 10
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG_UPDATE_EVERY_: 20
  - The Actor / Critic pair update is not performed after every step, but each _DDPG_UPDATE_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.

### Experience Replay

To be able to use all events ecountered during the training (not only ones from the last episode) Experience Replay was implemented. It is a buffer containing ("state", "action", "reward", "next_state", "done") tuples. The buffer is latter used to train the neural network. This improves training, as the NN inside the network has access to events not seen in the current episode.

## Ideas for Further Work

I would like to try following ideas to improve the solution:

- Use prioritized replay buffer.
- Implement D4PG algoritm to improve the result.

## Project files
- Navigation.ipynb - notebook with DQN agent implementation
- README.md - instructions for running the notebook
- Report.md - this file (report from the project)
- checkpoint.pth - checkpoint file with trained DQN agent