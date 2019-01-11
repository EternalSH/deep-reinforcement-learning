# Summary

I am not sure what should I try to solve this project. The parameters I have already choose do not give any promising results.

## Learning Algorithm



### Neural Network Parameters

#### Actor

1. First hidden layer
  - No. of units: ??
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: ??
  - Activation function: ReLU
3. Output layer
  - No. of units: 4 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: ??
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: ??
  - Activation function: ReLU
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

### Reinforcement Learning Parameters

1. _n\_episodes_: ???
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: ???
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE_EPSILON_: ???, with decay of ??? (_NOISE_EPSILON_DECAY_) and minimal value of ???
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG_LEARN_TIMES_: ??
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG_UPDATE_EVERY_: ??
  - The Actor / Critic pair update is not performed after every step, but each _DDPG_UPDATE_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.

### Experience Replay

To be able to use all events ecountered during the training (not only ones from the last episode) Experience Replay was implemented. It is a buffer containing ("state", "action", "reward", "next_state", "done") tuples. The buffer is latter used to train the neural network. This improves training, as the NN inside the network has access to events not seen in the current episode.

## Ideas for Further Work

## Project files
