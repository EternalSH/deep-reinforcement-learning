# Summary

I am not sure what should I try to solve this project. The parameters I have already choose do not give any promising results.

## Learning Algorithm


## Failed attempts

### Attempts with max\_t = 500 or max\t_ = 2000, no early break (not stopping episode after first _done_)

#### Actor

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
3. Output layer
  - No. of units: 2 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

#### Training parameters
1. Actor learning rate: 1e-3
2. Critic learning rate: 1e-3

#### Reinforcement Learning Parameters

1. _n\_episodes_: 20
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 500
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE\_EPSILON_: 5.0, with decay of 1e-3 (_NOISE\_EPSILON\_DECAY_) and minimal value of 0.0
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG\_LEARN\_TIMES_: 4
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG\_UPDATE\_EVERY_: 1
  - The Actor / Critic pair update is not performed after every step, but each _DDPG\_UPDATE\_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.

--- 

#### Actor

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
3. Output layer
  - No. of units: 2 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

#### Training parameters
1. Actor learning rate: 1e-3
2. Critic learning rate: 1e-3

#### Reinforcement Learning Parameters

1. _n\_episodes_: 20
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 500
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE\_EPSILON_: 5.0, with decay of 1e-3 (_NOISE\_EPSILON\_DECAY_) and minimal value of 0.0
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG\_LEARN\_TIMES_: 4
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG\_UPDATE\_EVERY_: 1
  - The Actor / Critic pair update is not performed after every step, but each _DDPG\_UPDATE\_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.
6. _BATCH\_SIZE_: 512
  - Minibatch size used for training neural networks.

---

#### Actor

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
  - Dropout, p = 0.1
3. Output layer
  - No. of units: 2 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
  - Dropout, p = 0.05
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

#### Training parameters
1. Actor learning rate: 1e-3
2. Critic learning rate: 3e-3

#### Reinforcement Learning Parameters

1. _n\_episodes_: 20
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 500
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE\_EPSILON_: 5.0, with decay of 1e-3 (_NOISE\_EPSILON\_DECAY_) and minimal value of 0.0
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG\_LEARN\_TIMES_: 4
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG\_UPDATE\_EVERY_: 1
  - The Actor / Critic pair update is not performed after every step, but each _DDPG\_UPDATE\_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.

---

#### Actor

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
3. Output layer
  - No. of units: 2 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: 512
  - Activation function: ReLU
  - Batch normalization
2. Second hidden layer
  - No. of units: 256
  - Activation function: ReLU
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

#### Training parameters
1. Actor learning rate: 1e-4
2. Critic learning rate: 3e-4

#### Reinforcement Learning Parameters

1. _n\_episodes_: 20
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 500
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE\_EPSILON_: 5.0, with decay of 1e-3 (_NOISE\_EPSILON\_DECAY_) and minimal value of 0.0
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG\_LEARN\_TIMES_: 4
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG\_UPDATE\_EVERY_: 1
  - The Actor / Critic pair update is not performed after every step, but each _DDPG\_UPDATE\_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.

---

Model performance: oscilating with values under 0.0. No progress in training.

### Ideas also checked:
1. Change _BUFFER\_SIZE_ - check 1e5, 1e6 i 1e8 - no change, apart from the training time
2. Remove dropout or batch normalization - no change

### Experience Replay

To be able to use all events ecountered during the training (not only ones from the last episode) Experience Replay was implemented. It is a buffer containing ("state", "action", "reward", "next_state", "done") tuples. The buffer is latter used to train the neural network. This improves training, as the NN inside the network has access to events not seen in the current episode.
The experience buffer is common and shared for both actors.

### Neural Network Parameters



## Ideas for Further Work

## Project files
