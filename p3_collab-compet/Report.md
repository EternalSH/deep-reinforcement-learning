# Summary

This is implementation of my DDPG agent for reacher problem. The implemetation is in _Tennis-Solved.ipynb_ file (Jupyter Notebook).

## Learning Algorithm

My solution uses version of the MADDPG algorithm. The agents use separate replay buffers, but each one has knowledge about other's actions actions and uses it in training. Also, as in my implementation of Project 2, for this implementation noise decay is present - the more episodes have passed, the noise is becoming weaker (its value is multiplied by epsilon, the epsilon decays to zero). It strongly encourages agents to explore the action space for c.a. 300 episodes before using their policy. 

Overall, it is an Actor - Critic solution with both created as simple, two-layered neural networks and not shared replay buffer.

## Best model

#### Actor

1. First hidden layer
  - No. of units: 256
  - Activation function: ReLU
2. Second hidden layer
  - No. of units: 128
  - Activation function: ReLU
3. Output layer
  - No. of units: 2 (equal to actions dimension)
  - Activation function: tanh

#### Critic

1. First hidden layer
  - No. of units: 256
  - Activation function: ReLU
2. Second hidden layer
  - No. of units: 128
  - Activation function: ReLU
3. Output layer
  - No. of units: 1 (calculates the action value function - a scalar)
  - Activation function: none

#### Training parameters
1. Actor learning rate: 0.001
2. Critic learning rate: 0.001

#### Reinforcement Learning Parameters

1. _n\_episodes_: 3000
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 2000
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree. The chosen implementation does not stop after episode is done.
3. _NOISE\_EPSILON_: 5.5, with decay of 0.0006 (_NOISE\_EPSILON\_DECAY_) and minimal value of 0.0004
  - The _eps_ noise multiplier for the action values. It is controling exploration/exploitation ratio. It is decaying to zero, meaning that the algorithm starts with strong preference to exploration, but goes into exploitation with more and more learning sessions.
4. _DDPG\_LEARN\_TIMES_: 1
  - After each step, if learning session is performed, it is repeated multiple times. The number is mentioned above.
5. _DDPG\_UPDATE\_EVERY_: 1
  - The Actor / Critic pair update is not performed after every step, but each _DDPG\_UPDATE\_EVERY_ step. This helps keep the agent stable and not break due to single outlier in training.
6. _BATCH\_SIZE_: 128
  - Minibatch size used for training neural networks.
7. _OU\_THETA_: 0.12
  - Ornstein-Uhlenbeck noise volatility
8. _OU\_SIGMA: 0.2
  - Ornstein-Uhlenbeck noise speed of mean reversion. 

### Experience Replay

To be able to use all events ecountered during the training (not only ones from the last episode) Experience Replay was implemented. It is a buffer containing ("state", "action", "reward", "next_state", "done") tuples. The buffer is latter used to train the neural network. This improves training, as the NN inside the network has access to events not seen in the current episode.
The experience buffer is common and shared for both actors.

## Ideas for Further Work

My first idea would be to train the agent for a longer time. However, with replay buffer with size of 1e6 and episode length up to 1000, max number of episodes is 1000 before Agents starts forgetting s,a,r,s' pairs. To address that I would try:
  - Use prioritized replay buffer, so agent keeps in memory most useful pairs.
  - Use more advanced neural network in Actor and Critic: use dropout, add more layers. I have not found batch normalization giving impact on the results - probably due to the nature of the Unity env used and its already normalized states. 

## Project files
  - Tennis-Solved.ipynb - notebook with DQN agent implementation
  - README.md - instructions for running the notebook
  - Report.md - this file (report from the project)
  - agent_\$i_checkpoint_\$nn.pth - checkpoint file with trained actor / critic of \$i agent
