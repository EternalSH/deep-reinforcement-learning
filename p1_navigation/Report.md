# Summary

This is first implementation of my DRL agent for banana collection problem. The implemetation is in _Navigation.ipynb_ file (Jupyter Notebook)

## The problem

The environment is a continous limited 2D space. In the space there is an agent and number of purple and yellow bananas. The agent can move in the environment, by stepping over a banana the agent collects it. The goal of the algoritm is to effectively collect yellow bananas while not picking purple ones.

The agent has four actions available: go forward, go backward, turn left, turn right.

## Learning Algorithm

For the first iteration I have used most vanila DRL agent that I have learned in the course. I have done it to verify that fundamentals of my work are correct before going into more advanced techniques.

The main idea behind the agent is to use Neural Network to estimate action value function. Trained network is a function that for each state returns estimated value of each available state. It can be used in following way: given a known state s, network outputs all pairs of possible action and value extected after choosing the action.

### Neural Network Parameters

1. First hidden layer
  - No. of units: 64
  - Activation function: ReLU
2. Second hidden layer
  - No. of units: 64
  - Activation function: ReLU
3. Output layer
  - No. of units: 4 (equal to number of actions)
  - Activation function: none

The neural network responsible for the agent's behaviour has three layers of neurons: two hidden layers and one output layer. Both layers have number of units allowing the network to learn the environment, as the algorithm reached target mean reward. The ReLU function was chosen as a standard activation function for non-linear problems.

### Reinforcement Learning Parameters

1. _n\_episodes_: 1000
  - Limits max number of episodes that agent can learn the environment. It is chosen to ensure the notebook will stop within a reasonable time.
2. _max\_t_: 1000
  - Limits steps per episode. Used default value from other assignments from the RDL nanodegree.
3. _eps_: from 0.01 to 1.00
  - The _eps_ value determines probability of using random action instead of the trained policy. Setting it to 1.0 forces agent to pick actions randomly, while using 0.0 value locks agent into the policy only. Choosing the right value is basically solving a exploration - exploitation problem - balancing both using working plan and trying to find a better one. Approach used in the algorithm uses varying value of the _eps_ - starting from _eps\_start_ at firstand multiplying by _eps\_decay_ factor after each episode, up to _eps\_end_ value. This makes agent favor exploration at the beginning, but prefer exploitation in late episodes.
  1. _eps\_start_: 1.000
  2. _eps\_end_: 0.010
  3. _eps\_decay_: 0.995

### Experience Replay

To be able to use all events ecountered during the training (not only ones from the last episode) Experience Replay was implemented. It is a buffer containing ("state", "action", "reward", "next_state", "done") tuples. The buffer is latter used to train the neural network. This improves training, as the NN inside the network has access to events not seen in the current episode.

### Fixed Q-Targets

To improve stability of the model, Fixed Q-Targets method was incorporated. In a nutshell the basic idea is to use two NN inside the DQN agent: 

- One NN that is constantly updated (trained) after each episode
- One NN that is used by the agent to make decisions (select actions) in the environment. Its weights are updated once each N episodes with values from the first NN.

This approach gives the agent the benefit of more stable behaviour: single outlayer or misdecision will not change the policy drastically. Instead, number of observations are cumulated before updating the policy.

## Rewards

![Graph of score per episode](Report-images/mean-reward-01.png)

The target value of the mean reward per 100 episodes window, equal to 13.0, has been reached in 407 episodes.

## Ideas for Further Work

As mentioned in the beginning, my aim is to verify my approach before going deeper into enhancing the algorithm. I would like to try dueling DQN, as well as test different NN architecture for this particular problem.

## Project files
- Navigation.ipynb - notebook with DQN agent implementation
- README.md - instructions for running the notebook
- Report.md - this file (report from the project)
- checkpoint.pth - checkpoint file with trained DQN agent