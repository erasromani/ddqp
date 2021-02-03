# Deep Deterministic Policy Gradient

For this exercise, I implemented DDPG to the OpenAI Gym BipedalWalkerHarcore-v3 environment. The agent learned through experience using only pixels as input. Similar techniques used in [Lillicarp et al, Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) were implemented here such as the use of a replay buffer and two separate networks for the both the critic and actor. Other techniques used included use of epsilon decay and polyak averaging between the target and local network. Below is the epsilon decay schedule used. 

For the actor, a standard four-layer 1D convolutional neural network was used followed by two fully connected layers. For the critic, a standard four-layer 1D convolutional neural was used to encode the state information. The encoding was concatenated with the action which was then fed into a three layer fully connected network to yield the q value. Unfortunately, I did not have time to do an adequate grid search approach for the hyperparameters as each run was taking too long. Training loss did not appear to be a good metric for determining whether or not the system is learning. I therefore had to rely on the reward as an indicator. Below is a loss and reward per episode throughout training.

Note that although the final reward per episode is relatively low, there does appear to be some learning happening. I believe additional hyperparameter tuning for the learning rate of the actor and the critic would have yielded better results. 
I evaluated the system performance using the weights obtained from different points in time throughout training. As you can see from the figure below, the best performance is attained after 4000 training episodes. After 4000 episodes, the performance starts to drop.


Below are screenshots of the agent during testing for reference.

