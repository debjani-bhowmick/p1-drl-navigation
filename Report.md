# Udacity Deep Reinforcement Learning Nanodegree 
# Project 1: Navigation

## Table of Contents
1. [Description](Description])
2. [Environment](Environment)
3. [Getting Started](Getting_Started)
   * [Getting the Code](Installing)
   * [Code Execution](Executing_Program)
   * [File Description](File_Description)
4. [Establish Baseline Using A Random Action Policy](Establish_baseline_using_a_random_action_policy)
5. [Description Of Algorithms Used](Description_of_algorithms_used)
6. [Run Experiments to measure agent performance](Run_experiments_to_measure_agent_performance)
7. [Select best performing agent](Select_best_performing_agent)
8. [Authors](Authors)
9. [License](License)
10. [Acknowledgement](Acknowledgement)

## Description
This report as well as the ssociated code are meant to serve as an assignment project for the partial fulfilment of Deep Reinforcement Learning Nanodegree hosted by Udacity.The goal of this project is to  **Train an agent to navigate a virtual world and collect as many yellow bananas as possible while avoiding blue bananas**. An understanding of the environment in which the agent needs to operate is shown in the image below.

![In Project 1, train an agent to navigate a large world.](images/banana_agent.gif)

In the setup shown above, the agent is rewarded with a score of +1 whenever it successfully collects a yellow banana. However, it is also penalized with a score of -1 whenever it commits the error of collecting a blue banana. The reinforcement learning system actions are to be directed towards maximizing the cumulative scores, thus, the agent is supposed to end up with a policy which understands that the blue bananas are to be avoided and the yellow bananas are to be collected as many as possible through changing its actions on certain states.

Before running the actual code, user needs to install the right dependencies in the notebook. The corresponding libraries are then imported and the simulation environment is initialized. After the full setup is done, the goal is to explore the State and Action Spaces. State space is a vector defined by 37 features including the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Further, the action space has a dimension of four (turn left and right, move forward and backward). To learn how the Python API controls the agent and receives the feedback from the environment, a code cell is provided with a random action agent.

Based on the exisiting policy, agents take action in the environment at that time step. The primary objective of the learning algorithm is to find an optimal policy&mdash;i.e., a policy that maximizes the reward for the agent. It is important to note that the set of possible actions are not known a priori, thus the optimal policy has to be discovered by interacting with the environment and recording observations. Therefore, the agent "learns" the policy through a process of trial-and-error that iteratively maps various environment states to the actions that yield the highest reward. This type of algorithm is called **Q-Learning**.

 We further provide below the details related to the choice of environment, details related to its setup, instructions related to the installation of the code, experimental details and the main findings, among others. 

## Environment <a name="Environment"></a>

The environment used in this project is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents)

Note: We point out here that the environment used in the project provided by Udacity is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page.

> The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API. 

A stated earlier, the environment rewards a score of +1 whenever a yelloe banana is collected, and an adverse reward of -1 is provided for whenever collecting a blue banana. The goal of the problem is to maximum the overall cumulative score, which implies collecting maximum possible number of yellow bananas while minimizing the collection of blue bananas as much as possible. 

The state space has a dimension of 37 which contain velocity of the agent along with information on ray-based perception of objects around the agent's forward direction. Based on this information, the goal for the agent is to learn to select the best actions. There are 4 discrete choices of actions available to the agent, and these can be stated as

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and **in order to solve the environment, the agent must get an average score of +16 over 100 consecutive episodes.**

## Getting Started <a name="Getting_Started"></a>
### Step 1: Setting up the Environment : 
For details related to setting up the Python environment for this project, please follow the instructions provided in the DRLND GitHub repository[https://github.com/udacity/deep-reinforcement-learning]. These instructions can be found in README.md at the root of the repository. By following these instructions, user will be able to install the required PyTorch library, the ML-Agents toolkit, and a few more Python packages required for this project.

(For Windows users) The ML-Agents toolkit supports Windows 10 currently. In general, ML-Agents toolkit could possibly be used for other versions, however, it has not been tested officially, and we recommend choosing Windows 10. Also, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

Further, the specific files to look into in the repository is `python/setup.py and requiremnets.txt`. The readme provides thorough details related to setting up the environment.

### Step 2: Download the Unity Environment
For this project, installing Unity is not required, rather a pre-built environment is readily available to download from one of the links provided below. Note that based on the operating system, the choice can differ, hence the version link needs to be used.

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
After the environment is downloaded, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) To know whether you are using a 32-bit or a 64-bit machine, visit this [link](https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d).

### Getting the Code <a name="Installing"></a>

This code is hosted on github and can be downloaded as a zip directly or through cloning the repository.To clone the git repository:
 
```[git clone debjani-bhowmick/p1-drl-navigation)](https://github.com/debjani-bhowmick/p1-drl-navigation)```

### Code Execution <a name="Executing_Program"></a>
To run the code, see `Navigation.ipynb` notebook in the project's directory. Details on training and saving the model are provided in this notebook. 

### File Description <a name=" File_Description"></a>
This project structure is divided into three directories:

<b> model.py:</b> Contains the neural network architecture.

<b> dqn_agent.py:</b> Contains the implementation of the DQN algorithm.

<b> Navigation.ipynb:</b> IPython notebook designed to provide an understanding of how the agent works step by step using different algorithms.

<b> folder:python:</b> This folder has been directly copied from the original repository of Udacity Deep Reinforcement Learning Nanodegree, and contains the files related to installation and set up of the environment.

<b> folder:checkpoints:</b> Contains the models saved during training.

<b>folder:Images:</b> Contains screenshots of the results as well as additional images used for this document.


### Establish Baseline <a name="Establish Baseline"></a>

To evaluate the performance of the agent, it is important that a suitable baseline is chosen. In this study, I started by testing an agent that selects actions (uniformly) at random at each time step.

```python
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))
```

Based on this agent, I obtained a cumulative score of zero. Ideally, any score above this implies that the agent is performing better than the baseline of making random choices. Since our agent needs to achieve a score of at least 16 over 100 consecutive episodes, it is implied that smarter algorithms are needed and the random choices would not help. 

## Description of algorithms used <a name="Description of algorithms used"></a>

#### Q-Function
To discover an optimal policy, a Q-function has been set up. The Q-function calculates the expected reward `R` for all possible actions `A` in all possible states `S`.

<img src="assets/Q-function.png" width="19%" align="top-left" alt="" title="Q-function" />

The optimal policy `??*` for our case can then be defined as the action that maximizes the Q-function for a given state across all possible states. The optimal Q-function `Q*(s,a)` maximizes the total expected reward for an agent starting in state `s` and choosing action `a`, then following the optimal policy for each subsequent state.

<img src="images/optimal-policy-equation.png" width="47%" align="top-left" alt="" title="Optimal Policy Equation" />

For obtaining discounted returns at future time steps, the Q-function can be expanded to include the hyperparameter gamma `??`.

<img src="images/optimal-action-value-function.png" width="67%" align="top-left" alt="" title="Optimal Action Value Function" />

While this algorithm is already a simple yet effective choice, a challenge associated with it is in choosing the right action to take while the agent is still learning the policy. Should the agent choose an action based on the Q-values observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? This is known as the **exploration vs. exploitation dilemma**.

#### Epsilon Greedy Algorithm

The challenge outlined above can be addressed using the **????-greedy algorithm**, and we implement it next in the project. With this approach, our agent can systematically manage the exploration vs. exploitation trade-off. The agent "explores" by picking a random action with some probability epsilon `????`. However, the agent continues to "exploit" its knowledge of the environment by choosing actions based on the policy with probability (1-????).

The value of epsilon is slowly reduced over time. Thus, while the agent favors exploration in the early stages of its learning process, it tends to favor exploitation as it gains more experience. The starting and ending values for epsilon, and the rate at which it decays are three hyperparameters that are later tuned during experimentation.

For details related to the logic behind ????-greedy approach, see `agent.act()` method [here](https://github.com/debjani-bhowmick/p1-drl-navigation/master/model/agent/agent.py#L66) in `agent.py` of the source code.


#### Deep Q-Network (DQN)

Deep Q-Learning involves the use of a deep neural network for approximating the Q-function. Given a network `F`, discovering the optimal policy in this case is equivalent to  finding the best weights `w` such that `F(s,a,w) ??? Q(s,a)`.

The neural network architecture used for this project can be found [here](https://github.com/debjani-bhowmick/p1-drl-navigation/master/model/model.py#L5) in the `model.py` file of the source code. The network contains three fully connected layers with 64, 64, and 4 nodes respectively. In our experience, testing with bigger networks (more nodes) as well as deeper networks (more layers) did not produce better results.

As for the network inputs, rather than feeding-in sequential batches of experience tuples, I randomly sample from a history of experiences using an approach called Experience Replay.


#### Experience Replay

With the experience replay incorporated, an RL agent is capable of learning from its past experience. 

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. As part of the learning step, sampling is performed by the agent from this buffer. Sampling of experience is performed in a manner that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. Through multiple passes performed over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found [here](https://github.com/debjani-bhowmick/p1-drl-navigation/master/model/agent/agent.py#L133) in the `agent.py` file of the source code.

#### Double Deep Q-Network (DDQN)

One issue with Deep Q-Networks is they can overestimate Q-values (see [Thrun & Schwartz, 1993](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)). The accuracy of the Q-values depends on which actions have been tried and which states have been explored. If the agent hasn't gathered enough experiences, the Q-function will end up selecting the maximum value from a noisy set of reward estimates. Early in the learning process, this can cause the algorithm to propagate incidentally high rewards that were obtained by chance (exploding Q-values). This could also result in fluctuating Q-values later in the process.

<img src="images/overestimating-Q-values.png" width="50%" align="top-left" alt="" title="Overestimating Q-values" />

We can address this issue using Double Q-Learning, where one set of parameters `w` is used to select the best action, and another set of parameters `w'` is used to evaluate that action.  

<img src="images/DDQN-slide.png" width="40%" align="top-left" alt="" title="DDQN" />

The DDQN implementation can be found [here](https://github.com/debjani-bhowmick/p1-drl-navigation/master/agent/agent.py#L96) in the `agent.py` file of the source code.


#### Dueling Networks 

The underlying idea of these networks is that they utilize two streams: one that estimates the state value function `V(s)`, and another that estimates the advantage for each action `A(s,a)`. These two values are then combined to obtain the desired Q-values.  

<img src="images/dueling-networks-slide.png" width="60%" align="top-left" alt="" title="DDQN" />

The reasoning behind this approach is that state values don't change much across actions, thus these could be estimated directly. However, the advantage function is used since we would like to measure the impact that individual actions have in each state.

The dueling agents are implemented within the fully connected layers [here](https://github.com/debjani-bhowmick/p1-drl-navigation/master/model/model.py#L21) in the `model.py` file of the source code.


### Model parameters and results

Details related to parameter values for used by the agent are stated below. These details can also be found in `dqn_agent.py` file of the code.

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4 
Optimizer : Adam
```

The employed neural networks in the implementation comprise the architecture as stated below, and this is defined in `model.py` of the code.

```
Input nodes (37) -> Fully Connected Layer (64 nodes, Relu activation) -> Fully Connected Layer (64 nodes, Relu activation) -> Ouput nodes (4)
```


Given the chosen architecture and parameters, our results are :

### 4. Run Experiments:

After the different components related to the algorithm have been set up and things are in place, the next goal is to measure the performance of our agent in the Banana environment. Performance is measured by the fewest number of episodes required to solve the environment.

Details related to the full set of experiments are shown in the table below. As can be seen, these experiments compare different combinations of the components and hyperparameters discussed above. For all the experiments, agents utilize a replay buffer.

<img src="images/experimentation_table.png" width="80%" align="top-left" alt="" title="experimentation_table" />



### 5. Select best performing agent

The best performing agents were able to achieve the desired goal in the provided environment in around 400-500 episodes. Although the overall set includes agents that utilized Double DQN and Dueling DQN, ultimately, best performance was achieved with a combined agent using Double DQN with replay buffer and Dueling (setup: Epsilon decay: 0.98 and epsilon end = 0.02).

<img src="images/DDQN_RB_Dueling_0.98_0.02.png" width="60%" align="top-left" alt="" title="Score evolution during the training of DDQN_RB_Dueling_Agent" />

**These results meets the project's expectation as the agent is able to receive an average reward (over 100 episodes) of at least +13 ( I kept 16 as limit). This agent met the criteria with least iteration, so I choose it as best among others.


### 6. Future improvemnet ideas:

#### Test the replay buffer 
  Here in this project all agents were trained with replay buffer, so as of now there is no scope to understand the effect of replay buffer. It would be nice to keep it optional while coding to check the impact the replay buffer on the performance.

#### Add prioritized experience replay 
  Prioritized replay selects experiences based on a priority value that is proportinal to the error. This can improve learning by increasing the probability of rare instances by sampling important experience vectors.


### Lic<b>ensing, Authors, Acknowledgements <a name=" Licensing, Authors, Acknowledgements"></a>
<b> Author:</b> Debjani Bhowmick
  
<b> Acknowledgements: </b>Author would like to thank Udacity for providing an amazing Data Science Nanodegree Program.


