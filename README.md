# SnakeRL

The goal of this project is to apply what I learned of Reinforcement Learning from the course "Reinforcement Learning Specialization" of University of Alberta & Alberta Machine Intelligence Institute on Coursera. (https://coursera.org/share/18f2f63dd50e447d53aeb43391ed8774)

## Objectives
Evaluate to understand how the following parameters and RL algorithm influence the desired performance in a real project: 
1) Choice of states
2) Choice of actions
3) Choice of Rewards
4) Hyperparameters
5) Choice of algorithm: Deep Q-Learning Network, Actor-Critic Network
6) Choice of neural network (MLP, CNN...) and numbers of layers and neurones.
7) Use of Memmory Replay for planning

## Results

The Snake Game with RL is still on development, but it already achieved some good results if the training time is large.
Currently it gets 16 score, being each score one food eaten.
There is space for enhancing the results, and the future tests and implementation consists in: 
* Apply Actor-Critic algorithm to the problem. It theoretically is more suitable for this problem and it is an algorithm more robust. 
* Apply the concept of offline learning, which usually get better results because of its stability, in comparison with online learning.
* Tuning the number of layers and neurons.
* Vary more the choice of states

## Currently Conclusions

### 1) Choice of States
The choice of states rules one of the most important parts of an RL project. During the development of SnakeRL, it wasn't possible to achieve good results until the choice of states and actions had been reliable. It seems that you have to put in the choice of states everything that an AI, or a person, would need to do the task.

In this case, at the beginning of the project I chose as states the entire grid of the game divided into two grids: one for the snake, and the other for the food.
These grids were represented by tensors with 0 being empty space, and 1 being the snake body or the food.
Only when I added the information of its head AND the direction it was going, the agent started to get good results; that's when it achieved a score of 16.

I've tried to put only the information of the head, because supposedly the direction information is already implicit in it, but it only got good results with the direction information explicitly applied too.

### 2) Choice of Actions

The choice of actions is an important part too. The actions were initially considered as left, right, up, and down; but after some testing, they were changed to relative directions with respect to the current direction of the snake. This was done to eliminate redundancy because the opposite direction was always an invalid choice, as it did nothing during the game, so it was reduced to only 3 actions, as in the game there are in fact just 3 actions while playing (Left, Up, Right).

The influence of this change hasn't been evaluated yet, making it a future update.

### 3) Choice of Reward

For a while, it was kept +10 for eating and -10 for each death. It was tried to raise the magnitude of both values, or to raise the relative difference, but no difference was observed. So, it kept these initial values and they will be tuned later when the Actor-Critic algorithm is in place.

### 4) Hyperparameters

gamma = 0.98 and alpha = 0.0001 were parameters that gave better results while tuning within a small range. It is necessary to tune them, as well as other hyperparameters, when a better solution is found (Application of Actor-Critic, for example).

### 5) Choice of algorithm: Deep Q-Learning Network, Actor-Critic Network

The Deep Q-Learning Network was implemented, giving good results, but not sufficient to finish the game. The Actor-Critic and other algorithms need to be tested.

### 6) Choice of neural network (MLP, CNN...) and numbers of layers and neurones.

CNN and MLP were evaluated, with better results obtained using CNN applied to the grid. The number of layers and neurons still needs to be tuned. (All tuning parts will be done after Actor-Critic is implemented.)

### 7) Use of Memmory Replay for planning

With the use of memory replay, the algorithm learned considerably faster. On the other hand, it slightly slowed down the execution.


## Conclusion: Next Steps

* Implement Actor-Critic algorithm
* Hyperparameter and neural network tunneling.
* Test other choices of states
* Implement offline algoritmns
* Implement other more advanced algorithms.


