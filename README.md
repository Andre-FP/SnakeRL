# SnakeRL

The goal of this project is to apply what I learned of Reinforcement Learning from the course "Reinforcement Learning Specialization" of University of Alberta & Alberta Machine Intelligence Institute on Coursera. (https://coursera.org/share/18f2f63dd50e447d53aeb43391ed8774)

## Objectives
Evaluate for understanding how the following parameters and RL algorithmn influence the performance desired in a real project: 
1) Choice of states
2) Choice of actions
3) Choice of Rewards
4) Hyperparameters
5) Choice of algorithmn: Deep Q-Learning Network, Actor-Critic Network
6) Choice of neural network (MLP, CNN...) and numbers of layers and neurones.
7) Use of Memmory Replay for planning

## Results

The Snake Game with RL is still on development, but it already acquired some good results if the training time is large.
Currently it gets 16 score, being each score one food eaten.
There is space for enhancing the results, and the future tests and implementation consists in: 
* Apply Actor-Critic algorithmn to the problem. It theoretically is more suitable for this problem and it is an algorithmn more robust. 
* Apply the concept of offline learning, which usually get better results because of its stability, in comparison with online learning.
* Tunning the numbers of layers and neurones.
* Vary more the choice of states

## Currently Conclusions

### 1) Choice of States
The choice of states rules one of the most importants parts of an RL project. During the development of SnakeRL, it wasn't being possible to achieve good results until the choice of states an action have been reliable. It seems that you have to put in the choice of states everything that an AI, or a person, would need to do the task. 

In this case, in the beggining of the project I chosed as states all the grid of the game devided in two grids: one for the snake, and the other for the food.
These grids were represented by tensors with 0 being empty space, and 1 being the snake body or the food.
Only when I added the information of its head AND the direction it was going, the agent started to get good results, that's was when it achieved 16 of score.

I've tried to put only the information of the head, because supposely the direction information are already implicit on it, but it only got good results with the information of direction explicitly apllied too.

### 2) Choice of Actions

The choice of actions is an important part too. The action initially were considered the left, right, up and down; but after some testing it was changed for relative directions in respect of the current direction of the snake. It was done to eliminate redundancy because the opposite direction always was an invalid choice as it did nothing during the game, so it was resumed for only 3 actions as in the game there are in fact just 3 action while playing (Left, Up, Right).

The influence of its change haven't be evaluated yet, becoming a future update.

### 3) Choice of Reward

For a while it was kept +10 for eating and -10 for each death. It was tried to rise the magnitude of both values, or to rise the relative difference but it was not observed any difference. So it kept these initial values and it will be tunned later when the actor critic algorithmn being in place.

### 4) Hyperparameters

gamma = 0.98 and alpha = 0.0001 were parameters that gave better results while tuning within a small range. It is necessary tunneling them, as well as other hyperparameters, when a better solution is finded (Aplication of Actor-Critic, for example).  

### 5) Choice of algorithmn: Deep Q-Learning Network, Actor-Critic Network

The Deep Q-Learning Network were implemented giving good results, but not suficient to finish the game. The Actor-Critic and other algorithmns need to be tested

### 6) Choice of neural network (MLP, CNN...) and numbers of layers and neurones.

CNN and MLP were evaluated, getting better results with the CNN applied to the grid. The numbers of layers and neurones need to be tuned also. (All tunneling parts will be done after Actor-Critic implemented)

### 7) Use of Memmory Replay for planning

With the use of memmory replay, the algorithmn learned incontably faster. On the other hand, it slightly slowed down the execution.


## Conclusion: Next Steps

* Implement Actor-Critic algorithmn
* Hyperparameter and neural network tunneling.
* Test others choices of states
* Implement offline algoritmns
* Implement others more advanced algorithmns.


