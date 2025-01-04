import torch
from torch import nn
from utils.entities import Snake, Food, Statistics
import numpy as np
from collections import deque
import random


 
#   O que você fez hoje (03/12):
#   
#   1 - Corrigi o mapeamento da posição da cobra nos estados.
#   2 - Implementei a memória de replay. Com batch E repetição.
#   
#
#   TODO:
#   - Verificações:
#   
# V 1 - Verificar se tem algum erro na implementação da memória de replay
# 
# v 2 - Verificar se tem algum outro erro no código inteiro. 
#        -> Fazer print por print (essa parte deveria ser como?... E etc)
#
#   3 - Procurar otimizações do código (ChatGPT pode ajudar)
# V 4 - Fazer visualização da quantidade de passos até morrer, e da quantidade
#       média de passos para cada comida comida.
# V 5 - Salvar no mlflow toda vez que apertar Ctrl + C


#   - Melhorias:
# V 1 - Eliminar uma ação. Colocar só esquerda, frente e direita relativa.
# V 2 - Tentar acrescentar a informação da cabeça da cobra E/OU do sentido dela
#       também. (para dar a informação do sentido dela, e de onde ela está. 
#       Esse último porque quando fica em linha reta completa, não dá para saber
#       onde a cabeça está, e parece que é o mesmo estado) (procurar saber mais)

# D 2.1 - Testar a cabeça em um canal diferente da CNN, se não tiver funcionado. (Funcionou bem, posso deixar para depois)

# V 3 - Inicializar bem os parâmetros da rede (ver vídeo do coursera) -> (Pytorch que faz isso (assumo que é ótimo))
#   4 - Testar diferentes combinações de camadas e neurônios.
#   5 - Experimentar Dropout.
# V 6 - Testar com CNN, a partir dos estados da cobra concatenado com a comida, 
#       ou tudo junto como se fosse uma imagem mesmo, mas sem imagem, apenas a 
#       matriz. 
#           -> Testar com uma grid só. Food como "3". Talvez cabeça como "2".
#           -> Colocar direção e cabeça como estados também.
#
#   -- Melhorias RL:
#   1 - Testar recompensa diferente (módulo menor de recompensa)
#   2 - Testar com expected Sarsa (não vai mudar muita coisa, mas é uma testada 
#       rápida de implementar)
#   3 - Implementar o Actor-Critic no pytorch e para o problema.
#   4 - Tentar diferentes formas de estados. Estudar quais estratégias existem
#       para isso, e testar as que forem pertinentes.
#   5 - Depois de tudo isso, se não funcionar mesmo, verificar bem o código se
#       está tudo certo. Se sim, variar os parâmetros de acordo com alguma lógica.
#   6 - Se ainda der ruim, procurar outros métodos que existam. Mas principalmente,
#        ver como as pessoas fizeram e resolveram o problema.
#   7 - Pesquisar sobre offline methods, para testar mais alguma coisa, se quiser e puder.
#   8 - Se as suas não funcionaram, implementar a solução das pessoas. 
#   

class Agent:
    def __init__(self, snake: Snake, food: Food, stats: Statistics):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.snake = snake
        self.food = food
        self.statistics = stats
        self.actions = ["LEFT", "FOWARD", "RIGHT"]
        
        self.last_state = self.get_state()
        self.last_direction = self.snake_direction_onehot
        self.last_action = None

        # Rewards
        self.R_GAME_OVER = -10
        self.R_EAT_FOOD = 10
        self.R_NEUTRAL = 0
        self.R_OPPOSITE = 0

        # PARAMETERS
        self.gamma = 0.98
        self.learning_rate = 1e-4       # Optimizer AdamW
        self.amsgrad = True             # Optimizer AdamW
        self.tau = 1                    # Softmax Policy
        self.batch_size = 64
        self.memory_size = 10000
        self.n_replays = 10

        self.replay_memory = ReplayMemory(self.memory_size)
        self.model = AgentDeepQNetwork(
           # self.last_state.shape[1]*self.last_state.shape[2], 
            len(self.actions)
        ).to(self.device)

        # Otimizer AdamW - Good in complex NNs and in RL
        self.optimizer = torch.optim.AdamW(
           self.model.parameters(), 
           lr=self.learning_rate, 
           amsgrad=self.amsgrad
        )
        
        
    @property
    def snake_pos_grid(self):
        """Executes everytime snake_pos_grid property is visited""" 

        #print("\n\nSnake free positions:")
        #print(self.snake.all_free_positions)

        #print("\n\nSnake body:")
        #print(self.snake.body)
        snake_position = ~self.snake.all_free_positions & 1
        head = self.snake.body[-1]
        snake_position[head[1], head[0]] = 2
        return snake_position

    @property
    def food_pos_grid(self):
        """Executes everytime food_pos_grid property is visited"""

        food_grid = np.zeros(
            (self.food.screen_height // self.food.block_size, 
             self.food.screen_width // self.food.block_size), 
            dtype=int
        )
        food_grid[self.food.position[1], self.food.position[0]] = 1

        #print("\nFood Grid:")
        #print(food_grid)

        return food_grid
    
    @property
    def snake_direction_onehot(self):
        directions = list(self.snake.directions.keys())
        direction_onehot = torch.zeros(
            len(directions), dtype=torch.float32
        ).to(self.device)
        direction_onehot[directions.index(self.snake.direction)] = 1
        direction_onehot = direction_onehot.unsqueeze(0)
        return direction_onehot
        

    def get_state(self):
        return torch.tensor(
                np.stack([self.snake_pos_grid, self.food_pos_grid], axis=0), 
                dtype=torch.float32
            ).to(self.device).unsqueeze(0)

    def policy(self, state, snake_direction, reward=None):
        softmax_policy = torch.nn.Softmax(dim=0)
        action_values = self.model(state, snake_direction)[0]
        
        # Control Softmax
        act_vals_control = action_values/self.tau
        act_vals_control -= torch.max(act_vals_control)
        
        
        actions_soft_probs = softmax_policy(act_vals_control)
        action_index = torch.multinomial(actions_soft_probs, 1).item()
        if reward == self.R_GAME_OVER or reward == self.R_EAT_FOOD:
            pass
            #self.statistics.update_probs_chart(actions_soft_probs.tolist())
            #self.statistics.update_act_values_chart(act_vals_control.tolist())

        return action_index

    def agent_start(self):
        self.last_state = self.get_state()
        self.last_direction = self.snake_direction_onehot

        self.last_action = self.policy(self.last_state, self.last_direction)
        return self.last_action


    def rl_update(self, last_state, last_directions, last_action, reward, 
                  state, directions, is_terminal):
        
        ###### 1 - Get Q(s, a) and Q(s + 1, a):
        Q_s_a = self.model(last_state, last_directions).gather(1, last_action).squeeze(1)
                                        #      (batch, n_actions)
        
        # Get Q(s + 1, :)
        with torch.no_grad():
            Q_next_s_actions = torch.zeros((last_state.shape[0], len(self.actions))).to(self.device)
            Q_next_s_actions[~is_terminal] = self.model(state[~is_terminal], directions[~is_terminal])
            Q_next_s_a_max = torch.max(Q_next_s_actions, dim=1).values
        
        # Q-Learning (After, test with Expected Sarsa)

        #################### RL Update ####################

        ###### 2 - Calculate TD - Error
        target_exp_return = reward + self.gamma*Q_next_s_a_max
        estimated = Q_s_a

        ###### 3 - Back Propagation
        # Defining loss function and Loss
        loss_func = nn.MSELoss()
        loss = loss_func(estimated, target_exp_return)
        
        # Propagate the gradients for each weight
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping - prevent gradient explosion
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        
        # Execute one step of back propagation with AdamW
        self.optimizer.step()

    def soft_update(self):
        
        # Batch (state, action, reward, next_state, is_terminal)
        for _ in range(self.n_replays):
            experiences = self.replay_memory.sample(self.batch_size)
            states, directions, actions, rewards, next_states, \
                next_directions, is_terminal = zip(*experiences)

            states = torch.cat(states, axis=0)
            directions = torch.cat(directions, axis=0)
            next_states = torch.cat(next_states, axis=0)
            next_directions = torch.cat(next_directions, axis=0)
            actions = torch.tensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.stack(rewards, axis=0)#.unsqueeze(0)
            is_terminal = torch.tensor(is_terminal).to(self.device)

            self.rl_update(states, directions, actions, rewards, 
                           next_states, next_directions, is_terminal)



    def agent_step(self, reward):

        terminal_state = reward == self.R_GAME_OVER
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        if terminal_state:
            state = torch.zeros_like(self.last_state).to(self.device)
        else:
            state = self.get_state()

        # Save experience in memory
        # (state, action, reward, next_state, terminal)
        self.replay_memory.append(
            (self.last_state, self.last_direction, self.last_action, 
             reward, state, self.snake_direction_onehot, terminal_state)
        )

        ############# Update #############
        ###### 1 - Get Q(s, a) and Q(s + 1, a):
        # Get Q(s, a)
        Q_s_a = self.model(self.last_state, self.last_direction)[0][self.last_action]

        with torch.no_grad():
            Q_next_s_actions = torch.zeros((1, len(self.actions))).to(self.device)
        
        # Get Q(s + 1, :)
        if not terminal_state:
            with torch.no_grad():
                Q_next_s_actions[:] = self.model(state, self.snake_direction_onehot)
        
        # Q-Learning (After, test with Expected Sarsa)
        with torch.no_grad():
            Q_next_s_a_max = torch.max(Q_next_s_actions)
        
        #################### RL Update ####################

        ###### 2 - Calculate TD - Error
        target_exp_return = reward + self.gamma*Q_next_s_a_max
        estimated = Q_s_a

        ###### 3 - Back Propagation
        # Defining loss function and Loss
        loss_func = nn.MSELoss()
        loss = loss_func(estimated, target_exp_return)
        
        # Propagate the gradients for each weight
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping - prevent gradient explosion
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        
        # Execute one step of back propagation with AdamW
        self.optimizer.step()


        #################### Soft Update (Replay memory) ####################
        if len(self.replay_memory) >= self.batch_size:
            self.soft_update()

        ############# Next Action #############
        self.last_action = self.policy(state, self.snake_direction_onehot, reward)
        self.last_state = state
        self.last_direction = self.snake_direction_onehot
        
        return self.last_action
    

    def agent_end(self, reward):
        ############# Update #############
        ###### 1 - Get Q(s, a) and Q(s + 1, a):
        # Get Q(s, a)
        Q_s_a = self.model(self.last_state, self.last_direction)[0][self.last_action]

        next_state = torch.zeros_like(self.last_state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        # (state, action, reward, next_state, terminal)
        self.replay_memory.append(
            (self.last_state, self.last_direction, 
             self.last_action, reward, next_state, 
             self.snake_direction_onehot, True)
        )

        ###### 2 - Calculate TD - Error
        target_exp_return = reward
        estimated = Q_s_a

        ###### 3 - Back Propagation
        # Defining loss function and Loss
        loss_func = nn.MSELoss()
        loss = loss_func(estimated, target_exp_return)
        
        # Propagate the gradients for each weight
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping - prevent gradient explosion
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        
        # Execute one step of back propagation with AdamW
        self.optimizer.step()

        if len(self.replay_memory) >= self.batch_size:
            self.soft_update()



class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def append(self, experience):
        # (state, direction, action, reward, next_state, terminal)
        self.memory.append(experience)

    def sample(self, batch_size):
        experiences = self.memory
        if batch_size < len(self):
            experiences = random.sample(self.memory, batch_size)
        return experiences

    def __len__(self):
        return len(self.memory)


"""class AgentDeepQNetwork(nn.Module):
    def __init__(self, len_states, len_output):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len_states, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, len_output),
        )

    def forward(self, x):
        x = self.flatten(x)
        #print("forward: x =", x)
        #print("forward: x.shape =", x.shape)
        logits = self.linear_relu_stack(x)
        return logits"""
        


class AgentDeepQNetwork(nn.Module):
    def __init__(self, len_output):
        super(AgentDeepQNetwork, self).__init__()
        
        # Convolutional Layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduz para 4x3
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduz para 2x1
        )
        
        # Fully Connected Layers
        self.fc_stack = nn.Sequential(
            nn.Linear(32 * 2 * 1 + 4, 128),  # Ajuste baseado na saída do último pooling
            nn.ReLU(),
            nn.Linear(128, len_output)
        )

    def forward(self, x, direction):
        #print("forward: x =", x)
        #print("forward: x.shape =", x.shape)

        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv_stack

        # Concatenate the direction information
        x = torch.cat((x, direction), dim=1)   
        x = self.fc_stack(x)
        return x