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
#   0 - Colocar no GitHub
#   1 - Verificar se tem algum erro na implementação da memória de replay
#   2 - Verificar se tem algum outro erro no código inteiro. 
#        -> Fazer print por print (essa parte deveria ser como?... E etc)
#
#   3 - Procurar otimizações do código (ChatGPT pode ajudar)
#   4 - Fazer visualização da quantidade de passos até morrer, e da quantidade
#       média de passos para cada comida comida.


#   - Melhorias:
#   1 - Eliminar uma ação. Colocar só esquerda, frente e direita relativa.
#   2 - Tentar acrescentar a informação da cabeça da cobra E/OU do sentido dela
#       também. (para dar a informação do sentido dela, e de onde ela está. 
#       Esse último porque quando fica em linha reta completa, não dá para saber
#       onde a cabeça está, e parece que é o mesmo estado) (procurar saber mais)
#   3 - Inicializar bem os parâmetros da rede (ver vídeo do coursera)
#   4 - Testar diferentes combinações de camadas e neurônios.
#   5 - Experimentar Dropout.
#   6 - Testar com CNN, a partir dos estados da cobra concatenado com a comida, 
#       ou tudo junto como se fosse uma imagem mesmo, mas sem imagem, apenas a 
#       matriz. 
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
        self.actions = list(self.snake.directions.keys())
        
        self.last_state = self.get_state()
        self.last_action = None

        # Rewards
        self.R_GAME_OVER = -50
        self.R_EAT_FOOD = 50
        self.R_NEUTRAL = 0
        self.R_OPPOSITE = 0

        # PARAMETERS
        self.gamma = 0.98
        self.learning_rate = 1e-4       # Optimizer AdamW
        self.amsgrad = True             # Optimizer AdamW
        self.tau = 1                    # Softmax Policy
        self.batch_size = 60
        self.memory_size = self.batch_size*4
        self.n_replays = 10

        self.replay_memory = ReplayMemory(self.memory_size)
        self.model = AgentDeepQNetwork(
            self.last_state.shape[1]*self.last_state.shape[2], 
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


        return ~self.snake.all_free_positions & 1

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

    def get_state(self):
        return torch.tensor(
            np.append(
                self.snake_pos_grid, self.food_pos_grid, axis=0), 
                dtype=torch.float32
            ).to(self.device).unsqueeze(0)

    def policy(self, state, reward=None):
        softmax_policy = torch.nn.Softmax(dim=0)
        action_values = self.model(state)[0]
        
        # Control Softmax
        act_vals_control = action_values/self.tau
        act_vals_control -= torch.max(act_vals_control)
        
        
        actions_soft_probs = softmax_policy(act_vals_control)
        action_index = torch.multinomial(actions_soft_probs, 1).item()
        if reward == self.R_GAME_OVER or reward == self.R_EAT_FOOD:
            self.statistics.update_probs_chart(actions_soft_probs.tolist())
            self.statistics.update_act_values_chart(act_vals_control.tolist())

        return action_index

    def agent_start(self):
        #print("\n\nagent_start")
        self.last_state = self.get_state()
        #print("self.last_state =", self.last_state)
        #print("self.last_state.shape =", self.last_state.shape)

        self.last_action = self.policy(self.last_state)
        #print("Action =", self.last_action, self.actions[self.last_action])
        return self.last_action


    def rl_update(self, last_state, last_action, reward, state, is_terminal):
        
        ###### 1 - Get Q(s, a) and Q(s + 1, a):
        # Get Q(s, a)
        #print("Soft Update: last_state =", last_state)
        #print("last_state.shape =", last_state.shape)
        #print("self.model(last_state) =", self.model(last_state))
        #print("last_action =", last_action)

        Q_s_a = self.model(last_state).gather(1, last_action).squeeze(1)
        #print("Q_s_a =", Q_s_a)
        
                                        #      (batch, n_actions)
        
        #print("is_terminal =", is_terminal)

        # Get Q(s + 1, :)
        with torch.no_grad():
            Q_next_s_actions = torch.zeros((last_state.shape[0], len(self.actions))).to(self.device)
            Q_next_s_actions[~is_terminal] = self.model(state[~is_terminal])
            Q_next_s_a_max = torch.max(Q_next_s_actions, dim=1).values
        
        #print("Q_next_s_actions =", Q_next_s_actions)

        # Q-Learning (After, test with Expected Sarsa)

        #print("Q_next_s_a_max =", Q_next_s_a_max)
        #print("Q_next_s_a_max.shape =", Q_next_s_a_max.shape)
        #print("reward =", reward)
        #print("reward.shape =", reward.shape)


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


    def agent_step(self, reward):

        #print("\n\nagent_step")
        
        terminal_state = reward == self.R_GAME_OVER
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        if terminal_state:
            state = torch.zeros_like(self.last_state).to(self.device)
        else:
            state = self.get_state()


        # Save experience in memory
        # (state, action, reward, next_state, terminal)
        self.replay_memory.append(
            (self.last_state, self.last_action, reward, state, terminal_state)
        )

        ############# Update #############
        ###### 1 - Get Q(s, a) and Q(s + 1, a):
        # Get Q(s, a)
        ##print("Real Update: self.last_state =", self.last_state)
        ##print("self.last_state.shape =", self.last_state.shape)
        ##print("self.model(self.last_state) =", self.model(self.last_state))
        ##print("self.last_action =", self.last_action)

        Q_s_a = self.model(self.last_state)[0][self.last_action]
        ##print("Q_s_a =", Q_s_a)

        with torch.no_grad():
            Q_next_s_actions = torch.zeros((1, len(self.actions))).to(self.device)
        
        # Get Q(s + 1, :)
        if not terminal_state:
            with torch.no_grad():
                Q_next_s_actions[:] = self.model(state)
        
        ##print("Q_next_s_actions =", Q_next_s_actions)
        # Q-Learning (After, test with Expected Sarsa)
        with torch.no_grad():
            Q_next_s_a_max = torch.max(Q_next_s_actions)
        
        ##print("Q_next_s_a_max =", Q_next_s_a_max)

        
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
        # Batch (state, action, reward, next_state, is_terminal)
        for _ in range(self.n_replays):
            experiences = self.replay_memory.sample(self.batch_size)
            states, actions, rewards, next_states, is_terminal = zip(*experiences)

            #print("\n\nReplay Memory len =", len(self.replay_memory))
            #print("Rewards =", rewards)
            #print("states =", states)
            #print("len(states) =", len(states))
            #print("actions =", actions)
            #print("next_states =", next_states)
            #print("is_terminal =", is_terminal)


            states = torch.cat(states, axis=0)
            next_states = torch.cat(next_states, axis=0)
            actions = torch.tensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.stack(rewards, axis=0)#.unsqueeze(0)
            is_terminal = torch.tensor(is_terminal).to(self.device)

            #print("\n\nAfter reshape:")
            #print("Rewards =", rewards)
            #print("states =", states)
            #print("states.shape =", states.shape)
            #print("actions =", actions)
            #print("next_states =", next_states)
            #print("is_terminal =", is_terminal)



            self.rl_update(states, actions, rewards, next_states, is_terminal)

        ############# Next Action #############
        self.last_action = self.policy(state, reward)
        self.last_state = state
        
        return self.last_action
    

    def agent_end(self, reward):
        ############# Update #############
        ###### 1 - Get Q(s, a) and Q(s + 1, a):
        # Get Q(s, a)
        Q_s_a = self.model(self.last_state)[0][self.last_action]

        next_state = torch.zeros_like(self.last_state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        # (state, action, reward, next_state, terminal)
        self.replay_memory.append((self.last_state, self.last_action, reward, next_state, True))

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




class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def append(self, experience):
        # (state, action, reward, next_state, terminal)
        self.memory.append(experience)

    def sample(self, batch_size):
        experiences = self.memory
        if batch_size < len(self):
            experiences = random.sample(self.memory, batch_size)
        return experiences

    def __len__(self):
        return len(self.memory)


class AgentDeepQNetwork(nn.Module):
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
        return logits
        