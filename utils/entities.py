import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Snake:
    def __init__(self, block_size, screen_width, screen_height):
        self.block_size = block_size
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.direction = None
        self.directions = {
            "LEFT": [-1, 0],
            "RIGHT": [1, 0],
            "UP": [0, -1],
            "DOWN": [0, 1],
        }
        
        self.body = self.get_initial_body(5)
        self.length = len(self.body)
        self.all_free_positions = self.get_all_free_positions()
        
    def get_initial_body(self, initial_lenght=5):
        body = [
            [
                random.randrange(0, self.screen_width//self.block_size - 1, 1),
                random.randrange(0, self.screen_height//self.block_size - 1, 1),
            ]
        ]
        init_direct = random.choice(list(self.directions.keys()))
        self.direction = init_direct
        for i in range(initial_lenght - 1):
            body.append(
                [(body[i][0] + self.directions[init_direct][0]) % (self.screen_width//self.block_size), 
                 (body[i][1] + self.directions[init_direct][1]) % (self.screen_height//self.block_size)]
            )
            #print("body =", body)
            #print("body[i][0] + 1 =", body[i][0] + 1)
            #print("self.screen_width//self.block_size =", self.screen_width//self.block_size)
            #print("(body[i][0] + 1) o/o self.screen_width//self.block_size =", (body[i][0] + 1) % (self.screen_width//self.block_size))

        return np.array(body)

    def get_all_free_positions(self):
        all_free_positions = np.ones(
            (self.screen_height // self.block_size, 
             self.screen_width // self.block_size), 
            dtype=bool
        )
        all_free_positions[self.body[:, 1], self.body[:, 0]] = 0
        return all_free_positions


    def move(self):
        if self.direction:
            head = self.body[-1]
            dx, dy = self.directions[self.direction]
            new_head = [head[0] + dx, head[1] + dy]

            # Wrap around screen
            new_head[0] %= self.screen_width//self.block_size
            new_head[1] %= self.screen_height//self.block_size
            self.body = np.append(self.body, [new_head], axis=0)
            
            # Maintain length
            if len(self.body) > self.length:
                self.body = np.delete(self.body, 0, 0)

            self.all_free_positions = self.get_all_free_positions()

            

    def grow(self):
        self.length += 1

    def check_collision(self):
        head = self.body[-1]
        return any(np.equal(self.body[:-1], head).all(1))

    def draw(self, screen, color):
        for segment in self.body:
            pygame.draw.rect(
                screen, color, 
                [segment[0]*self.block_size, 
                 segment[1]*self.block_size, 
                 self.block_size, 
                 self.block_size]
            )


class Food:
    def __init__(self, snake: Snake):
        self.snake = snake
        self.block_size = snake.block_size
        self.screen_width = snake.screen_width
        self.screen_height = snake.screen_height

        # Set the initial position of the food
        self.position = np.array([None, None])
        self.generate_position()

    def generate_position(self):
        free_indices = np.argwhere(self.snake.all_free_positions == 1)
        chosen_index = random.choice(free_indices)
        self.position[0] = chosen_index[1]
        self.position[1] = chosen_index[0]

    def draw(self, screen, color):
        pygame.draw.rect(
            screen,
            color,
            [self.position[0]*self.block_size, 
             self.position[1]*self.block_size, 
             self.block_size, self.block_size],
        )


class Statistics:
    def __init__(self):
        self.n_gameovers = 0


    def init_prob_chart(self, actions):
        self.fig_prob, self.ax_prob, self.bars_probs = self.init_bar_chart(
            actions, limit=1
        )

    def update_probs_chart(self, action_values):
        self.update_bar_chart(self.bars_probs, action_values)

    def init_act_values_chart(self, actions):
        self.fig_avs, self.ax_avs, self.bars_avss = self.init_bar_chart(
            actions, title="Action Values Q(s, a)", ylabel="Q(s, a)"
        )

    def update_act_values_chart(self, action_values):
        self.update_bar_chart(self.bars_avss, action_values, self.ax_avs)


    def init_bar_chart(self, actions_labels, title="Policy pi(s)", ylabel="Probability", limit=None, num_bars=4):
        fig_bar, ax_bar = plt.subplots()
        ax_bar.set_title(title)
        ax_bar.set_xlabel("Action")
        ax_bar.set_ylabel(ylabel)
        bars = ax_bar.bar(range(1, num_bars + 1), [0] * num_bars, color='blue', edgecolor='black')

        ax_bar.set_xticks(range(1, num_bars + 1))
        ax_bar.set_xticklabels(actions_labels, ha='center')

        ax_bar.set_xlim(0, num_bars + 1)
        if limit is not None:
            ax_bar.set_ylim(0, limit)  # Adjust interval as needed for q_values
        return fig_bar, ax_bar, bars


    def update_bar_chart(self, bars, values, ax=None):
        # Update bar data
        for bar, action_value_prob in zip(bars, values):
            #print(f'bar {bar}, action_value_prob {action_value_prob}')
            bar.set_height(action_value_prob)

        if ax:
            ax.set_ylim(min(values) - 1, max(values) + 1)

        # For visualization
        plt.pause(0.1)  





    def init_plot_score(self):

        plt.ion()  # Turns on interative mode
        self.fig_score, self.ax_score = plt.subplots()
        self.x_games, self.y_score, self.mov_avg_score = [], [], []
        
        # Set lines
        self.line, = self.ax_score.plot(
            self.x_games, self.y_score, 'b-', label="Score"
        )  
        self.line_mov_avg, = self.ax_score.plot(
            self.x_games, self.mov_avg_score, 'o-', label="Mob Mean Score"
        )  

        # Axis configs
        self.ax_score.set_xlim(0, 10)
        self.ax_score.set_ylim(0, 10)

        # Configure o eixo X para ter uma quantidade razoável de marcações
        self.ax_score.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
        self.ax_score.yaxis.set_major_locator(MaxNLocator(integer=True))

        self.ax_score.set_title("Training Snake Agent")
        self.ax_score.set_xlabel("Games")
        self.ax_score.set_ylabel("Score")
        self.ax_score.legend()

        # Add annotations for the ends of the lines
        self.annotation1 = self.ax_score.annotate("", xy=(0, 0), xytext=(0, 0),
                          textcoords="offset points", color="red", fontsize=10)
        self.annotation2 = self.ax_score.annotate("", xy=(0, 0), xytext=(0, 0),
                          textcoords="offset points", color="blue", fontsize=10)
        return True

    def compute_mov_avg_score(self, lenght=5):
        if len(self.y_score) >= lenght:
            return sum(self.y_score[-1:-lenght-1:-1])/lenght
        return sum(self.y_score)/len(self.y_score)


    def update_plot_score(self, score):        
        # New values of X and Y
        self.n_gameovers += 1
        self.x_games.append(self.n_gameovers)  
        self.y_score.append(score)        
        self.mov_avg_score.append(self.compute_mov_avg_score(lenght=10))        
        
        # Update line data
        self.line.set_xdata(self.x_games)      
        self.line.set_ydata(self.y_score)
        self.line_mov_avg.set_xdata(self.x_games)      
        self.line_mov_avg.set_ydata(self.mov_avg_score)

        # Adjusts X and Y limits if needed
        self.ax_score.set_xlim(0, max(self.x_games) + 1)  
        self.ax_score.set_ylim(min(self.y_score) - 1, max(self.y_score) + 1)
        
        # Update annotation text and position for Line 1
        self.annotation1.set_text(f"{self.y_score[-1]:.4f}")  # Numeric value for Line 1
        self.annotation1.xy = (self.x_games[-1], self.y_score[-1])  # Place annotation at the end of Line 1

        # Update annotation text and position for Line 2
        self.annotation2.set_text(f"{self.mov_avg_score[-1]:.4f}")  # Numeric value for Line 2
        self.annotation2.xy = (self.x_games[-1], self.mov_avg_score[-1])  # Place annotation at the end of Line 2

        plt.pause(0.1)  # Pausa para atualização visual