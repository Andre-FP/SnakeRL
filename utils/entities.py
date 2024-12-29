import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import mlflow
import os


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
        self.ax_prob = None
        self.ax_avs = None
        self.ax_death = None
        self.ax_step_food = None
        self.ax_score = None
        

    def save_statistics_mlflow(self):
        axis = [self.ax_death, self.ax_step_food, self.ax_score]
        plots_names = ['StepsDeath.png', 'StepsFood.png', 'Score.png']
        for i, ax in enumerate(axis):
            ax.figure.savefig(plots_names[i])
            mlflow.log_artifact(plots_names[i])
            os.remove(plots_names[i])


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


    def init_plot_graph(self, title="Training Snake Agent", xlabel="Games", ylabel="Score"):
        
        plt.ion()  # Turns on interative mode
        _, ax_score = plt.subplots()
        x_games, y_score, mov_avg_score = [], [], []
        
        # Set lines
        line, = ax_score.plot(
            x_games, y_score, 'b-', label="Score"
        )  
        line_mov_avg, = ax_score.plot(
            x_games, mov_avg_score, 'o-', label="Mob Mean Score"
        )  

        # Axis configs
        ax_score.set_xlim(0, 10)
        ax_score.set_ylim(0, 10)

        # Configure o eixo X para ter uma quantidade razoável de marcações
        ax_score.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
        ax_score.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax_score.set_title(title)
        ax_score.set_xlabel(xlabel)
        ax_score.set_ylabel(ylabel)
        ax_score.legend()

        # Add annotations for the ends of the lines
        annotation1 = ax_score.annotate("", xy=(0, 0), xytext=(0, 0),
                          textcoords="offset points", color="red", fontsize=10)
        annotation2 = ax_score.annotate("", xy=(0, 0), xytext=(0, 0),
                          textcoords="offset points", color="blue", fontsize=10)
        
        return ax_score, annotation1, annotation2, \
            x_games, y_score, mov_avg_score, line, line_mov_avg


    def init_plot_score(self):
        self.ax_score, self.annot1_score, self.annot2_score, \
            self.x_games_score, self.y_score, self.mov_avg_score, \
            self.line_score, self.line_mov_avg_score = self.init_plot_graph(
                title="Training Snake Agent", 
                xlabel="Games", 
                ylabel="Score"
            )
        return True
    
    def init_plot_steps_death(self):
        self.ax_death, self.annot1_death, self.annot2_death, \
            self.x_games_death, self.y_death, self.mov_avg_death, \
            self.line_death, self.line_mov_avg_death = self.init_plot_graph(
                title="Training Snake Agent - Steps until Death", 
                xlabel="Games", 
                ylabel="Steps - Death"
        )
        return True
    
    def init_plot_steps_food(self):
        self.ax_step_food, self.annot1_step_food, self.annot2_step_food, \
            self.x_games_step_food, self.y_step_food, self.mov_avg_step_food, \
            self.line_step_food, self.line_mov_avg_step_food = self.init_plot_graph(
                title="Training Snake Agent - Steps until Food", 
                xlabel="Games", 
                ylabel="Steps - Food"
            )
        return True
    

    def compute_mov_avg_score(self, buffer_score, lenght=5):
        if len(buffer_score) >= lenght:
            return sum(buffer_score[-1:-lenght-1:-1])/lenght
        return sum(buffer_score)/len(buffer_score)


    def update_plot_graph(self, score, ax: plt.Axes, annot1: plt.Annotation, 
                          annot2: plt.Annotation, x_games: list, 
                          y_value: list, y_avg: list, 
                          line: Line2D, line_avg: Line2D):

        # New values of X and Y
        x_games.append(len(x_games) + 1)  
        y_value.append(score)        
        y_avg.append(self.compute_mov_avg_score(y_value, lenght=10))        
        
        # Update line data
        line.set_xdata(x_games)      
        line.set_ydata(y_value)
        line_avg.set_xdata(x_games)      
        line_avg.set_ydata(y_avg)

        # Adjusts X and Y limits if needed
        ax.set_xlim(0, max(x_games) + 1)  
        ax.set_ylim(min(y_value) - 1, max(y_value) + 1)
        
        # Update annotation text and position for Line 1
        annot1.set_text(f"{y_value[-1]:.4f}")  # Numeric value for Line 1
        annot1.xy = (x_games[-1], y_value[-1])  # Place annotation at the end of Line 1

        # Update annotation text and position for Line 2
        annot2.set_text(f"{y_avg[-1]:.4f}")  # Numeric value for Line 2
        annot2.xy = (x_games[-1], y_avg[-1])  # Place annotation at the end of Line 2

        plt.pause(0.1)  # Pausa para atualização visual 


    def update_plot_score(self, score):        
        self.update_plot_graph(
            score, self.ax_score, self.annot1_score, self.annot2_score, 
            self.x_games_score, self.y_score, self.mov_avg_score, 
            self.line_score, self.line_mov_avg_score
        )

    def update_plot_steps_death(self, steps_death):        
        self.update_plot_graph(
            steps_death, self.ax_death, self.annot1_death, self.annot2_death, 
            self.x_games_death, self.y_death, self.mov_avg_death, 
            self.line_death, self.line_mov_avg_death
        )

    def update_plot_steps_food(self, steps_food):   
        self.update_plot_graph(
            steps_food, self.ax_step_food, self.annot1_step_food, self.annot2_step_food, 
            self.x_games_step_food, self.y_step_food, self.mov_avg_step_food, 
            self.line_step_food, self.line_mov_avg_step_food
        )     
    