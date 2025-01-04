import numpy as np
import pygame
from utils.agent import Agent
from utils.entities import Snake, Food, Statistics
from collections import deque
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import mlflow


class Game:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.block_size = 100
        self.speed = 100

        self.colors = {
            "background": (0, 0, 0),
            "snake": (0, 0, 255),
            "food": (0, 255, 0),
            "score": (255, 255, 255),
            "button": (255, 0, 0),
            "button_text": (255, 255, 255),
        }

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("bahnschrift", 25)
        self.snake = Snake(self.block_size, self.width, self.height)
        self.food = Food(self.snake)
        self.statistics = Statistics()

        self.agent = Agent(self.snake, self.food, self.statistics)
        self.last_two_actions = deque(maxlen=2)

        self.running = True
        self.game_over = False
        self.win = False
        self.score = 0
        self.steps_death = 0
        self.steps_food = 0

        self.total_spaces = (self.width // self.block_size) * (self.height // self.block_size)


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and not (self.game_over or self.win):
                if event.key == pygame.K_LEFT and (self.snake.direction != "RIGHT" or self.snake.length == 1):
                    self.snake.direction = "LEFT"
                elif event.key == pygame.K_RIGHT and (self.snake.direction != "LEFT" or self.snake.length == 1):
                    self.snake.direction = "RIGHT"
                elif event.key == pygame.K_UP and (self.snake.direction != "DOWN" or self.snake.length == 1):
                    self.snake.direction = "UP"
                elif event.key == pygame.K_DOWN and (self.snake.direction != "UP" or self.snake.length == 1):
                    self.snake.direction = "DOWN"
            elif event.type == pygame.MOUSEBUTTONDOWN and (self.game_over or self.win):
                # Check if "Retry" button is clicked
                mouse_pos = pygame.mouse.get_pos()
                if 350 <= mouse_pos[0] <= 450 and 300 <= mouse_pos[1] <= 350:
                    self.restart_game()
            elif event.type == pygame.KEYDOWN and (self.game_over or self.win):
                # Check if Enter key is pressed to restart the game
                if event.key == pygame.K_RETURN:
                    self.restart_game()


    def update(self):
        if not (self.game_over or self.win):
            self.snake.move()
            if np.all(self.snake.body[-1] == self.food.position):
                self.snake.grow()
                self.food.generate_position()
                self.score += 1

            # Check for collision
            elif self.snake.check_collision():
                self.game_over = True

            # Check if the snake fills the entire grid
            if self.snake.length == self.total_spaces:
                self.win = True

    def draw(self):
        self.screen.fill(self.colors["background"])
        if not (self.game_over or self.win):
            self.snake.draw(self.screen, self.colors["snake"])
            self.food.draw(self.screen, self.colors["food"])
            self.show_score()
        elif self.agent and self.game_over:
            self.show_game_over()
        elif self.win:
            self.show_win()
        pygame.display.update()

    def show_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.colors["score"])
        self.screen.blit(score_text, [10, 10])

    def show_game_over(self):
        # Display game over text
        game_over_text = self.font.render(
            f"Game Over! Final Score: {self.score}", True, self.colors["score"]
        )
        self.screen.blit(game_over_text, [self.width // 2 - 150, self.height // 2 - 50])

        # Draw the "Retry" button
        pygame.draw.rect(self.screen, self.colors["button"], [350, 300, 100, 50])
        retry_text = self.font.render("Retry", True, self.colors["button_text"])
        self.screen.blit(retry_text, [375, 310])

    def show_win(self):
        # Display win text
        win_text = self.font.render(
            f"Congratulations! You Won! Final Score: {self.score}", True, self.colors["score"]
        )
        self.screen.blit(win_text, [self.width // 2 - 200, self.height // 2 - 50])

        # Draw the "Play Again" button
        pygame.draw.rect(self.screen, self.colors["button"], [350, 300, 100, 50])
        retry_text = self.font.render("Play Again", True, self.colors["button_text"])
        self.screen.blit(retry_text, [375, 310])

    def restart_game(self):
        self.snake = Snake(self.block_size, self.width, self.height)
        self.food = Food(self.snake)
        if self.agent:
            self.agent.snake = self.snake
            self.agent.food = self.food
    
        self.score = 0
        self.game_over = False
        self.win = False

    
    def get_reward(self):
        if self.game_over:
            reward = self.agent.R_GAME_OVER
            # Plot Score Evolution
            self.statistics.update_plot_score(self.score)
            self.statistics.update_plot_steps_death(self.steps_death)
            self.steps_death = 0
            self.restart_game()

        elif len(self.snake.body) < self.snake.length: # Food eaten
            reward = self.agent.R_EAT_FOOD
            self.statistics.update_plot_steps_food(self.steps_food)
            self.steps_food = 0
        
        else:
            reward = self.agent.R_NEUTRAL
            if len(self.last_two_actions) == 2:
                if "RIGHT" in self.last_two_actions and "LEFT" in self.last_two_actions\
                   or "UP" in self.last_two_actions and "DOWN" in self.last_two_actions:
                   reward = self.agent.R_OPPOSITE
        return reward

    def get_real_direction(self, relative_action):
        directions = list(self.snake.directions.keys())
        idx_current = directions.index(self.snake.direction)
        if relative_action == "RIGHT":
            idx_current += 1
        elif relative_action == "LEFT":
            idx_current -= 1
        idx_current = idx_current % 4
        return directions[idx_current] 


    def press_keyboard(self, idx_action):
        action_rel = self.agent.actions[idx_action]
        action = self.get_real_direction(action_rel)   

        if action == "UP":
            key = pygame.K_UP
        elif action == "DOWN":
            key = pygame.K_DOWN
        elif action == "LEFT":
            key = pygame.K_LEFT
        elif action == "RIGHT":
            key = pygame.K_RIGHT

        simulated_event = pygame.event.Event(pygame.KEYDOWN, key=key)
        pygame.event.post(simulated_event)
        self.steps_death += 1
        self.steps_food += 1


    def act_agent(self):
        if not self.started:
            idx_next_action = self.agent.agent_start()
            self.press_keyboard(idx_next_action)
            self.started = True
            
            # Save last action
            self.last_two_actions.append(self.agent.actions[idx_next_action])
            return

        reward = self.get_reward()

        # Take current states, update agent, generate next action, save (s, a).
        if reward == self.agent.R_GAME_OVER:
            self.agent.agent_end(reward)
            idx_next_action = self.agent.agent_start()
        else:        
            idx_next_action = self.agent.agent_step(reward)

        # Press in the keyboard
        self.press_keyboard(idx_next_action)

        # Save last action
        self.last_two_actions.append(self.agent.actions[idx_next_action])


    def ui_save_statistics(self):
        # Criação do botão de confirmação
        screen = pygame.display.set_mode((500, 300))
        font = pygame.font.Font(None, 36)
        
        yes_button = pygame.Rect(100, 150, 120, 50)
        no_button = pygame.Rect(280, 150, 120, 50)
        
        running_dialog = True
        while running_dialog:
            screen.fill((30, 30, 30))
            question = font.render("Save statistics in Mlflow?", True, (255, 255, 255))
            screen.blit(question, (50, 50))

            pygame.draw.rect(screen, (0, 200, 0), yes_button)
            pygame.draw.rect(screen, (200, 0, 0), no_button)
            
            yes_text = font.render("Yes", True, (255, 255, 255))
            no_text = font.render("No", True, (255, 255, 255))
            screen.blit(yes_text, (yes_button.x + 40, yes_button.y + 10))
            screen.blit(no_text, (no_button.x + 40, no_button.y + 10))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_dialog = False
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if yes_button.collidepoint(event.pos):
                        print("Saving statistics in Mlflow")
                        running_dialog = False
                    elif no_button.collidepoint(event.pos):
                        print("Discarding statistics in Mlflow")
                        running_dialog = False


    def set_mlflow(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("snake-rl")
        mlflow.start_run()

        ## Save parameters
        # Game
        mlflow.log_param("screen_width", self.width)
        mlflow.log_param("screen_height", self.height)
        mlflow.log_param("block_size", self.block_size)
        mlflow.log_param("speed", self.speed)

        # Agent
        mlflow.log_param("actions", self.agent.actions)
        mlflow.log_param("states_shape", self.agent.last_state.shape)
        mlflow.log_param("gamma", self.agent.gamma)
        mlflow.log_param("learning_rate", self.agent.learning_rate)
        mlflow.log_param("amsgrad", self.agent.amsgrad)
        mlflow.log_param("tau_softmax_policy", self.agent.tau)   
        mlflow.log_param("memory_batch_size", self.agent.batch_size)
        mlflow.log_param("memory_size", self.agent.memory_size)
        mlflow.log_param("n_replays", self.agent.n_replays)

        # Rewards
        mlflow.log_param("R_GAME_OVER", self.agent.R_GAME_OVER)
        mlflow.log_param("R_EAT_FOOD", self.agent.R_EAT_FOOD)
        mlflow.log_param("R_NEUTRAL", self.agent.R_NEUTRAL)
        mlflow.log_param("R_OPPOSITE", self.agent.R_OPPOSITE)



    def ui_save_statistics(self):
        root = tk.Tk()
        root.withdraw()  # Esconde a janela principal

        option = messagebox.askyesno(
            "Save Statistics", 
            "Do you want to save the statistics in Mlflow?"
        )
        
        if option:
            print("Saving statistics in Mlflow")
            self.set_mlflow()
            self.statistics.save_statistics_mlflow()
            mlflow.log_artifact(__file__)
            mlflow.log_artifact("./utils/agent.py")
            mlflow.log_artifact("./utils/entities.py")

            mlflow.end_run()

        else:
            print("Discarding statistics")


    def run(self):
        self.started = False
        #self.statistics.init_prob_chart(self.agent.actions)
        #self.statistics.init_act_values_chart(self.agent.actions)
        self.statistics.init_plot_score()
        self.statistics.init_plot_steps_death()
        self.statistics.init_plot_steps_food()
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.act_agent()

            self.clock.tick(self.speed)

        plt.close("all")
        self.ui_save_statistics()
        pygame.quit()



# Run the game
if __name__ == "__main__":
    Game().run()
