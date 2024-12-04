import numpy as np
import pygame
from utils.agent import Agent
from utils.entities import Snake, Food, Statistics
from collections import deque


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
            if self.snake.check_collision():
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
            if self.statistics.n_gameovers == 0:
                self.statistics.init_plot_score()
            self.statistics.update_plot_score(self.score)
            
            self.restart_game()

        elif len(self.snake.body) < self.snake.length: # Food eaten
            reward = self.agent.R_EAT_FOOD
        
        else:
            reward = self.agent.R_NEUTRAL
            if len(self.last_two_actions) == 2:
                if "RIGHT" in self.last_two_actions and "LEFT" in self.last_two_actions\
                   or "UP" in self.last_two_actions and "DOWN" in self.last_two_actions:
                   reward = self.agent.R_OPPOSITE
        return reward

    def press_keyboard(self, idx_action):
        action = self.agent.actions[idx_action]

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

    def act_agent(self):
        if not self.started:
            self.press_keyboard(self.agent.agent_start())
            self.started = True
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



    def run(self):
        self.started = False
        self.statistics.init_prob_chart(self.agent.actions)
        self.statistics.init_act_values_chart(self.agent.actions)
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.act_agent()

            self.clock.tick(self.speed)

        pygame.quit()


# Run the game
if __name__ == "__main__":
    Game().run()
