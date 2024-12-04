import numpy as np
import pygame
import random


class Snake:
    def __init__(self, block_size, screen_width, screen_height):
        self.block_size = block_size
        self.body = np.array([
            [
                random.randrange(0, screen_width//block_size - 1, 1),
                random.randrange(0, screen_height//block_size - 1, 1),
            ]
        ])
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.length = 1
        self.direction = None
        self.directions = {
            "LEFT": [-1, 0],
            "RIGHT": [1, 0],
            "UP": [0, -1],
            "DOWN": [0, 1],
        }
        self.all_free_positions = self.get_all_free_positions()
        

    def get_all_free_positions(self):
        all_free_positions = np.ones(
            (self.screen_width // self.block_size, 
             self.screen_height // self.block_size), 
            dtype=bool
        )
        all_free_positions[self.body[:, 0], self.body[:, 1]] = 0
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
            self.get_all_free_positions()

            # Maintain length
            if len(self.body) > self.length:
                self.body = np.delete(self.body, 0, 0)

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
        self.position[0] = chosen_index[0]
        self.position[1] = chosen_index[1]
    
    def draw(self, screen, color):
        pygame.draw.rect(
            screen,
            color,
            [self.position[0]*self.block_size, 
             self.position[1]*self.block_size, 
             self.block_size, self.block_size],
        )


class Game:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.block_size = 40
        self.speed = 15

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
        elif self.game_over:
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

        # Draw the "Retry" button
        pygame.draw.rect(self.screen, self.colors["button"], [350, 300, 100, 50])
        retry_text = self.font.render("Retry", True, self.colors["button_text"])
        self.screen.blit(retry_text, [375, 310])

    def restart_game(self):
        self.snake = Snake(self.block_size, self.width, self.height)
        self.food = Food(self.snake)
        self.score = 0
        self.game_over = False
        self.win = False

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.speed)

        pygame.quit()


# Run the game
if __name__ == "__main__":
    Game().run()
