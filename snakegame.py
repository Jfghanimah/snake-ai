import pygame
from environment import Snake
from snake_bot import SnakeBot
import sys

snake_env = Snake(18, 18)  # Create a 10x10 snake env map
snake_bot = SnakeBot(snake_env)

pygame.init()
window_size = (snake_env.map_width * 30, snake_env.map_height * 30)  # Set the window size to match the map size
screen = pygame.display.set_mode(window_size)

# Set up a font to use for rendering the numbers
font = pygame.font.Font(None, 24)

# Set up variables to track the current direction and whether the snake is moving
direction = None
moving = False
bot_mode = False
next_food = False
start_len = snake_env.length

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # A key was pressed
            if event.key == pygame.K_UP:
                direction = 0
                moving = True
            elif event.key == pygame.K_DOWN:
                direction = 1
                moving = True
            elif event.key == pygame.K_LEFT:
                direction = 2
                moving = True
            elif event.key == pygame.K_RIGHT:
                direction = 3
                moving = True
            elif event.key == pygame.K_q:
                direction = snake_bot.best_move()
                moving = True
            elif event.key == pygame.K_w:
                bot_mode = True
            elif event.key == pygame.K_e:
                bot_mode = False
            elif event.key == pygame.K_a:
                next_food = True
                start_len = snake_env.length

    # Update the game state
    if moving or bot_mode or next_food:
        if bot_mode:
            direction = snake_bot.best_move()
            snake_env.step(direction)
            moving = False
        elif next_food:
            if start_len == snake_env.length:
                direction = snake_bot.best_move()
                result = snake_env.step(direction)
                moving = False
                if result == 'ate':
                    next_food = False
        else:
            snake_env.step(direction)
            moving = False

    # Render game objects
    screen.fill((0, 0, 0))
    # Draw the snake
    for i in range(snake_env.map_height):
        for j in range(snake_env.map_width):
            if snake_env.map[i][j] > 0:
                pygame.draw.circle(screen, (50, 100, 50), (j*30+15, i*30+15), 15)
                # Render the number as text
                number = font.render(str(snake_env.map[i][j]), True, (255, 255, 255))
                # Blit the text onto the screen
                screen.blit(number, (j*30+7, i*30+7))

    # Draw the head
    head_pos = snake_env.head_pos
    pygame.draw.circle(screen, (50, 150, 50), (head_pos[1]*30+15, head_pos[0]*30+15), 15)
    # Draw the food
    food_pos = snake_env.food
    pygame.draw.circle(screen, (255, 0, 0), (food_pos[1]*30+15, food_pos[0]*30+15), 15)


    pygame.display.flip()