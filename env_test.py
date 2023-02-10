from environment import Snake
from snake_bot import SnakeBot


snake = Snake(8,8)

#bot = SnakeBot(snake_env=snake)
key_mapping = {'w': 0, 's': 1, 'a': 2, 'd': 3}

while snake.is_dead == False:
    #print(bot.best_move())
    direction = input("Move: ")
    if direction in key_mapping:
        snake.step(key_mapping[direction])
        snake.print()