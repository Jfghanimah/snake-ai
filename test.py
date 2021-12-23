from environment import Snake

snake = Snake(8,8)

while snake.is_dead == False:
    direction = int(input("Move: "))
    snake.step(direction)
    snake.print()