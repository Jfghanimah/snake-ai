import numpy as np

class Snake:
    def __init__(self, map_height, map_width):
        if map_height < 3 or map_width < 3:
            raise ValueError("Snake board must be at least 3x3.")

        self.map_height = map_height
        self.map_width = map_width
        self.reset()

    def reset(self):
        """Reset the environment to the starting state."""
        self.map = np.zeros((self.map_height, self.map_width), dtype=int)
        self.is_dead = False
        self.is_won = False
        self.steps = 0
        self.length = 2
        self.head_pos = (1, 1)
        self.tail_pos = (1, 0)
        self.map[self.head_pos] = self.length
        self.map[self.tail_pos] = 1

        self.spawn_food()

    def print(self):
        print(f"------------Step: {self.steps}-----------")
        print(f"Head pos: {self.head_pos}")
        print(f"Tail pos: {self.tail_pos}")
        print(f"Food pos: {self.food}")
        print(f"Length: {self.length}")
        print(self.map)

    def spawn_food(self):
        if self.length == self.map_height * self.map_width:
            self.is_won = True
            return

        unoccupied = np.where(self.map == 0)
        idx = np.random.randint(0, len(unoccupied[0]))
        self.food = (unoccupied[0][idx], unoccupied[1][idx])
        self.map[self.food] = -1

    def step(self, action):
        if self.is_dead or self.is_won:
            return

        if action == 0:
            new_pos = (self.head_pos[0] - 1, self.head_pos[1])
        elif action == 1:
            new_pos = (self.head_pos[0] + 1, self.head_pos[1])
        elif action == 2:
            new_pos = (self.head_pos[0], self.head_pos[1] - 1)
        elif action == 3:
            new_pos = (self.head_pos[0], self.head_pos[1] + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1

        out_of_bounds = (
            new_pos[0] < 0
            or new_pos[0] >= self.map_height
            or new_pos[1] < 0
            or new_pos[1] >= self.map_width
        )
        if out_of_bounds:
            self.is_dead = True
            return "died"
        elif self.map[new_pos] > 1:
            self.is_dead = True
            return "died"
        elif new_pos == self.food:
            self.length += 1
            self.head_pos = new_pos
            self.map[new_pos] = self.length
            self.spawn_food()
            return "ate"
        else:
            self.head_pos = new_pos
            self.map[self.map > 0] -= 1
            self.map[self.head_pos] = self.length
            self.tail_pos = tuple(np.argwhere(self.map == 1)[0])
            return "moved"
