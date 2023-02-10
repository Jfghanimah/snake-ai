import numpy as np

# This is the Snake class, which represents a Snake game environment


class Snake:
    def __init__(self, map_height, map_width):
        # Initialize the map dimensions and any other settings
        self.map_height = map_height
        self.map_width = map_width
        self.reset()

    def reset(self):
        # Reset the game state
        # All coordinates should be in the y,z format starting with 0,0 at the top left
        self.map = np.zeros((self.map_height, self.map_width), dtype=int)
        self.is_dead = False
        self.is_won = False
        self.steps = 0
        self.length = 2
        self.head_pos = (1, 1)
        self.tail_pos = (1, 0)
        self.map[self.head_pos] = self.length
        self.map[self.tail_pos] = 1

        self.food = (2, 4)  # should probably be somewhere else
        self.map[self.food] = -1

    def print(self):
        # Print the current game state
        print(f"------------Step: {self.steps}-----------")
        print(f"Head pos: {self.head_pos}")
        print(f"Tail pos: {self.tail_pos}")
        print(f"Food pos: {self.food}")
        print(f"Length: {self.length}")
        print(self.map)

    def spawn_food(self):

        if self.length == self.map_height * self.map_width:
            self.is_won = True
            print(f"Congratulations you won in {self.steps} steps!")
            return
        
        # Spawn apple randomly that isnt where the snake currently is
        # Find all unoccupied poss
        unoccupied = np.where(self.map == 0)
        # Choose a random unoccupied pos
        idx = np.random.randint(0, len(unoccupied[0]))
        self.food = (unoccupied[0][idx], unoccupied[1][idx])
        self.map[self.food] = -1

    def step(self, action):
        # Take a step in the environment
        if self.is_dead or self.is_won:
            return

        # Update head pos based on action
        if action == 0:  # Up
            new_pos = (self.head_pos[0]-1, self.head_pos[1])
        elif action == 1:  # Down
            new_pos = (self.head_pos[0]+1, self.head_pos[1])
        elif action == 2:  # Left
            new_pos = (self.head_pos[0], self.head_pos[1]-1)
        elif action == 3:  # Right
            new_pos = (self.head_pos[0], self.head_pos[1]+1)
        else:
            print("Not a valid action")
            return

        
        self.steps += 1

        if self.steps % 100 == 0:
            print(f"Snake Steps: {self.steps}")
   
        out_of_bounds = (new_pos[0] < 0) | (new_pos[0] >= self.map_width) | (new_pos[1] < 0) | (new_pos[1] >= self.map_height)
        if out_of_bounds:
            self.is_dead = True
            print(new_pos)
            print("Snake went out of bounds...")
            return "died"
        elif self.map[new_pos] > 1: # we colided
            self.is_dead = True
            print(new_pos)
            print("Snake ate itself...")
            return "died"
        elif new_pos == self.food:
            self.length += 1
            self.head_pos = new_pos
            self.map[new_pos] = self.length
            self.spawn_food()
            #print(f"Got food! Len: {self.length}")
            return "ate"
        else: # just a move
            self.head_pos = new_pos
            # Decrease all positive numbers by 1
            self.map[self.map > 0] -= 1
            self.map[self.head_pos] = self.length
            self.tail_pos = tuple(np.argwhere(self.map == 1)[0])
