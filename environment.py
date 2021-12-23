import numpy as np


class Snake:
    def __init__(self, map_height, map_width):
        self.map_height = map_height
        self.map_width = map_width
        self.reset()


    def reset(self):
        self.map = np.zeros((self.map_height, self.map_width))
        self.is_dead = False     
        self.steps = 0
        self.length = 1
        self.head_position = (0,0)
        self.food = (5,5)
        
        self.map[self.head_position] = self.length
        self.map[self.food] = -1


    def show_state(self):
        print(f"----------{self.steps}----------")
        print(self.map)


    def spawn_food(self):
        pass


    def step(self, action):
        if self.is_dead:
            return
        
        self.steps +=1 
        cur_pos = self.head_position
       
        if action == 0: # Up        
            self.head_position = (cur_pos[0], cur_pos[1]-1)
        elif action == 1: # Down
            self.head_position = (cur_pos[0], cur_pos[1]+1)        
        elif action == 2: # Left
            self.head_position = (cur_pos[0]-1, cur_pos[1])        
        elif action == 3: # Right
            self.head_position = (cur_pos[0]+1, cur_pos[1])

        # Check if died
        if self.map[self.head_position] > 0:
            self.is_dead = True
            return

        # Check if food
        if self.head_position == self.food:
            self.length += 1
            self.map[self.head_position] = self.length
        else:
            # Decrease all positive numbers by 1
            # Set new head on map
            pass
        
        self.spawn_food()