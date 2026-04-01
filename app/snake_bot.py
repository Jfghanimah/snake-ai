import heapq
import numpy as np


class SnakeBot:
    def __init__(self, snake_env):
        # Initialize the map dimensions and any other settings
        self.snake_env = snake_env
        self.sandbox = snake_env # I want this to be the sandbox for testing our pathfinder 
        
        # Need to create a way to mark which squares the tail can see - for safety
        self.tailmap = np.zeros((self.snake_env.map_height, self.snake_env.map_width), dtype=int)
            

    def best_move(self):
        # The logic behind here should be this to begin with
        # - Maintain direct path to tail at all times
        # - If possible go towards food, otherwise tail

        if self.snake_env.length == self.snake_env.map_width * self.snake_env.map_height - 1:
            if self.get_manhattan_distance(self.snake_env.head_pos, self.snake_env.food) == 1:
                return self.get_move(self.snake_env.head_pos, self.snake_env.food)

        self.update_tail_map() # Maybe call this at every step in the A* for more accurate results?
        path = self.path_find(self.snake_env.food)
        if len(path) == 0:
            #print("No path for food, tail instead!") This one actually needs to stall instead of find the best path
            path = self.worst_path_find(self.snake_env.tail_pos)
            
        return path[0]


    def update_tail_map(self):
        #returns a map that says whether or not a square has vision of the snakes tail.
        # does this by BFS searching starting from the tail and marking off on the tailmap
        # set tiles to 1 if tail can path to that tile 0 otherwise
        self.tailmap = np.zeros((self.snake_env.map_height, self.snake_env.map_width), dtype=int)
        future_tail_pos = tuple(np.argwhere(self.snake_env.map == 2)[0])
        queue = [self.snake_env.tail_pos]
        visited = set()
        while queue:
            #print(f"updating tail map, queue length: {len(queue)}")
            pos = queue.pop(0)
            self.tailmap[pos] = 1
            visited.add(pos)
            neighbors = self.generate_neighbors(pos)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)


    def path_find(self, goal):
        # Create a priority queue for A*
        #print(f"Trying to path toward {goal}")
        queue = []
        heapq.heapify(queue)

        visited = set()
        heapq.heappush(queue, (0, id(self.snake_env.head_pos), self.snake_env.head_pos, []))
        while queue:
            priority, id_node, node, path = heapq.heappop(queue)
            #print(f"path finding... path queue length: {len(queue)}, heapq: {priority, node, path, goal}")
            if node == goal:
                return path
            
            neighbors = self.generate_neighbors(node, True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [self.get_move(node, neighbor)]
                    new_priority = len(new_path) + self.get_priority(neighbor, goal)
                    visited.add(neighbor)
                    heapq.heappush(queue, (new_priority, id(neighbor), neighbor, new_path))

        #print(f"no path from{self.snake_env.head_pos} to {goal}")
        return [] # No path


    def worst_path_find(self, goal):
        # Create a priority queue for the worst path finding algorithm
        queue = []
        heapq.heapify(queue)

        visited = set()
        # Push the starting node onto the queue with a high priority value
        heapq.heappush(queue, (float('inf'), id(self.snake_env.head_pos), self.snake_env.head_pos, []))
        while queue:
            # Pop the node with the highest priority value from the queue
            priority, id_node, node, path = heapq.heappop(queue)
            if node == goal:
                return path

            neighbors = self.generate_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [self.get_move(node, neighbor)]
                    # Calculate the priority value for the neighbor as the negative of the length of the path plus the distance to the goal
                    new_priority = -(len(new_path) + self.get_manhattan_distance(neighbor, goal))
                    visited.add(node)
                    heapq.heappush(queue, (new_priority, id(neighbor), neighbor, new_path))
        return [] # No path


    def get_priority(self, node, goal, worse=False):
        # uses hueristics to calculate the priority (lower is better) of the nodey
        regular_distance = self.get_manhattan_distance(node, goal)
        priority_factor = 0

        # mid game idea
        if self.snake_env.length > (self.snake_env.map_width**2)/5 or True:
            # Reward path with higher snake walls (lower neighbors) to stay nice and neat
            # Calculate the number of neighbors the node has
            neighbors = self.generate_neighbors(node, check_tail_vis=True)
            num_neighbors = len(neighbors)
            priority_factor = num_neighbors*10

            # We still want to prioritize these good organizational moves when calculating the worst path
            if worse:
                priority_factor *= -1

        #late game idea??
        

        # smaller is better and will be prioritized
        return regular_distance + priority_factor


    def get_manhattan_distance(self, node, goal):
        # Calculate the Manhattan distance between the node and the goal
        y1, x1 = node
        y2, x2 = goal
        distance = abs(x1 - x2) + abs(y1 - y2)

        # If we have to turn at any point to reach the goal,
        # add an additional cost to the distance
        if x1 != x2 and y1 != y2:
            distance += 1

        return distance


    def generate_neighbors(self, node, check_tail_vis=False):
        # Generate the neighbors of the node by trying to move up, down, left, and right
        y, x = node
        neighbors = [(y, x-1), (y, x+1), (y-1, x), (y+1, x)]
        neighbors = [n for n in neighbors if self.is_valid_move(n, check_tail_vis)]
        return neighbors


    def is_valid_move(self, neighbor, check_tail_vis):
        # Check if the neighbor is outside the map
        if neighbor[0] < 0 or neighbor[0] >= self.snake_env.map_width or neighbor[1] < 0 or neighbor[1] >= self.snake_env.map_height:
            return False
        # Or the snake itself
        if self.snake_env.map[neighbor] > 1:  # we can still replace tail :)
            return False
        
        if check_tail_vis: # We only want to run this for the A* food search not the BFS vision maps
            # Or doesnt see the tail?
            if self.tailmap[neighbor] == 0:
                return False
        return True


    def get_move(self, node, neighbor):
        # translates the node and neighbor to an actual direction
        y1, x1 = node
        y2, x2 = neighbor
        if x1 < x2:
            return 3 #right
        elif x1 > x2:
            return 2 #left
        elif y1 < y2:
            return 1 #down
        elif y1 > y2:
            return 0 #up

