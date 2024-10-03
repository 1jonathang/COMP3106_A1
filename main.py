import pandas as pd
import numpy as np
from collections import deque
import heapq

class State:
    """
    creating a class to represent the state our agent is currently at
    """
    def __init__(self, pos, cur_keys, doors_opened, keys_collected):
        self.pos = pos
        self.cur_keys = cur_keys
        self.doors_opened = frozenset(doors_opened) # using `frozenset` here because sets are unhashable, and we need this class to be hashable
        self.keys_collected = frozenset(keys_collected)

    def __eq__(self, other):
        """
        function check if two states are equal to eachother
        """
        return (
            self.pos == other.pos
            and self.cur_keys == other.cur_keys
            and self.doors_opened == other.doors_opened
            and self.keys_collected == other.keys_collected
        )

    def __hash__(self):
        """
         generates hash values for a state based off of its attributes
         we need to do this since we'll be storing states in a dictionary later
        """
        return hash((self.pos, self.cur_keys, self.doors_opened, self.keys_collected))
     
   
def heuristic(pos, goal):
   """
   our heuristic function for A* search utilizing the Manhattan distances
   """
   return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def find_start_goal(df):
   """
   Args:
       df (pandas dataframe): our environment grid

   Returns:
       (tuple), (tuple): coordinates of start and goal cells
   """
   # dimensions of grid
   m, n = df.shape

   # looping through grid to find start and goal cells
   start, goal = None, None
   for i in range(m):
      for j in range(n):
         value = df.iloc[i, j]
         if value == 'S':
            start = (i, j)
         if value == 'G':
            goal = (i, j)
   return start, goal

def pathfinding(filepath):
    # reading the csv file with pandas, need to remove header, and all data is of type str
    df = pd.read_csv(filepath, header=None, dtype=str)
    # getting the dimensions of the grid
    m, n = df.shape

    # finding the start and goal positions
    start, goal = find_start_goal(df)
    # if start or goal cells don't exist then break and return -1
    if start is None or goal is None:
       return -1
   
    # initializing our start state with the start position, and everything 0 and empty
    start_state = State(start, 0, frozenset(), frozenset())
    # our priority queue
    heap = []
    # need to use a counter so all heap elements are unique, otherwise error will occur
    counter = 0
    # pushing initial start state to our priority queue, its going to consist of [heuristic value, f-score, counter, state]
    heapq.heappush(heap, (heuristic(start, goal), 0, counter, start_state))
    counter += 1

    # g-score, cost of the path from the start cell to current cell
    g_score = {start_state: 0}
    # this is where we're gonna reconstruct the optimal path after finding it
    came_from = {}
    # set to keep track of cells we've already explored
    explored = set()
    # keeping track of the number of states we explored
    states_explored = 0

    # the entire A* search logic
    while heap:
        # popping off the priority queue
        _, _, _, current_state = heapq.heappop(heap)
        # incrementing states explored
        states_explored += 1

        # checking if we reached the goal
        if current_state.pos == goal:
            # reconstructing our optimal path
            optimal_path = []
            state = current_state
            while state in came_from:
                optimal_path.append(state.pos)
                state = came_from[state]
            # adding our start position, then reversing it because we just constructed it in reverse order
            optimal_path.append(start_state.pos)
            optimal_path.reverse()
            # this is our total cost to reach the goal
            optimal_cost = g_score[current_state]
            return optimal_path, optimal_cost, states_explored
        if current_state in explored:
            continue
        explored.add(current_state)

        # iterating over neighbors of current cell
        row, col = current_state.pos
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d_row, d_col in dirs:
            new_row, new_col = row + d_row, col + d_col
            # checking if new cell is within bounds
            if 0 <= new_row < m and 0 <= new_col < n:
                cell = df.iloc[new_row, new_col]
                new_pos = (new_row, new_col)
                # initializing class variables
                new_keys_in_possession = current_state.cur_keys
                new_doors_opened = set(current_state.doors_opened)
                new_keys_collected = set(current_state.keys_collected)
                # creating a flag to make sure movement to this cell can't happen
                can_move = False

                # determining if we can move to neighboring cell based off what their type is
                if cell in ("O", "S", "G"):
                    can_move = True
                elif cell == "K":
                    can_move = True
                    # if the neighbor cell is a key and we haven't collected it yet, then collect it
                    if new_pos not in current_state.keys_collected:
                        new_keys_in_possession += 1
                        new_keys_collected.add(new_pos)
                elif cell == "D":
                    # need to check if we've opened this door already
                    if new_pos in current_state.doors_opened:
                        can_move = True
                    elif current_state.cur_keys >= 1:
                        # if this door is closed, "open it" (decrementing our number of keys by 1)
                        can_move = True
                        new_keys_in_possession -= 1
                        new_doors_opened.add(new_pos)
                    else:
                        # can't open door because we don't have enough keys
                        can_move = False
                else:
                    can_move = False

                # if we can move to this neighbor cell, then create a new state for it
                if can_move:
                    neighbor_state = State(
                        new_pos,
                        new_keys_in_possession,
                        frozenset(new_doors_opened),
                        frozenset(new_keys_collected),
                    )
                    # calculating new g-score, cost to reach neighbor from start state. basically + 1 of g-score from previous cell
                    new_g_score = g_score[current_state] + 1
                    if neighbor_state in explored:
                        continue
                    # checking if this path to the neighbor is better than any previous one
                    if (
                        neighbor_state not in g_score
                        or new_g_score < g_score[neighbor_state]
                    ):
                        came_from[neighbor_state] = current_state
                        g_score[neighbor_state] = new_g_score
                        # f(n) = g(n) + h(n)
                        f_score = new_g_score + heuristic(new_pos, goal)
                        heapq.heappush(heap, (f_score, new_g_score, counter, neighbor_state))
                        counter += 1
    # edge case where not possible to reach goal
    return [], -1, states_explored
 
path, cost, states = pathfinding("data/E0/grid.csv")
print("\t path: ", path, " \n \t cost: ", cost, "\n \t states: ", states)
