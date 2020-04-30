# AI for Logistics - Robots in a warehouse
# Importing the libraries
import numpy as np

# Setting the parameters gamma and alpha for the Q-Learning
gamma = 0.75
alpha = 0.9

# PART 1 - BUILDING THE ENVIRONMENT
# Defining the states
location_to_state = {"A": 0,
                     "B": 1,
                     "C": 2,
                     "D": 3,
                     "E": 4,
                     "F": 5,
                     "G": 6,
                     "H": 7,
                     "I": 8,
                     "J": 9,
                     "K": 10,
                     "L": 11}

# Defining the actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Defining the rewards
R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], ])

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING

# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}


# Making the final function that will return the optimal route
def route(starting_location, ending_location):
    r_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    r_new[ending_state, ending_state] = 1000
    q = np.array(np.zeros([12, 12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)  # 1
        playable_actions = []
        for j in range(12):  # 2
            if r_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)  # 3
        td = r_new[current_state, next_state] + gamma * q[next_state, np.argmax(q[next_state,])] - q[
            current_state, next_state]  # 4
        q[current_state, next_state] = q[current_state, next_state] + alpha * td  # 5
    ro = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_s = np.argmax(q[starting_state,])
        next_location = state_to_location[next_s]
        ro.append(next_location)
        starting_location = next_location
    return ro


# PART 3 - GOING INTO PRODUCTION

# Making the final function that returns the optimal route
def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]


# Printing the final route
print('Route:')
print(best_route('E', 'K', 'G'))
