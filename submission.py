from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time # for time management in AgentMinimax

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    packages = env.packages
    
    # Feature 1: Credit, for a greedy agent, we need to be maximizing our own credit.
    f_credit = robot.credit
    
    # Feature 2: Battery, we want to have more battery to be able to move more.
    f_battery = robot.battery
    
    # Feature 3: Distance to objective (next package or destination)
    dist = 0
    if robot.package is not None: # We have a package, objective is to go to destination
        dest = robot.package.destination
        dist = manhattan_distance(robot.position, dest)
    else: # We need a package, objective is to go to nearest package
        if not packages:
            dist = 0 # No packages left
        else:
            # Find min distance to any package
            dists = [manhattan_distance(robot.position, p.position) for p in packages]
            dist = min(dists) if dists else 0

    # idk, these seem fine from tournament testing
    w_credit = 10.0
    w_battery = 1.0
    w_dist = 2.0
    
    # We subtract distance feature because we want to minimize it
    heuristic_value = (w_credit * f_credit) + (w_battery * f_battery) - (w_dist * dist)
    
    return heuristic_value

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def utility_heuristic(self, env, robot_id):
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)

    # implmemented with iterative deepening
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # we should record the start time to manage time limits, later we will check elapsed time, 
        # and stop execution if we exceed it
        start_time = time.time()

        # sometimes, the agent "crashes" the game right at the time limit, so after some research, 
        # we should add a small "time buffer" to avoid that. so we define
        # "time_buffer" as the small time we stop the search before the actual time limit
        time_buffer = 0.05 

        # the id of the other agent just to make life easier (and code readable)
        other_agent_id = (agent_id + 1) % 2
        
        # this is a custom exception to handle timeout; we raise it when we exceed time limit
        class TimeoutException(Exception):
            pass # pass here because we dont have any special handling or info

        # we check the time elapsed, if we exceed the time limit (minus buffer), we raise TimeoutException
        def check_timeout():
            if time.time() - start_time >= time_limit - time_buffer:
                raise TimeoutException()

        # Minimax: Min Value (Opponent's Turn)
        def min_value(curr_env, depth):
            # at every new call, we check for timeout, game over/clear, or depth limit reached
            check_timeout()
            if curr_env.done() or depth == 0:
                return  self.utility_heuristic(curr_env, agent_id)
            
            value = float('inf') # v = +infinity
            operators, children = self.successors(curr_env, other_agent_id)
            
            # If no legal moves (terminal state), evaluate state
            if not children:
                return  self.utility_heuristic(curr_env, agent_id) # return utility value of the current state

            for child in children:
                value = min(value, max_value(child, depth - 1))
            return value

        # Minimax: Max Value (Agent's Turn)
        def max_value(curr_env, depth):
            # at every new call, we check for timeout, game over/clear, or depth limit reached
            check_timeout()
            if curr_env.done() or depth == 0: # if game over/clear or depth limit reached
                return  self.utility_heuristic(curr_env, agent_id) # return utility value of the current state
            
            value = float('-inf') # v = -infinity
            operators, children = self.successors(curr_env, agent_id)
            
            # If no legal moves (terminal state), evaluate state
            if not children:
                return  self.utility_heuristic(curr_env, agent_id) # return utility value of the current state

            for child in children:
                value = max(value, min_value(child, depth - 1))
            return value

        # we define a best_op variable to store the best move found so far, 
        # this is the operator the agent will take decided by RB-minimax
        ####### best_op = None
        ####### 
        ####### # we store the first legal move in case we timeout before finding any better move
        ####### initial_ops = env.get_legal_operators(agent_id)
        ####### if not initial_ops:
        #######     return None
        ####### # NOTE: im not sure if this is the behavior the segel wants, but we have to have a fallback move
        best_op = "park" 

        # we start searching with depth = 1, 
        # and increase depth until timeout or game over/clear or max depth reached  
        # this is used with min_value and max_value functions for them to calculate values at certain depth
        current_depth = 1

        # try catch to handle timeout exception
        try:
            while True: # we loop until timeout, game over/clear, or max depth reached, each iteration calculates one depth level
                # we get all the chlildren and operators for the current env state
                operators, children = self.successors(env, agent_id)
                
                # if we are in a state where there are no legal moves, we break, as we are stuck
                # we could have checked `children` instead of `operators`, but both are equivalent here
                if not operators:
                    break
                
                # from here its basically the `value` function as we learned in the tutorials

                # these are variables to track the best move for THIS depth level
                best_op_in_depth = "park" # we default to first legal move (so we have fallback)
                max_val = float('-inf') # v = -infinity, we are a Max node
                
                for op, child in zip(operators, children):
                    # After we move, it's the opponent's turn (Min node)
                    val = min_value(child, current_depth - 1) # we calculate the value of each child, children are min nodes
                    
                    #
                    if val > max_val:
                        max_val = val
                        best_op_in_depth = op
                
                # If we completed the depth without timeout, update the global best_op
                best_op = best_op_in_depth
                current_depth += 1 # increase depth for next iteration
                
        except TimeoutException:
            # caught a timeout exception, we return the 
            # best_op from the last fully completed depth.
            pass
            
        return best_op


class AgentAlphaBeta(Agent):
    def utility_heuristic(self, env, robot_id):
        # Using the same logic as Minimax for consistency
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        time_buffer = 0.05 
        other_agent_id = (agent_id + 1) % 2
        
        class TimeoutException(Exception):
            pass

        def check_timeout():
            if time.time() - start_time >= time_limit - time_buffer:
                raise TimeoutException()

        def min_value(curr_env, depth, alpha, beta):
            check_timeout()
            if curr_env.done() or depth == 0:
                return self.utility_heuristic(curr_env, agent_id)
            
            v = float('inf')
            _, children = self.successors(curr_env, other_agent_id)
            
            if not children:
                return self.utility_heuristic(curr_env, agent_id)

            for child in children:
                v = min(v, max_value(child, depth - 1, alpha, beta))
                # Pruning logic: If this value is already worse than what Max can get elsewhere
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(curr_env, depth, alpha, beta):
            check_timeout()
            if curr_env.done() or depth == 0:
                return self.utility_heuristic(curr_env, agent_id)
            
            v = float('-inf')
            _, children = self.successors(curr_env, agent_id)
            
            if not children:
                return self.utility_heuristic(curr_env, agent_id)

            for child in children:
                v = max(v, min_value(child, depth - 1, alpha, beta))
                # Pruning logic: If this value is already better than what Min will allow
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        best_op = "park" 
        current_depth = 1

        try:
            while True:
                operators, children = self.successors(env, agent_id)
                if not operators:
                    break
                
                best_op_in_depth = operators[0]
                max_val = float('-inf')
                # Initialize alpha and beta for each iterative deepening step
                alpha = float('-inf')
                beta = float('inf')
                
                for op, child in zip(operators, children):
                    val = min_value(child, current_depth - 1, alpha, beta)
                    if val > max_val:
                        max_val = val
                        best_op_in_depth = op
                    
                    # Update alpha at the root level as well
                    alpha = max(alpha, max_val)
                
                best_op = best_op_in_depth
                current_depth += 1
                
        except TimeoutException:
            pass
            
        return best_op


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)