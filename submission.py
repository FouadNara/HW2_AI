from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time # for time management in AgentMinimax

# time buffer so to avoid timeout problems and agents running out of time even tho they finished at the exact time limit
EPSILON_TIME = 0.05 

# SECTION: section A heuristic
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)

    # terminal state
    if env.done():
        if robot.credit > other_robot.credit:
            return float('inf')  # win
        elif robot.credit < other_robot.credit:
            return -float('inf') # loss
        else:
            return 0 # draw

    # f1: credit (our main objective, we want this to be maximized so ww give it a big positive weight)
    f1 = robot.credit
    w_credit  = 15

    # f2: distance to next goal (we want this to be minimized so we give it a negative weight)
    f2 = 0
    w_dist    = -5
    if robot.package is not None:
        f2 = manhattan_distance(robot.position, robot.package.destination)
    else:
        avail = [p for p in env.packages if p.on_board]
        if avail:
            # distance to pick up + distance to deliver (closer package)
            f2 = min(manhattan_distance(robot.position, p.position) + manhattan_distance(p.position, p.destination) for p in avail)

    # f3: battery
    f3 = robot.battery
    w_battery = 2

    return w_credit * f1 + w_dist * f2 + w_battery * f3
# !SECTION

# SECTION: section B minimax heuristic
def minimax_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    
    # our main objective is to maximize the credit difference
    # we also give it a huge weight  
    score = 1000 * (robot.credit - other_robot.credit) 

    # we will add a small contribution from battery
    score += robot.battery

    # we are looking to minimize the distance to our current goal
    if robot.package: # robot has a package
        dist = manhattan_distance(robot.position, robot.package.destination)
        score += (500 - dist) # we add a big BONUS for having a package, this encourages having packages 
    
    else: # robot has no package

        active_packages = [p for p in env.packages if p.on_board]
        
        # from all the "available" packages, find the closest one
        if active_packages:
            min_dist = float('inf')
            for p in active_packages:
                d = manhattan_distance(robot.position, p.position)
                if d < min_dist:
                    min_dist = d
            
            # we add a BONUS for picking up a package, this encourages the robot to look for packages
            score += (200 - min_dist)

    return score
# !SECTION

# SECTION: section B & D eval function for minimax and expectimax
def eval_function(env: WarehouseEnv, agent_id: int):
        if env.done(): # terminal state
            my_robot = env.get_robot(agent_id)
            op_robot = env.get_robot((agent_id + 1) % 2)

            
            if my_robot.credit > op_robot.credit:
                return float('inf')  # Win by Score
            elif my_robot.credit < op_robot.credit:
                return -float('inf') # Loss by Score
            else:
                # we will use battery as a tiebreaker
                if my_robot.battery > op_robot.battery:
                    return 10000  # Win by Battery
                elif my_robot.battery < op_robot.battery:
                    return -10000 # Loss by Battery
                else:
                    return 0 # Absolute Draw
                    
        return minimax_heuristic(env, agent_id)
# !SECTION

# SECTION: section A improved greedy agent
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
# !SECTION

# SECTION: section B minimax agent
class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        end_time = start_time + time_limit - EPSILON_TIME

        # fallback solution in case we timeout before finishing depth=1
        legal = env.get_legal_operators(agent_id)
        if legal:
            ops, children = self.successors(env, agent_id)
            best_op = random.choice(legal) if legal else None
            best_val = -float('inf')
            for op, child in zip(ops, children):
                v = eval_function(child, agent_id)
                if v > best_val:
                    best_val = v
                    best_op = op

        depth = 1

        while True:
            try:
                # we dont actually care abou the value here, just the operator of teh best move
                _, op = self.minimax(env, agent_id, depth, end_time, maximizing_player=True)
                if op is not None:
                    best_op = op
                depth += 1
            except TimeoutError:
                break

        return best_op


    def minimax(self, env: WarehouseEnv, agent_id: int, depth: int, end_time: float, maximizing_player: bool=True):
        if maximizing_player:
            return self.max_value(env, agent_id, depth, end_time)
        else:
            return self.min_value(env, agent_id, depth, end_time)


    def max_value(self, env: WarehouseEnv, agent_id: int, depth: int, end_time: float):
        if time.time() > end_time:
            raise TimeoutError()

        if depth == 0 or env.done():
            return eval_function(env, agent_id), None

        # MAX plays as agent_id (the root)
        operators, children = self.successors(env, agent_id)
        if not operators:
            return eval_function(env, agent_id), None

        # reorder the moves so we can look at more promising moves first
        pairs = list(zip(operators, children))
        pairs.sort(key=lambda oc: eval_function(oc[1], agent_id), reverse=True)

        best_val = -float('inf')
        best_op = pairs[0][0]

        for op, child in pairs:
            if time.time() > end_time:
                raise TimeoutError()

            val, _ = self.minimax(child, agent_id, depth - 1, end_time, maximizing_player=False)
            if val > best_val:
                best_val = val
                best_op = op

        return best_val, best_op


    def min_value(self, env: WarehouseEnv, agent_id: int, depth: int, end_time: float):
        if time.time() > end_time:
            raise TimeoutError()

        if depth == 0 or env.done():
            return eval_function(env, agent_id), None

        opponent_id = (agent_id + 1) % 2

        # MIN plays as opponent_id, but........... we still evaluate from agent_id's perspective
        operators, children = self.successors(env, opponent_id)
        if not operators:
            return eval_function(env, agent_id), None

        # reorder the moves so we can look at more promising moves first
        pairs = list(zip(operators, children))
        pairs.sort(key=lambda oc: eval_function(oc[1], agent_id))

        best_val = float('inf')
        best_op = pairs[0][0]

        for op, child in pairs:
            if time.time() > end_time:
                raise TimeoutError()

            val, _ = self.minimax(child, agent_id, depth - 1, end_time, maximizing_player=True)
            if val < best_val:
                best_val = val
                best_op = op

        return best_val, best_op
# !SECTION

# SECTION: section C alpha-beta agent
class AgentAlphaBeta(Agent):
    def utility_heuristic(self, env, robot_id):
        # Using the same logic as Minimax for consistency
        return minimax_heuristic(env, (robot_id + 1) % 2)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        time_buffer = EPSILON_TIME
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
# !SECTION

# SECTION: section D expectimax agent
class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.expectimax(env, agent_id, 1, agent_id, time.time() + time_limit - EPSILON_TIME)

    def expectimax(self, env: WarehouseEnv, agent_id: int, depth: int, turn: int, end_time: float):
        if time.time() > end_time:
            raise TimeoutError()

        if depth == 0 or env.done():
            return eval_function(env, agent_id)

        operators, children = self.successors(env, agent_id)

        if not operators:
            return eval_function(env, agent_id)
        
        if turn == agent_id: # its our agents turn, we find max over children
            CurMax = -float('inf')
            for child in children:
                val = self.expectimax(child, agent_id, depth - 1, (turn + 1) % 2, end_time)
                CurMax = max(CurMax, val)
            return CurMax
        
        else: # turn == (agent_id + 1) % 2
            # all children are equally likely, but the acions "move west" (move left) 
            # and "pick up" are 3 times more likely
            Probability = [1] * len(operators) #a list for probabilities

            sum = 0
            index = 0
            for child, op in zip(children, operators):
                if op == "move west" or op == "pick up":
                    sum += 3
                    Probability[index] = 3
                else:
                    sum += 1
                index += 1
                
            Probability = [p / sum for p in Probability] # Normalizing the  probabilities
            # now here we have the probabilities of each child

            # here we calculate the expected value [sum over children (x * prob(x))]
            CurExp = 0
            for child, prob in zip(children, Probability):
                CurExp += prob * self.expectimax(child, agent_id, depth - 1, (turn + 1) % 2, end_time)

            return CurExp
# !SECTION

# SECTION: hard coded agent
# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop off"] 
        # i fixed "pick_up" and "drop_off" to "pick up" and "drop off" (like in Warehouse.py, otherwise it would be illegal moves)

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
# !SECTION

