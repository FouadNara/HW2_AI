from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    steps = env.num_steps
    robot = env.get_robot(robot_id)
    pass

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return self.rb_alpha_beta(env, env.start, agent_id, time_limit, 0, -int("inf"), int("inf"))

    def rb_alpha_beta(self, env: WarehouseEnv, state, agent_id, time_limit, turn, a, b):
        if env.done() or time_limit == 0:
            return h(state, agent_id)

        agent = env.get_robot(agent_id)
        children = agent.successors(env, agent_id)
        if turn == agent_id:
            curMax = -int("inf")
            for c in children:
                v = self.rb_alpha_beta(env, c, agent_id, time_limit - 1, 1 - agent_id, a, b)
                curMax = max(v, curMax)
                a = max(curMax, a)
                if curMax >= b:
                    return int("inf")
            return curMax
        else:
            curMin = int("inf")
            for c in children:
                v = self.rb_alpha_beta(env, c, agent_id, time_limit - 1, 1 - agent_id, a, b)
                curMin = min(v, curMin)
                b = min(curMin, b)
                if curMin <= a:
                    return -int("inf")
            return curMin
        


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