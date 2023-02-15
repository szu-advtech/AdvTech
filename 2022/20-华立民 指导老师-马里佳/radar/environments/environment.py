import random
import numpy
import math
from gym import spaces
from radar.utils import get_param_or_default, check_value_not_none, get_value_if

class Environment:
    
    def __init__(self, params):
        self.nr_agents = get_param_or_default(params, "nr_agents", 1)
        self.nr_actions = params["nr_actions"]
        self.time_step = 0
        self.discounted_return = 0
        self.undiscounted_return = 0
        self.time_limit = params["time_limit"]
        self.gamma = params["gamma"]
        self.actions = list(range(self.nr_actions))

    def relative_position(self, agent, other_position):
        x_0, y_0 = agent.position
        x, y = other_position
        dx = x - x_0
        dy = y - y_0
        return (dx, dy)

    def global_state(self):
        pass

    def local_observation(self, agent=None, adversary_ids=[]):
        pass

    def get_agent(self, id):
        pass

    def domain_statistic(self):
        return 0

    def joint_observation(self, adversary_ids=[]):
        return [self.local_observation(self.get_agent(i), adversary_ids) for i in range(self.nr_agents)]

    def reset(self, adversary_ids=[]):
        self.time_step = 0
        self.discounted_return = 0
        self.undiscounted_return = 0
        
    def step(self, joint_action, adversary_ids=[]):
        pass

    def is_legal_action(self, action, agent=None):
        return True

    def state_summary(self):
        return {
            "nr_agents": self.nr_agents,
            "nr_actions": self.nr_actions,
            "time_step": self.time_step,
            "discounted_return": self.discounted_return,
            "undiscounted_return": self.undiscounted_return,
            "time_limit": self.time_limit,
            "gamma": self.gamma
        }

NOOP = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_WEST = 3
MOVE_EAST = 4

# Used to encode explicit graph edges, restricting possible moves
GRAPH_MAP = {
    "^": MOVE_NORTH,
    "v": MOVE_SOUTH,
    "<": MOVE_WEST,
    ">": MOVE_EAST
}

GRIDWORLD_ACTIONS = [NOOP, MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]

class GridWorldObject:

    def __init__(self, id, initial_position, env, fixed_initial_position=False):
        self.id = check_value_not_none(id, "id")
        self.env = check_value_not_none(env, "env")
        self.position = get_value_if(initial_position,
            initial_position is not None, self.env.free_random_position())
        self.fixed_initial_position = fixed_initial_position
        self.initial_position = get_value_if(initial_position, fixed_initial_position, None)
        self.done = False

    def reset(self):
        self.position = get_value_if(self.initial_position,
            self.fixed_initial_position, self.env.free_random_position())
        self.done = False

    def move(self, action):
        if action != NOOP and self.env.is_legal_action(action, self):
            x, y = self.position
            if not self.env.graph_yx or self.env.has_edge(action, x, y):
                if action == MOVE_NORTH and y + 1 < self.env.height:
                    return self.set_position((x, y + 1))
                if action == MOVE_SOUTH and y - 1 >= 0:
                    return self.set_position((x, y - 1))
                if action == MOVE_WEST and x - 1 >= 0:
                    return self.set_position((x - 1, y))
                if action == MOVE_EAST and x + 1 < self.env.width:
                    return self.set_position((x + 1, y))

    def set_position(self, new_position):
        # obstacle collision check
        if new_position in self.env.obstacles:
            return
        self.position = new_position

    def state_summary(self):
        return {
            "id": self.id,
            "position": self.position,
            "fixed_initial_position": self.fixed_initial_position,
            "initial_position": self.initial_position,
            "done": self.done,
        }

class GridWorldEnvironment(Environment):

    def __init__(self, params):
        super(GridWorldEnvironment, self).__init__(params)
        self.width = params["width"]
        self.height = params["height"]
        self.view_range = get_param_or_default(params, "view_range", 5)
        self.collisions_allowed = get_param_or_default(params, "collisions_allowed", False)
        self.obstacles = get_param_or_default(params, "obstacles", [])
        self.fixed_initial_position = get_param_or_default(params, "fixed_initial_position", False)
        self.initial_positions = get_param_or_default(params, "initial_positions", \
            [None for _ in range(self.nr_agents)])
        self.obstacle_free_positions = [(x, y) for x in range(self.width) \
            for y in range(self.height) if (x, y) not in self.obstacles]
        if "agents" not in params:
            self.agents = [GridWorldObject(i, pos, self, self.fixed_initial_position) \
                for i, pos in enumerate(self.initial_positions)]
        else:
            self.agents = params["agents"]
        self.action_space = spaces.Discrete(get_param_or_default(params, "nr_actions", len(GRIDWORLD_ACTIONS)))
        if "graph_grid_yx" in params:
            self.graph_yx = params["graph_grid_yx"]
        else:
            self.graph_yx = self.gen_full_mesh()
    
    def visible_positions_of(self, agent_id):
        x_0, y_0 = self.agents[agent_id].position
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        return [(x,y) for x in range(-x_center+x_0, x_center+1+x_0) for y in range(-y_center+y_0, y_center+1+y_0)]

    def free_random_position(self):
        return random.choice(self.obstacle_free_positions)

    def is_legal_action(self, action, agent=None):
        collisions_allowed = self.collisions_allowed
        if self.nr_agents == 1:
            agent = self.agents[0]
            collisions_allowed = True
        if agent.done:
            return False 
        if action != NOOP:
            x, y = agent.position
            new_position = agent.position
            if action == MOVE_NORTH and y + 1 < self.height:
                new_position = (x, y + 1)
            if action == MOVE_SOUTH and y - 1 >= 0:
                new_position = (x, y - 1)
            if action == MOVE_WEST and x - 1 >= 0:
                new_position = (x - 1, y)
            if action == MOVE_EAST and x + 1 < self.width:
                new_position = (x + 1, y)
            not_in_obstacles = new_position not in self.obstacles
            if collisions_allowed:
                return not_in_obstacles
            no_collision = new_position not in [other_agent.position \
                for other_agent in self.agents if other_agent.id != agent.id and not other_agent.done]
            return not_in_obstacles and no_collision
        return True
    
    def reset(self, adversary_ids=[]):
        super().reset(adversary_ids)
        for agent in self.agents:
            agent.reset()
        return [self.local_observation(agent) for agent in self.agents]

    def get_agent(self, agent_id):
        return self.agents[agent_id]

    def state_summary(self):
        summary = super(GridWorldEnvironment, self).state_summary()
        summary["agents"] = [agent.state_summary() for agent in self.agents]
        summary["obstacles"] = self.obstacles
        summary["width"] = self.width
        summary["height"] = self.height
        return summary

    def has_edge(self, movement, x, y):
        edges_str = self.graph_yx[y][x]  # note: graph is shape (y,x) for easy reading/encoding
        edges = [GRAPH_MAP[char] for char in edges_str]  # convert from string to movement int
        return movement in edges

    def gen_full_mesh(self):
        graph = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                edge = ""
                if h > 0:
                    edge = edge + "^"
                if h < self.height-1:
                    edge = edge + "v"
                if w > 0:
                    edge = edge + "<"
                if w < self.width-1:
                    edge = edge + ">"
                row.append(edge)
            graph.append(row)
        return graph[::-1]  # inverse y-axis