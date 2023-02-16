import random
import numpy
from gym import spaces
from radar.environments.environment import GridWorldEnvironment, GridWorldObject, GRIDWORLD_ACTIONS
from radar.utils import get_param_or_default, check_value_not_none, get_value_if

AGENT_CHANNEL = 0
PREY_CHANNEL = 1
OBSTACLE_CHANNEL = 2
ADVERSARY_CHANNEL = 3

PREDATOR_PREY_CHANNELS = [AGENT_CHANNEL, PREY_CHANNEL, OBSTACLE_CHANNEL]

class Predator(GridWorldObject):

    def __init__(self, id, initial_position, env, fixed_initial_position=False):
        super(Predator, self).__init__(id, initial_position, env, fixed_initial_position)
        self.capture_participation = 0.0

    # noinspection PyAttributeOutsideInit
    def reset(self):
        super().reset()
        self.capture_participation = 0.0

    def add_capture_participation(self, participation):
        self.capture_participation += participation

class PredatorPreyEnvironment(GridWorldEnvironment):

    def __init__(self, params):
        super(PredatorPreyEnvironment, self).__init__(params)
        self.nr_channels_global = len(PREDATOR_PREY_CHANNELS)
        self.nr_channels_local = self.nr_channels_global + 2
        self.nr_preys = int(params["nr_agents"]/2)
        self.nr_capturers_required = 2
        self.failed_capture_penalty = get_param_or_default(params, "failed_capture_penalty", 0)
        self.agents = [Predator(i, None, self, False) for i in range(params["nr_agents"])] 
        self.preys = [GridWorldObject(i, None, self, False) for i in range(self.nr_preys)]
        self.global_observation_space = spaces.Box(-numpy.inf, numpy.inf,
                                                   shape=(self.nr_channels_global, self.width, self.height))
        self.view_range = get_param_or_default(params, "view_range", 5)
        default_capture_distance = 2
        self.capture_distance = get_param_or_default(params, "capture_distance", default_capture_distance)
        self.prey_capture_count = 0
        self.failed_capture_count = 0
        self.local_observation_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(self.nr_channels_local, self.view_range, self.view_range))

    def global_state(self):
        state = numpy.zeros(self.global_observation_space.shape)
        for agent in self.agents:
            x, y = agent.position
            state[AGENT_CHANNEL][x][y] += 1
        for prey in self.preys:
            x, y = prey.position
            state[PREY_CHANNEL][x][y] += 1
        for obstacle in self.obstacles:
            x, y = obstacle
            state[OBSTACLE_CHANNEL][x][y] += 1
        return state

    def local_observation(self, agent=None, adversary_ids=[]):
        if self.nr_agents == 1:
            agent = self.agents[0]
        observation = numpy.zeros(self.local_observation_space.shape)
        if agent.done:
            return observation
        x_0, y_0 = agent.position
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        visible_positions = [(x,y) for x in range(-x_center+x_0, x_center+1+x_0) for y in range(-y_center+y_0, y_center+1+y_0)]
        for visible_position in visible_positions:
            x, y = visible_position
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                dx, dy = self.relative_position(agent, visible_position)
                observation[OBSTACLE_CHANNEL+1][x_center+dx][y_center+dy] = 1
        observation[0][x_center][y_center] += 1
        for other_agent in self.agents:
            if other_agent.id != agent.id and other_agent.position in visible_positions:
                dx, dy = self.relative_position(agent, other_agent.position)
                if not (agent.id in adversary_ids and other_agent.id in adversary_ids):
                    observation[AGENT_CHANNEL+1][x_center+dx][y_center+dy] += 1
                else:
                    observation[ADVERSARY_CHANNEL+1][x_center+dx][y_center+dy] += 1
        for prey in self.preys:
            if prey.position in visible_positions:
                dx, dy = self.relative_position(agent, prey.position)
                observation[PREY_CHANNEL+1][x_center+dx][y_center+dy] += 1
        for obstacle in self.obstacles:
            if obstacle in visible_positions:
                dx, dy = self.relative_position(agent, obstacle)
                observation[OBSTACLE_CHANNEL+1][x_center+dx][y_center+dy] += 1
        return observation

    def domain_statistic(self, adversary_ids=[]):
        return sum([agent.capture_participation for i, agent in enumerate(self.agents) if i not in adversary_ids])

    def step(self, joint_action, adversary_ids=[]):
        self.time_step += 1
        rewards = numpy.zeros(self.nr_agents)
        agent_positions = []
        for i, agent, action in zip(range(self.nr_agents), self.agents, joint_action):
            if not agent.done:
                agent.move(action)
            agent_positions.append(agent.position)
        for prey in self.preys:
            x_1, y_1 = prey.position
            capturers = []
            main_capturers = []
            for i,agent_position in enumerate(agent_positions):
                x_0, y_0 = agent_position
                distance = max(abs(x_1-x_0), abs(y_1-y_0))
                if distance <= self.capture_distance:
                    capturers.append(i)
                if prey.position == agent_position:
                    main_capturers.append(i)
            nr_capturers = 1.0*len(capturers)
            if nr_capturers >= self.nr_capturers_required:
                for i in main_capturers:
                    participation = 1.0/len(main_capturers)
                    #assert participation < 1, "partiticipation was {}".format(participation)
                    rewards[i] += participation
                    self.agents[i].add_capture_participation(participation)
                    if i not in adversary_ids:
                        self.prey_capture_count += participation
                prey.reset()
        if self.time_step >= self.time_limit:
            for agent in self.agents:
                agent.done = True
        global_reward = sum(rewards)
        self.discounted_return += global_reward*(self.gamma**(self.time_step-1))
        self.undiscounted_return += global_reward
        assert len(self.preys) == self.nr_preys
        return self.joint_observation(adversary_ids),\
            rewards, [agent.done for agent in self.agents], {
                "preys": [prey.position for prey in self.preys]
            }

    def reset(self, adversary_ids=[]):
        super().reset(adversary_ids)
        self.obstacle_free_positions = [(x, y) for x in range(self.width) \
            for y in range(self.height) if (x, y) not in self.obstacles]
        for prey in self.preys:
            prey.reset()
        self.failed_capture_count = 0
        self.prey_capture_count = 0
        return self.joint_observation(adversary_ids)

    def state_summary(self):
        summary = super(PredatorPreyEnvironment, self).state_summary()
        summary["obstacles"] = self.obstacles
        summary["width"] = self.width
        summary["height"] = self.height
        summary["preys"] = [prey.state_summary() for prey in self.preys]
        summary["view_range"] = self.view_range
        summary["prey_capture_count"] = self.prey_capture_count
        summary["failed_capture_count"] = self.failed_capture_count
        return summary


PREDATOR_PREY_LAYOUTS = {
    # N = 2
    "PredatorPrey-2": ("""
           . . . . .
           . . . . .
           . . . . .
           . . . . .
           . . . . . 
        """, 2),
    # N = 4
    "PredatorPrey-4": ("""
           . . . . . . .
           . . . . . . .
           . . . . . . .
           . . . . . . .
           . . . . . . .
           . . . . . . .
           . . . . . . .
        """, 4),
    # N = 8
    "PredatorPrey-8": ("""
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
           . . . . . . . . . .
        """, 8)
}

def make(domain, params):
    params["nr_actions"] = len(GRIDWORLD_ACTIONS)
    params["gamma"] = 0.95
    params["obstacles"] = []
    params["time_limit"] = 50
    params["fixed_initial_position"] = False
    params["collisions_allowed"] = True
    layout, params["nr_agents"] = PREDATOR_PREY_LAYOUTS[domain]
    layout = layout.splitlines()
    params["width"] = 0
    params["height"] = 0
    for _,line in enumerate(layout):
        splitted_line = line.strip().split()
        if splitted_line:
            for x,cell in enumerate(splitted_line):
                if cell == '#':
                    params["obstacles"].append((x,params["height"]))
                params["width"] = x
            params["height"] += 1
    params["width"] += 1
    
    return PredatorPreyEnvironment(params)
