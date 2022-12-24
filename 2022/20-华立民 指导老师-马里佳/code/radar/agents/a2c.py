from radar.agents.ppo import PPOLearner
from torch.distributions import Categorical

class A2CLearner(PPOLearner):   #继承自PPOLearner方法

    def __init__(self, params):
        params["nr_epochs"] = 1
        super(A2CLearner, self).__init__(params)

    def policy_loss(self, advantage, probs, action, old_prob):  #覆写方法
        m = Categorical(probs)
        return -m.log_prob(action) * advantage