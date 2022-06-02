from .q_learner import QLearner
from .coma_learner import COMALearner
from .maser_q_learner import MASERQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY['maser_q_learner'] = MASERQLearner
