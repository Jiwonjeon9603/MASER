from .q_learner import QLearner
from .maser_q_learner import maserQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY['maser_q_learner'] = maserQLearner
