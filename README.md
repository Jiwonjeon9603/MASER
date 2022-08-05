# Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer (MASER)
This repository is an implementation of "MASER: Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer" accepted to ICML 2022.

## Requirements
You need to install StarCraft2 and SMAC
I provide all libraries and packages for this codes. Try the follow
```
pip install -r requirements.txt
```


## Run Example 

python3 src/main.py --config=qmix --env-config=sc2 with learner=maser_q_learner t_max=3005000 use_cuda=True save_model=True lam=0.03 map_print=3m_maser_sparse env_args.map_name=3m alpha=0.5 ind=1 mix=1 expl=1 dis=1 goal=maser device_num=0
