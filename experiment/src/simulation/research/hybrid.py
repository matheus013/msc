"""GA-gated discrete DQN: identical execution rules in demonstrations/train/test.

An action index denotes a nominal grid quantity. The fixed GA gate and floor
transform that nominal action in all three phases. This is our specified
variant, not a claim of exact reproduction of Zabraoui's implementation.
"""
import numpy as np
from simulation.core.inventory_env import InventoryEnv
from simulation.core.rl_torch import DoubleDQNAgent, _terminal_deficit_penalty

HYBRID_VERSION='ga_gated_dqn_consistent_v2'


class GAGuidedDQN(DoubleDQNAgent):
    def __init__(self,state_dim,cfg,ga_params):
        super().__init__(state_dim,cfg)
        self.ga={key:float(ga_params[key]) for key in ['ROP','Q','SS']}
        self.floor_ratio=float(cfg['HYBRID']['floor_ratio'])

    def executed_quantity(self,index,env):
        if env.inventory+sum(env.pipeline)>self.ga['ROP']+self.ga['SS']:return 0.
        return max(float(self.actions[index]),self.ga['Q']*self.floor_ratio)

    def episode(self,demand,cfg,seed,demonstration=False):
        env=InventoryEnv(demand,cfg,seed=seed);state=env.reset();done=False;transitions=[]
        while not done:
            if demonstration:
                # Label and execution use exactly the same transformed action.
                quantities=[self.executed_quantity(i,env) for i in range(self.n_act)]
                target=self.ga['Q'] if env.inventory+sum(env.pipeline)<=self.ga['ROP']+self.ga['SS'] else 0.
                index=int(np.argmin(np.abs(np.asarray(quantities)-target)))
            else:index,_=self.act(state,explore=True)
            quantity=self.executed_quantity(index,env)
            next_state,reward,done,_=env.step(quantity)
            transitions.append([state,index,reward,next_state,done]);state=next_state
        if transitions:transitions[-1][2]-=_terminal_deficit_penalty(env.kpis(),demand,cfg)
        return transitions

    def prepopulate(self,demand,cfg,n_transitions=1000):
        added=0;episode=0
        while added<n_transitions:
            transitions=self.episode(demand,cfg,2000+episode,demonstration=True)
            if not transitions:raise ValueError('Cannot prepopulate from an empty history')
            # Finish the physical episode to compute terminal reward consistently;
            # a truncated stored prefix has no fabricated terminal transition.
            for transition in transitions[:n_transitions-added]:
                self.remember(*transition);added+=1
            episode+=1
        return added

    def train(self,demand,cfg,verbose=False,demand_pool=None):
        for episode in range(int(cfg['DQN']['episodes'])):
            values=demand_pool[episode%len(demand_pool)] if demand_pool else demand
            transitions=self.episode(values,cfg,episode)
            for transition in transitions:self.remember(*transition);self.replay()
            self.reward_hist.append(sum(t[2] for t in transitions))

    def make_policy(self):
        self.q_net.eval()
        def policy(state,env):
            index,_=self.act(state,explore=False)
            return self.executed_quantity(index,env)
        return policy
