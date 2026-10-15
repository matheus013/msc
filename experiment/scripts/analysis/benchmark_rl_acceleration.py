"""Small diagnostic only. Never resumes the production census.

Compare scalar CPU/CUDA, a cached deterministic environment, and independent
Double-DQN network updates stacked with vmap (not shared policy training).
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import sys
import time
import cProfile
import pstats
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call,stack_module_state

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.production import production_config
from simulation.core.rl_torch import DoubleDQNAgent,PPOAgent,_DuelingQNet
import simulation.core.inventory_env as env_module
from simulation.core.device import available_devices

OriginalEnv=env_module.InventoryEnv


class CachedEnv(OriginalEnv):
    """Action-independent state features computed once, with causal prefix rules."""
    def __init__(self,demand,cfg,**kwargs):
        super().__init__(demand,cfg,**kwargs)
        self._fixed_states=np.stack([self._at_time(t) for t in range(self.T+1)])
        self.t=0

    def _at_time(self,t):
        self.t=t
        return OriginalEnv._state(self)

    def _state(self):
        if not hasattr(self,'_fixed_states'):return OriginalEnv._state(self)
        state=self._fixed_states[self.t].copy()
        state[0]=self.inventory/(self.init_inv+1e-6)
        state[2]=sum(self.pipeline)/(self.init_inv+1e-6)
        return state


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def check_environment(train,cfg):
    a=OriginalEnv(train,cfg);b=CachedEnv(train,cfg)
    np.testing.assert_array_equal(a.reset(),b.reset())
    orders=np.random.default_rng(912).uniform(0,15,len(train))
    for order in orders:
        x,rx,dx,ix=a.step(order);y,ry,dy,iy=b.step(order)
        np.testing.assert_array_equal(x,y)
        assert rx==ry and dx==dy and ix==iy
    assert a.kpis()==b.kpis()


def one_fit(name,train,device,cached,transitions,seed):
    os.environ['AIPE_DEVICE']=device
    cfg=production_config(train,seed)
    for key in ['DQN','PPO']:
        cfg[key]['episodes']=math.ceil(transitions/len(train));cfg[key]['device']=device
    check_environment(train,cfg)
    prepared=CachedEnv(train,cfg) if cached else None
    def factory(demand,config,**kwargs):
        assert config is cfg and np.array_equal(demand,train)
        prepared.reset()
        return prepared
    env_module.InventoryEnv=factory if cached else OriginalEnv
    try:
        sync(device);start=time.perf_counter()
        agent=(DoubleDQNAgent if name=='DQN' else PPOAgent)(OriginalEnv.STATE_DIM,cfg)
        sync(device);initialized=time.perf_counter()
        agent.train(train,cfg,verbose=False)
        sync(device);finished=time.perf_counter()
        return {'method':name,'origin':len(train),'device':device,'cached_env':cached,'seed':seed,
                'episodes':cfg[name]['episodes'],'transitions':cfg[name]['episodes']*len(train),
                'initialization_seconds':initialized-start,'training_seconds':finished-initialized,
                'final_training_reward':agent.reward_hist[-1]}
    finally:
        env_module.InventoryEnv=OriginalEnv


def independent_updates(n,steps,output):
    """Compare separate Adam updates with a stacked, independent-parameter Adam.

    This benchmark excludes rollout, replay sampling, exploration and target
    refresh. Thus its speedup is NOT an end-to-end training speedup.
    """
    device='cuda';torch.manual_seed(715)
    models=[_DuelingQNet(6,20,[64,64]).to(device) for _ in range(n)]
    targets=deepcopy(models)
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
    params,buffers=stack_module_state(models)
    target_params,target_buffers=stack_module_state(targets)
    base=deepcopy(models[0]).to('meta')
    opt=torch.optim.Adam(params.values(),lr=.001)
    states=torch.randn(n,32,6,device=device);next_states=torch.randn_like(states)
    actions=torch.randint(20,(n,32,1),device=device)
    rewards=torch.randn(n,32,device=device)*10
    dones=torch.randint(2,(n,32),device=device).float()

    def scalar_step():
        for i,(model,target,optimizer) in enumerate(zip(models,targets,optimizers)):
            q=model(states[i]).gather(1,actions[i]).squeeze(-1)
            with torch.no_grad():
                best=model(next_states[i]).argmax(-1,keepdim=True)
                y=rewards[i]+(1-dones[i])*.95*target(next_states[i]).gather(1,best).squeeze(-1)
            loss=torch.nn.functional.smooth_l1_loss(q,y)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()

    def loss_one(p,b,tp,tb,s,ns,a,r,d):
        q=functional_call(base,(p,b),(s,)).gather(1,a).squeeze(-1)
        with torch.no_grad():
            best=functional_call(base,(p,b),(ns,)).argmax(-1,keepdim=True)
            y=r+(1-d)*.95*functional_call(base,(tp,tb),(ns,)).gather(1,best).squeeze(-1)
        return torch.nn.functional.smooth_l1_loss(q,y)

    def vector_step():
        losses=torch.vmap(loss_one)(params,buffers,target_params,target_buffers,
                                   states,next_states,actions,rewards,dones)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()  # mean across agents would change each gradient
        norm=torch.stack([p.grad.reshape(n,-1).square().sum(1) for p in params.values()]).sum(0).sqrt()
        scale=(10./(norm+1e-6)).clamp(max=1.)
        for p in params.values():p.grad.mul_(scale.reshape(n,*([1]*(p.ndim-1))))
        opt.step()

    scalar_step();vector_step();sync(device)
    differences=[]
    for i,model in enumerate(models):
        for name,p in model.named_parameters():
            torch.testing.assert_close(params[name][i],p,rtol=1e-4,atol=2e-6)
            differences.append(float((params[name][i]-p).abs().max().item()))
    # Warm up kernels; then measure steady state, synchronize only at boundaries.
    for _ in range(3):scalar_step();vector_step()
    sync(device);start=time.perf_counter()
    for _ in range(steps):scalar_step()
    sync(device);scalar=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(steps):vector_step()
    sync(device);vector=time.perf_counter()-start
    return {'independent_agents':n,'minibatch_per_agent':32,'steps':steps,
            'scalar_seconds':scalar,'vector_seconds':vector,'update_speedup':scalar/vector,
            'one_step_max_weight_difference':max(differences),
            'scope':'Double DQN optimizer updates only; no end-to-end training claim'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--transitions',type=int,default=2048)
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    output=ROOT/'experiment/data/08_reporting/performance'/args.run
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    panels=ROOT/'experiment/data/08_reporting/research_v2/census_2026-10-14/panels'
    series={17:pd.read_parquet(panels/'BA_BA_17.parquet').iloc[0,:17].to_numpy(float),
            50:pd.read_parquet(panels/'M5_M5_50.parquet').iloc[0,:50].to_numpy(float)}
    # Initialize libraries outside reported fits.
    for dev in ['cpu','cuda']:
        if dev=='cuda' and not torch.cuda.is_available():continue
        for method in ['DQN','PPO']:one_fit(method,series[17],dev,False,68,7)
    rows=[]
    for repeat in range(2):
        for origin,train in series.items():
            for method in ['DQN','PPO']:
                modes=[('cpu',False),('cuda',False),('cpu',True),('cuda',True)]
                if repeat:modes.reverse()
                for device,cached in modes:
                    if device=='cuda' and not torch.cuda.is_available():continue
                    row=one_fit(method,train,device,cached,args.transitions,11+repeat)
                    rows.append(row);pd.DataFrame(rows).to_csv(output/'timings.csv',index=False)
                    print(json.dumps(row),flush=True)
    vector=[]
    if torch.cuda.is_available():
        for n in [1,8,32,128]:
            row=independent_updates(n,20,output);vector.append(row)
            pd.DataFrame(vector).to_csv(output/'independent_updates.csv',index=False)
            print(json.dumps(row),flush=True)
    # Profile separately; instrumentation excluded from speed comparisons.
    for device in ['cpu','cuda']:
        if device=='cuda' and not torch.cuda.is_available():continue
        profiler=cProfile.Profile();profiler.enable()
        one_fit('DQN',series[17],device,False,1024,11)
        profiler.disable()
        with (output/f'profile_dqn_{device}.txt').open('w',encoding='utf-8') as stream:
            pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
    (output/'manifest.json').write_text(json.dumps({'kind':'isolated performance diagnostics, not scientific evidence',
        'hardware':available_devices(),'transitions_requested':args.transitions,'repeats':2,
        'environment_cache_equivalence':'Exact state/reward/KPI equality checked for fixed random actions on both real training prefixes.',
        'prototype':'Independent model parameters, Adam moments and per-agent gradient clipping; equality tested for one update.',
        'limitations':['Two example series and two seeds only.','Training rewards on CPU/CUDA need not match because RNG streams differ.',
            'No complete RL rollout/replay implementation is vectorized by the update microbenchmark.'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')


if __name__=='__main__':main()
