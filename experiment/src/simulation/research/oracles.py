"""Hindsight inventory bounds. Never use these outputs to train/select a policy.

The MILP has perfect information about a fixed test trajectory, continuous
order quantities and binary order/stockout indicators. It is a conditional
planning optimum, not an achievable stochastic or universal inventory bound.
"""
import warnings
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint


def perfect_information_plan(demand, cfg, alpha=.7, time_limit=30):
    from simulation.research.protocol import simulate
    d=np.asarray(demand,float)
    if d.ndim!=1 or not len(d) or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Expected nonempty finite nonnegative demand')
    if not 0<=alpha<=1:raise ValueError('alpha outside [0,1]')
    sc=cfg['SIMULATION']; cc=cfg['COST']; L=int(sc['lead_time']); T=len(d)
    if sc.get('lead_time_convention')!='exact':raise ValueError('Exact lead time required')
    pipe=np.asarray(sc.get('initial_pipeline',np.zeros(L)),float)
    if len(pipe)!=L or (pipe<0).any():raise ValueError('Invalid initial pipeline')
    initial=float(sc['initial_inventory']); total=float(d.sum())
    if initial<0:raise ValueError('Negative initial inventory')
    available=initial; unavoidable=0.
    for t in range(min(L,T)):
        available+=pipe[t]
        unavoidable+=max(0.,d[t]-available)
        available=max(0.,available-d[t])
    max_service=(total-unavoidable)/total if total else None
    target_feasible=not total or max_service+1e-10>=alpha
    # If target is physically impossible, maximize service then minimize cost.
    shortage_limit=(1-alpha)*total if target_feasible else unavoidable
    # Variables: q (orders), I (ending inventory), u (shortage), y (order), z (inventory positive).
    q=np.arange(T); I=q+T; u=q+2*T; y=q+3*T; z=q+4*T
    M=max(1.,initial+float(pipe.sum())+total)
    objective=np.zeros(5*T)
    objective[q]=cc['ordering_cost_per_unit']; objective[I]=cc['holding_cost_per_unit']
    objective[u]=cc['stockout_cost_per_unit']; objective[y]=cc['ordering_cost_per_order']
    lower=np.zeros(5*T);upper=np.full(5*T,np.inf)
    upper[q]=total;upper[I]=M;upper[u]=d;upper[y]=1;upper[z]=1
    for t in range(T):
        if t+L>=T:upper[q[t]]=0 # no credit for orders arriving after test
    integers=np.zeros(5*T,int);integers[y]=1;integers[z]=1
    matrix=[];lo=[];hi=[]
    def add(values,low=-np.inf,high=np.inf):
        row=np.zeros(5*T)
        for col,value in values.items():row[col]=value
        matrix.append(row);lo.append(low);hi.append(high)
    for t in range(T):
        balance={I[t]:1.,u[t]:-1.}
        if t:balance[I[t-1]]=-1.
        if t>=L:balance[q[t-L]]=-1.
        rhs=(initial if t==0 else 0.)+(pipe[t] if t<L else 0.)-d[t]
        add(balance,rhs,rhs)
        add({q[t]:1.,y[t]:-total},high=0)
        # Lost-sales complementarity: inventory and shortage cannot coexist.
        add({I[t]:1.,z[t]:-M},high=0)
        add({u[t]:1.,z[t]:d[t]},high=d[t])
    add({int(j):1. for j in q},high=total)
    add({int(j):1. for j in u},high=shortage_limit+1e-9*max(1,total))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
        result=milp(objective,integrality=integers,bounds=Bounds(lower,upper),
            constraints=LinearConstraint(np.asarray(matrix),lo,hi),
            options={'time_limit':float(time_limit),'mip_rel_gap':1e-9,'threads':1})
    out={'status':int(result.status),'proven_optimal':bool(result.success),
         'message':result.message,'target_feasible':bool(target_feasible),
         'max_service':max_service,'alpha':float(alpha),'cost':None,'service':None,
         'dual_bound':None if getattr(result,'mip_dual_bound',None) is None else float(result.mip_dual_bound),
         'mip_gap':None if getattr(result,'mip_gap',None) is None else float(result.mip_gap)}
    if result.x is None:return out
    orders=np.maximum(result.x[q],0.)
    orders[np.abs(orders)<1e-8]=0.
    metrics,_=simulate(d,cfg,lambda state,env:float(orders[env.t]))
    if not np.isclose(metrics['TIC'],result.fun,rtol=1e-6,atol=1e-5):
        raise ValueError('MILP objective does not match inventory simulator')
    if total and metrics['ServiceLevel']+1e-7 < (alpha if target_feasible else max_service):
        raise ValueError('MILP schedule does not attain required feasible service')
    out.update(cost=float(metrics['TIC']),service=float(metrics['ServiceLevel']) if total else None,
               orders=orders.tolist(),served=float(metrics['DemandServed']))
    return out


def efficiency_percent(reference_cost,actual_cost):
    """100 * minimum cost / actual cost; caller must first check service feasibility."""
    if reference_cost<0 or actual_cost<0:raise ValueError('Negative cost')
    if actual_cost==0:return 100. if reference_cost==0 else None
    return 100.*reference_cost/actual_cost
