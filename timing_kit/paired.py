"""Cold-read CUDA-graph A/B/A harness. Host-only: never include in submissions."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import statistics
import math

@dataclass
class Arm:
    name: str
    prepare: object       # () -> private state
    run: object           # (state) -> output; graph safe
    check: object         # (state, output) -> None, raises on failure
    refresh: object       # (state, seed) -> None; same fresh data for every arm
    sources: tuple        # exact frozen source files
    choices: object = None

def source_hashes(paths):
    return {str(p):hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths}

def storage_ids(value):
    """Tensor storages in ordinary nested states, including views and objects."""
    import torch
    seen=set();storages=set()
    def visit(x):
        if id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,torch.Tensor):
            if x.numel():storages.add((str(x.device),x.untyped_storage().data_ptr()))
        elif isinstance(x,dict):
            for child in x.values():visit(child)
        elif isinstance(x,(list,tuple)):
            for child in x:visit(child)
        elif hasattr(x,'__dict__'):
            visit(vars(x))
    visit(value)
    return storages

def interval(values, seed=0):
    """Percentile bootstrap of independent ABBA round means (not replays)."""
    rng=random.Random(seed)
    means=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000))
    return {'mean':statistics.mean(values),'ci95':[means[49],means[1949]]}


_T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,
       14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,
       26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}

def t975(df):
    """Two-sided 95% Student t critical value."""
    return _T975[df] if df in _T975 else 1.96+2.4/df

def capture_interval(per_capture):
    """95% t-interval over per-capture means. Rounds that share a capture share its memory layout, so they are not
    independent: the capture is the unit of replication and df = n_captures - 1."""
    n=len(per_capture);m=statistics.mean(per_capture)
    if n<2:return {'mean':m,'ci95':[float('-inf'),float('inf')],'per_capture':list(per_capture),'n_captures':n,'ci_method':'capture-mean t'}
    half=t975(n-1)*statistics.stdev(per_capture)/math.sqrt(n)
    return {'mean':m,'ci95':[m-half,m+half],'per_capture':list(per_capture),'n_captures':n,'ci_method':f'capture-mean t(df={n-1})'}

def paired(arms, *, seed=1, rounds=None, captures=None, replay_checks=100, pad_seed=None, legacy=False):
    """Cold-read ABBA graph timing of arms ['base','candidate','aa'].

    Default (harness v2): 8 captures per arm, each state allocated after a seeded random 0-32 MiB throwaway
    allocation (kept alive), 48 rounds, and every CI taken across per-capture means (df = captures - 1). The A/A
    per-capture means are reported with their max |mean|, and `counted_gain` is true only if the gain CI low > 0 AND
    the gain mean exceeds that max |A/A mean|. Memory: states are 3 arms x captures; pass captures=4 for very large
    states. `legacy=True` restores the old behaviour (4 captures, 40 rounds, no random offsets, round-level bootstrap CI).
    """
    import torch
    assert [a.name for a in arms]==['base','candidate','aa']
    if rounds is None:rounds=40 if legacy else 48
    if captures is None:captures=4 if legacy else 8
    assert rounds>=40 and captures>=4 and replay_checks>=100
    warnings=[]
    if not legacy:
        assert rounds%captures==0,'v2 needs rounds to be a multiple of captures (equal rounds per capture)'
        if captures<8:warnings.append(f'captures={captures} < 8: CI has only df={captures-1}; A/A noise is under-covered')
    before={a.name:source_hashes(a.sources) for a in arms}
    assert sorted(before['base'].values())==sorted(before['aa'].values()), 'A/A sources differ'
    rng=random.Random(seed)
    pad_bytes=0 if pad_seed is None else random.Random(pad_seed).randrange(65537)*512
    pad=torch.empty(pad_bytes,dtype=torch.uint8,device='cuda')
    # Initialized once, then ONLY READ by sum before each replay. Sink is preallocated.
    cold_buffer=torch.randn(64*1024*1024,device='cuda',dtype=torch.float32)
    sink=torch.empty((),device='cuda')
    def cold_read(): torch.sum(cold_buffer,dim=(0,),out=sink)
    states={a.name:[] for a in arms}
    allocated=set()
    prepare_order=[]
    first=rng.randrange(3)
    offsets={a.name:[] for a in arms}
    offset_keep=[]  # throwaway allocations stay alive for the whole measurement
    for capture in range(captures):
        order=arms[(first+capture)%3:]+arms[:(first+capture)%3]
        for a in order:
            prepare_order.append([capture,a.name])
            if legacy:offset=0
            else:
                offset=random.Random(seed*1000003+capture*101+[x.name for x in arms].index(a.name)).randrange(65537)*512
                if offset:offset_keep.append(torch.empty(offset,dtype=torch.uint8,device='cuda'))
            offsets[a.name].append(offset)
            state=a.prepare()
            a.refresh(state,seed)
            storage=storage_ids(state)
            assert not storage.intersection(allocated), 'prepared arms/captures share tensor storage'
            allocated.update(storage)
            states[a.name].append({'state':state})
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture_order=[]
    with torch.cuda.stream(stream):
        for capture in range(captures):
            order=list(arms);rng.shuffle(order)
            for a in order:
                row=states[a.name][capture];s=row['state']
                a.run(s) # eager warm-up outside capture
                stream.synchronize()
                g=torch.cuda.CUDAGraph()
                # Event nodes must be part of replay. Host GC/descheduling
                # between separate record()/replay() calls otherwise gets timed.
                begin=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                with torch.cuda.graph(g,stream=stream):
                    begin.record()
                    output=a.run(s)
                    end.record()
                row.update(graph=g,output=output,begin=begin,end=end)
                # Poison after capture: correctness must observe an actual replay.
                if output.is_floating_point(): output.fill_(float('nan'))
                else: output.fill_(torch.iinfo(output.dtype).min)
                g.replay();stream.synchronize();a.check(s,output)
                capture_order.append([capture,a.name])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # Validate every graph again after 100 cold replays and fresh data.
    for a in arms:
        for row in states[a.name]:
            a.refresh(row['state'],seed+100003)
            for _ in range(replay_checks): cold_read();row['graph'].replay()
            torch.cuda.synchronize();a.check(row['state'],row['output'])
    raw=[];saving=[];aa_saving=[];ratios=[];aa_ratios=[];round_cap=[]
    for rnd in range(rounds):
        pair_results={}
        fresh_seed=seed+1000000+rnd
        for other in rng.sample(['candidate','aa'],2):
            order=['base',other,other,'base']
            if rng.randrange(2):order.reverse()
            cap=rnd%captures
            for name in ['base',other]:
                a=next(a for a in arms if a.name==name)
                a.refresh(states[name][cap]['state'],fresh_seed)
            times={'base':[],other:[]}
            for name in order:
                a=next(a for a in arms if a.name==name);row=states[name][cap]
                cold_read();row['graph'].replay()
                row['end'].synchronize()
                us=row['begin'].elapsed_time(row['end'])*1000
                a.check(row['state'],row['output'])
                raw.append({'round':rnd,'pair':other,'capture':cap,'arm':name,'us':us,'input_seed':fresh_seed})
                times[name].append(us)
            b,c=statistics.mean(times['base']),statistics.mean(times[other])
            pair_results[other]=(b-c,b/c)
        round_cap.append(rnd%captures)
        saving.append(pair_results['candidate'][0]);ratios.append(pair_results['candidate'][1])
        aa_saving.append(pair_results['aa'][0]);aa_ratios.append(pair_results['aa'][1])
    after={a.name:source_hashes(a.sources) for a in arms}
    assert before==after,'sources changed during measurement'
    def by_capture(values):
        return [statistics.mean(v for v,c in zip(values,round_cap) if c==k) for k in range(captures)]
    result={'harness':'legacy' if legacy else 'v2','seed':seed,'rounds':rounds,'captures_per_arm':captures,
            'replay_checks_per_capture':replay_checks,'warnings':warnings,
            'source_md5':before,'prepare_choices':{a.name:a.choices for a in arms},
            'prepare_order':prepare_order,'capture_order':capture_order,'pad_seed':pad_seed,'pad_bytes':pad_bytes,
            'alloc_offsets_bytes':offsets,
            'flush':{'kind':'read torch.sum','bytes':cold_buffer.numel()*4},'raw':raw,
            'median_us':{a.name:statistics.median(r['us'] for r in raw if r['arm']==a.name) for a in arms},
            'median_paired_speedup':statistics.median(ratios),
            'timing':'External CUDA event nodes inside captured replay; no fixed-cost subtraction',
            'gpu':torch.cuda.get_device_name()}
    if legacy:
        result.update(saving_us=interval(saving,seed),aa_saving_us=interval(aa_saving,seed),
                      speedup=interval(ratios,seed),aa_speedup=interval(aa_ratios,seed))
        effect=result['saving_us']['ci95'];noise=result['aa_saving_us']['ci95']
        result['resolved_gain']=effect[0]>max(0,noise[1])
        return result
    # v2: the capture (memory layout + graph) is the unit of replication.
    result.update(saving_us=capture_interval(by_capture(saving)),aa_saving_us=capture_interval(by_capture(aa_saving)),
                  speedup=capture_interval(by_capture(ratios)),aa_speedup=capture_interval(by_capture(aa_ratios)),
                  saving_us_round_bootstrap=interval(saving,seed),aa_saving_us_round_bootstrap=interval(aa_saving,seed))
    aa_means=result['aa_saving_us']['per_capture']
    result['aa_distribution']={'per_capture_means_us':aa_means,'max_abs_mean_us':max(abs(x) for x in aa_means),
                               'sd_us':statistics.stdev(aa_means),'mean_us':statistics.mean(aa_means)}
    effect=result['saving_us']['ci95'];noise=result['aa_saving_us']['ci95']
    result['resolved_gain']=effect[0]>max(0,noise[1])
    # lead 22:1x: a gain only counts with >= 8 captures per arm (A/A layout noise is under-covered below that)
    result['counted_gain']=bool(captures>=8 and not legacy and effect[0]>0 and result['saving_us']['mean']>result['aa_distribution']['max_abs_mean_us'])
    if captures<8 or legacy: result['counted_gain_refused']=f'captures={captures} legacy={legacy}: counted_gain needs v2 with >= 8 captures; for states too large for 8, run >= 2 separate processes with captures=4 (different seeds) and count the gain with swarm/kit/timing/pool_runs.py (counted_gain_pooled)'
    return result
