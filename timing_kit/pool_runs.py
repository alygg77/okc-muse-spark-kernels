"""Pool paired.py v2 results from >= 2 SEPARATE processes (each with captures >= 4, its own seed) and count a gain on the
pooled per-capture means. For states too large for 8 captures x 3 arms in one process.

    python3 -I pool_runs.py --out pooled.json run_a.json run_b.json [run_c.json ...] [--key shape,m] [--ignore-pad]

Each input is a paired() result dict, a list of them, or a dict with a 'runs' / 'results' list. Results are grouped by their
caller-added identifying fields (default: every one present of shape, case, label, name, m, bs, seed_index, variant, stages,
delay, tiles, repeat) plus pad_seed (so each pad is pooled on its own); --key names the grouping fields instead (pad_seed is still
added unless --ignore-pad). Per group the pooled gain is counted only if: >= 2 inputs, harness 'v2', identical source_md5 in every
input, distinct seeds (the random allocation offsets are seeded from the seed), >= 8 pooled captures, pooled-gain CI low > 0
(95% t over the pooled per-capture means) AND the pooled gain mean > max |A/A per-capture mean| over all pooled captures.
Output field `counted_gain_pooled` (false with `refused` when a rule fails). Needs no GPU or torch."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
_PAIRED=HERE/'paired.py' if (HERE/'paired.py').exists() else HERE/'timing'/'paired.py'
_spec=importlib.util.spec_from_file_location('kit_paired',_PAIRED);_paired=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_paired)
capture_interval=_paired.capture_interval

AUTO_KEYS=('shape','case','label','name','m','bs','seed_index','variant','stages','delay','tiles','repeat')

def results_in(obj):
    if isinstance(obj,list):
        for x in obj:yield from results_in(x)
    elif isinstance(obj,dict):
        if 'saving_us' in obj and 'aa_saving_us' in obj:yield obj
        else:
            for k in ('runs','results'):
                if k in obj:yield from results_in(obj[k])

def group_key(r,keys,ignore_pad):
    fields=[k for k in (keys or AUTO_KEYS) if k in r]
    key={k:r[k] for k in fields}
    if not ignore_pad:key['pad_seed']=r.get('pad_seed')
    return key

def pool_group(key,items):
    """items: [(path, result)]. Returns the group record."""
    rec={'key':key,'inputs':[p for p,_ in items],'n_processes':len(items),'counted_gain_pooled':False}
    def refuse(msg):
        rec['refused']=msg;return rec
    if len(items)<2:return refuse('needs >= 2 separate processes (inputs) for this group')
    rs=[r for _,r in items]
    if any(r.get('harness')!='v2' for r in rs):return refuse('every input must be a paired.py v2 result (harness == "v2")')
    if any('per_capture' not in r['saving_us'] or 'per_capture' not in r['aa_saving_us'] for r in rs):return refuse('input lacks per_capture means')
    if any(r['source_md5']!=rs[0]['source_md5'] for r in rs):return refuse('source_md5 differs between inputs: the arms are not the same bytes')
    seeds=[r['seed'] for r in rs]
    if len(set(seeds))!=len(seeds):return refuse(f'seeds repeat {seeds}: identical random offsets/schedule, not separate layouts')
    gain_caps=[x for r in rs for x in r['saving_us']['per_capture']]
    aa_caps=[x for r in rs for x in r['aa_saving_us']['per_capture']]
    gain=capture_interval(gain_caps);aa=capture_interval(aa_caps)
    speed=capture_interval([x for r in rs for x in r['speedup']['per_capture']])
    mx=max(abs(x) for x in aa_caps)
    rec.update(seeds=seeds,captures_per_process=[r['captures_per_arm'] for r in rs],n_captures=len(gain_caps),
               gain_us=gain,aa_us=aa,speedup=speed,aa_max_abs_per_capture_us=mx,
               median_us={a:statistics.mean(r['median_us'][a] for r in rs) for a in rs[0]['median_us']},
               per_process_gain_mean_us=[r['saving_us']['mean'] for r in rs],per_process_aa_mean_us=[r['aa_saving_us']['mean'] for r in rs],
               resolved_gain_pooled=bool(gain['ci95'][0]>max(0,aa['ci95'][1])))
    if len(gain_caps)<8:return refuse(f'only {len(gain_caps)} pooled captures (< 8)')
    rec['counted_gain_pooled']=bool(gain['ci95'][0]>0 and gain['mean']>mx)
    return rec

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files',nargs='+');ap.add_argument('--out',default='pooled.json');ap.add_argument('--key',default=None)
    ap.add_argument('--ignore-pad',action='store_true')
    a=ap.parse_args()
    keys=a.key.split(',') if a.key else None
    groups={}
    for path in a.files:
        data=json.loads(Path(path).read_text());found=list(results_in(data))
        if not found:sys.exit(f'{path}: no paired() results found')
        seen=set()
        for r in found:
            k=json.dumps(group_key(r,keys,a.ignore_pad),sort_keys=True)
            if k in seen:sys.exit(f'{path}: two results share the group key {k}; pass --key with more fields')
            seen.add(k);groups.setdefault(k,[]).append((path,r))
    out={'tool':'pool_runs.py','tool_md5':hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
         'paired_md5':hashlib.md5(_PAIRED.read_bytes()).hexdigest(),
         'inputs':{p:hashlib.md5(Path(p).read_bytes()).hexdigest() for p in a.files},
         'groups':[pool_group(json.loads(k),v) for k,v in groups.items()]}
    Path(a.out).write_text(json.dumps(out,indent=2))
    for g in out['groups']:
        if 'gain_us' in g:
            print(f"{json.dumps(g['key'])}: procs {g['n_processes']} caps {g['n_captures']} gain {g['gain_us']['mean']:.3f} us CI [{g['gain_us']['ci95'][0]:.3f}, {g['gain_us']['ci95'][1]:.3f}] "
                  f"max|A/A cap| {g['aa_max_abs_per_capture_us']:.3f} -> counted_gain_pooled {g['counted_gain_pooled']}"+(f" (REFUSED: {g['refused']})" if 'refused' in g else ''))
        else:print(f"{json.dumps(g['key'])}: counted_gain_pooled False (REFUSED: {g['refused']})")

if __name__=='__main__':main()
