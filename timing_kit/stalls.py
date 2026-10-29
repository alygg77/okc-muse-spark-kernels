"""Summarize Nsight CSV or CUPTI Chrome trace; always emits a waits: field."""
import argparse
import csv
import io
import json
from pathlib import Path

METRICS=','.join(['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum']+
    [f'smsp__average_warps_issue_stalled_{x}_per_issue_active.ratio'
     for x in ['barrier','long_scoreboard','short_scoreboard','wait','membar','math_pipe_throttle','not_selected','selected']])

def ncu_summary(text):
    lines=text.splitlines()
    start=next(i for i,l in enumerate(lines) if 'Metric Name' in l and 'Kernel Name' in l)
    kernels={}
    for r in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        if not r.get('Metric Name'):continue
        try:v=float(r['Metric Value'].replace(',',''))
        except ValueError:continue
        k=kernels.setdefault((r.get('ID',''),r['Kernel Name']),{})
        k[r['Metric Name']]=(v,r.get('Metric Unit',''))
    output=[]
    for (ident,name),metrics in kernels.items():
        stalls=sorted([(key.split('stalled_')[1].split('_per_issue')[0],v[0])
                       for key,v in metrics.items() if 'stalled_' in key and not any(x in key for x in ['not_selected','stalled_selected'])],
                      key=lambda x:x[1],reverse=True)
        duration,unit=metrics.get('gpu__time_duration.sum',(0,'nsecond'))
        ns=duration*{'nsecond':1,'usecond':1000,'msecond':1e6,'second':1e9}.get(unit,1)
        moved=sum(metrics.get(k,(0,''))[0] for k in ['dram__bytes_read.sum','dram__bytes_write.sum'])
        output.append({'kernel':name,'id':ident,'us':ns/1000,'dram_tb_s':moved/ns/1000 if ns else None,
                       'stalls_ratio':stalls,'waits':'waits: '+(', '.join(f'{k}={v:.3f} ratio' for k,v in stalls[:3]) or 'no stall counters')})
    return {'kind':'ncu','kernels':output,'note':'Issue-active warp-state ratios are not fractions of kernel wall time.'}

def cupti_summary(trace):
    kernels={}
    for e in trace.get('traceEvents',[]):
        if e.get('cat') not in ('kernel','gpu_kernel') or e.get('ph')!='X':continue
        kernels[e['name']]=kernels.get(e['name'],0)+e.get('dur',0)
    top=sorted(kernels.items(),key=lambda x:x[1],reverse=True)
    return {'kind':'cupti','kernel_sum_us':sum(kernels.values()),'kernels':top,
            'waits':'waits: CUPTI dominant kernels '+', '.join(f'{n} {us:.2f} us' for n,us in top[:3])+
                    '; timeline durations only, hardware stall reasons unavailable'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('path',nargs='?');ap.add_argument('--command',action='store_true')
    args=ap.parse_args()
    if args.command:
        print(f'ncu --clock-control none --target-processes all --kernel-name regex:kit_ --csv --metrics {METRICS} --log-file ncu.csv python3 selftest.py --profile-only --ncu')
        return
    p=Path(args.path)
    result=cupti_summary(json.loads(p.read_text())) if p.suffix=='.json' else ncu_summary(p.read_text())
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
