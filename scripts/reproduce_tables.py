"""Reconstruct existing cohort tables from released per-case scalar measurements."""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.aggregation import all_subjects,absolute,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('reproduced_tables'));args=p.parse_args()
    root=Path(__file__).resolve().parents[1];receipt=verify(root);sets=all_subjects(root);args.output.mkdir(parents=True,exist_ok=True)
    for name,values in sets.items():
        target=args.output/(name.lower().replace('/','_')+'.csv');rows=absolute(values)
        with target.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'status':'passed','tables_written':len(sets),'verification':receipt}))
if __name__=='__main__':main()
