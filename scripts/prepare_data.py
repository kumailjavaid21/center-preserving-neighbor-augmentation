"""Prepare explicitly supplied TRAIN/development volumes; no archive scanning."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    from src.io import ROOT,read
    from src.restoration.preprocessing import prepare_subject
    population=read(ROOT/'metadata/development_population.json');expected={x['subject']:{**x,'split':split} for key,split in [('train','train'),('development','validation')] for x in population[key]}
    manifest=Path(args.manifest);records=read(manifest)['subjects']
    if len(records)!=80 or len({x['subject'] for x in records})!=80 or {x['subject'] for x in records}!=set(expected):raise ValueError('Exactly published TRAIN64/development16 required')
    for record in records:
        sid=record['subject'];path=Path(record['path'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Manifest file paths must be relative to its directory')
        if not path.name.startswith(sid+'-'+expected[sid]['site']+'-') or not path.name.endswith('-T1.nii.gz'):raise ValueError('Subject/site/modality filename mismatch')
        prepare_subject(manifest.parent/path,expected[sid],Path(args.output)/sid)
if __name__=='__main__':main()
