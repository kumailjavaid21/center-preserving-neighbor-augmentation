"""Verify public file hashes and scalar aggregation without scientific dependencies."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.aggregation import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    manifest=json.loads((args.root/'metadata/public_result_manifest.json').read_text())
    for rel,row in manifest['files'].items():
        b=(args.root/rel).read_bytes()
        if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('Public file hash mismatch: '+rel)
    result=verify(args.root);result['public_files_hashed']=len(manifest['files']);print(json.dumps(result))
if __name__=='__main__':main()
