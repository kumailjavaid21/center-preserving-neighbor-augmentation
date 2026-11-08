"""Train one prescribed continuation from separately supplied fixed parents."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--stream',required=True,type=int)
    p.add_argument('--prepared-data',required=True);p.add_argument('--parent-checkpoint',required=True)
    p.add_argument('--noise-checkpoint',required=True);p.add_argument('--center-checkpoint',required=True)
    p.add_argument('--anchor-cache',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    from src.training import train
    train(args)
if __name__=='__main__':main()
