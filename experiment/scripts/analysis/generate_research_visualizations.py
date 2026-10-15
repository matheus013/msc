"""Render the five requested article diagnostics without running experiments."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.visualizations import generate_research_visualizations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run name')
    output=ROOT/'experiment/data/08_reporting/research_v2'/args.run
    report=generate_research_visualizations(output)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
