"""Offline evidence/compatibility tools. No robot connection or motion entry point."""
import argparse
import json
from pathlib import Path
import time

from .scan3d.evidence import workspace, catalog, calibration_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'calibration-replay'])
    parser.add_argument('--workspace')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = workspace(args.workspace)
    result = catalog(root) if args.action == 'audit' else calibration_replay(root)
    if args.output:
        output = Path(args.output).resolve()
        if output.is_relative_to(root):
            parser.error('Outputs must be outside the retained workspace.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
