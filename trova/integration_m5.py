"""Offline evidence/compatibility tools. No robot connection or motion entry point."""
import argparse
import json
from pathlib import Path
import time

from .scan3d.evidence import workspace, catalog, calibration_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'calibration-replay', 'orbit-replay'])
    parser.add_argument('--workspace')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = workspace(args.workspace)
    if args.action == 'audit':
        result = catalog(root)
    elif args.action == 'calibration-replay':
        result = calibration_replay(root)
    else:
        from .scan3d.orbit_planner import recorded_context, plan_orbit
        retained,context=recorded_context(root)
        result=plan_orbit(retained['q0'],context,sequential=True)
        result['provenance']='recorded_replanned'
    if args.output:
        output = Path(args.output).resolve()
        if output.is_relative_to(root):
            parser.error('Outputs must be outside the retained workspace.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
