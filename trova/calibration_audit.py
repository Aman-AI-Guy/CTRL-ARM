"""Read-only calibration inspection: optional Pi capture and guarded xArm reads, never an arm driver."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .geom import VisionStore, pose_changed
from .localization import make_router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--camera-host', default='trova-cam.local')
    parser.add_argument('--camera-port', type=int, default=8080)
    parser.add_argument('--robot-ip')
    parser.add_argument('--pixel', type=float, nargs=2)
    parser.add_argument('--measured', type=float, nargs=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    router = make_router(VisionStore.load(args.data / 'vision_calib.json'), args.data, args.candidate)
    result = {'motion_sent': False, 'motors_enabled': False, 'tcp_changed': False, 'live_requested': args.live}
    size = None
    if args.live:
        from .pi_cam import PiCam
        from .arm.telemetry import open_reader
        reader = None
        try:
            reader = open_reader(f'xarm:{args.robot_ip}') if args.robot_ip else None
            before = reader.read() if reader else {}
            cam = PiCam(args.camera_host, args.camera_port, timeout=4, retries=0, status_timeout=2)
            status = cam.status()
            jpeg = cam.get_jpeg()
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Pi snapshot does not decode')
            robot = reader.read() if reader else {}
            if robot.get('pose6') and robot.get('joints'):
                from .geom import tool_pose_matrix
                from .scan3d.flange import flange_pose
                result['controller_inferred_flange_tcp'] = (np.linalg.inv(flange_pose(robot['joints'])) @ tool_pose_matrix(robot['pose6'])).tolist()
                result['inferred_tcp_is_independent_measurement'] = False
            if reader and reader._inner:
                result['sdk_read_calls'] = list(reader._inner.sdk.calls)
            size = [image.shape[1], image.shape[0]]
            camera = {'kind': 'pi', 'host': args.camera_host, 'rotate_reported': status.get('rotate'),
                      'extra_rotate': 0, 'size': size, 'provenance': 'fresh_raspberry_pi_snapshot'}
            router.set_context(camera, robot)
            result.update(camera=camera, camera_status=status, robot=robot,
                          capture_stationary=bool(before and robot and not before.get('moving') and not robot.get('moving')
                              and not pose_changed(before.get('pose6'), robot.get('pose6'), mm=1, deg=.1)
                              and before.get('joints') and robot.get('joints')
                              and np.max(np.abs(np.subtract(before['joints'], robot['joints']))) <= .1),
                          captured_at=time.time(), jpeg_bytes=len(jpeg), sync_precision_claimed=False)
            if args.pixel and args.measured and router.historical and result['capture_stationary']:
                cal = router.historical.cal
                if size != [cal.width, cal.height] or status.get('rotate') != 180:
                    raise ValueError('Live optical profile/rotation does not match the historical candidate')
                measured = np.asarray(args.measured)
                predicted = cal.pixel_to_plane(*args.pixel, robot['joints'], measured[2])
                result['independent_comparison'] = {'diagnostic_point_mm': predicted.tolist(), 'measured_mm': args.measured,
                    'error_mm': float(np.linalg.norm(predicted-measured)), 'metric_permitted': False, 'provider_activated': False}
        except Exception as exc:  # noqa: BLE001 - a disconnected station remains unqualified
            result['live_error'] = str(exc)
        finally:
            if reader:
                reader.close()
    result['calibration'] = router.assess(size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
