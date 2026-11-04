from xarm.wrapper import XArmAPI
import os

ROBOT_IP = os.environ["XARM_IP"]

arm = XArmAPI(ROBOT_IP, is_radian=False)

try:
    if not arm.connected:
        raise ConnectionError("Robot not connected")

    print("Connected:", arm.connected)
    print("Firmware:", arm.get_version())
    print("State:", arm.get_state())
    print("Errors:", arm.get_err_warn_code())
    print("Position:", arm.get_position())
    print("Joint angles:", arm.get_servo_angle())

finally:
    arm.disconnect()
