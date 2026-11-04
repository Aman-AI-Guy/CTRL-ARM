
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.100"  # Replace with actual IP

arm = XArmAPI(ROBOT_IP, is_radian=False)

try:
    print("Connected:", arm.connected)

    if not arm.connected:
        raise ConnectionError("Robot not reachable")

    print("Version:", arm.get_version())
    print("State:", arm.get_state())
    print("Errors:", arm.get_err_warn_code())
    print("Position:", arm.get_position())
    print("Joint angles:", arm.get_servo_angle())

finally:
    arm.disconnect()
