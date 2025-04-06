import time
import board
import digitalio

from adafruit_motor import stepper

MUTE = False

DELAY = 2e-3
STEPS = 530

'''if not MUTE:
	print(stepper.BACKWARD)
	print(stepper.INTERLEAVE)
	print(stepper.DOUBLE)
'''
coils = (
	digitalio.DigitalInOut(board.D19),
	digitalio.DigitalInOut(board.D26),
	digitalio.DigitalInOut(board.D20),
	digitalio.DigitalInOut(board.D21),
)

for coil in coils:
	coil.direction = digitalio.Direction.OUTPUT

motor = stepper.StepperMotor(coils[0], coils[1], coils[2], coils[3], microsteps=None)

try:
	'''for step in range(STEPS):
		motor.onestep()
		time.sleep(DELAY)

	if not MUTE:
		print("forward complete")

	for step in range (STEPS):
		motor.onestep(direction=stepper.BACKWARD)
		time.sleep(DELAY)

	if not MUTE:
        	print(f"backward  complete")'''

	for step in range (STEPS):
		motor.onestep(style=stepper.DOUBLE)
		time.sleep(DELAY)

	if not MUTE:
		print("double  complete")


	time.sleep(3)

	for step in range(STEPS ):
		motor.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
		time.sleep(DELAY)

	if not MUTE:
		print("backward and double  complete")

	'''for step in range(STEPS):
		motor.onestep(style=stepper.INTERLEAVE)
		time.sleep(DELAY)

	if not MUTE:
        	print("interleave  complete")
	time.sleep(0.1)
	for step in range(STEPS):
		motor.onestep(direction=stepper.BACKWARD, style=stepper.INTERLEAVE)
		time.sleep(DELAY)

	if not MUTE:
		print("backward and interleave  complete")'''

finally:
	motor.release()
