import time
import RPi.GPIO
from hx711 import HX711

class Scale():
	def __init__(self):
		#Pin Configuration for HX711
		self.DT = 4  #Data pin (GPIO4)
		self.SCK =27 # Clock pin (GPIO6)
		# Initialize HX711
		self.hx = HX711(self.DT, self.SCK)
		# Define offset 
		self.OFFSET = 0
		# Define scale factor
		self.SCALE_FACTOR = 59.15
		#Define weight offset
		self.WEIGHT_OFFSET = -2.9

		# Reference voltage 
		self.VREF = 5.0 #5V
		#self.calibration()
	
	def get_raw_data(self):
		# Get the raw data (digital value)
		raw_data = self.hx.get_raw_data()
		if raw_data:
			return raw_data[-1] #Return the most recent data point
		return None
	
	def raw_to_voltage(self,raw_data):
		# Convert raw data to voltage (rough estimate)
		# HX711 outputs data as an integer, which is scaled based on reference voltage
		max_raw_value = 8388607 #24-bit is max
		voltage = (raw_data/max_raw_value)*self.VREF
		return voltage - self.OFFSET

	def voltage_to_weight(self,voltage):
		# Convert voltage to weight using the scale factor
		weight_mg = (voltage * self.SCALE_FACTOR) - self.WEIGHT_OFFSET
		return weight_mg

	def calibration(self):
		# Step 1: Run automatically, measure voltage at 0 weight (this is v0)
		v0 = self.raw_to_voltage(self.get_raw_data() or 0)
		# w0 = mv0+b, v0=voltage, w0=0 --> 0 = m*v0+b -> m=-b/v0
		# Step 2: Wait for weight applied above threshold
		w_temp = 0 
		while w_temp < 20:
			w_temp = self.voltage_to_weight(self.raw_to_voltage(self.get_raw_data() or 0))
		time.sleep(0.1)
		# Step 3: Measure voltage at applied weight (v1), assuming standard and known (ex w1 = 50g)
		v1 = self.raw_to_voltage(self.get_raw_data() or 0)
		# Step 4: Apply point slope formula 50g = m(v1-v0)+b, m is scale factor, b is offset
		self.OFFSET = w0
		self.SCALE_FACTOR = (w1-2*self.WEIGHT_OFFSET)/50	
	
def main():
	try:
		s = Scale()	
		print("Reading voltage and weight. Press Ctrl+C to stop")
		pill_weight_g = float(input("Enter the weight of a single pill (g): "))
		while True:
			raw_data = s.get_raw_data()
			if raw_data is not None:
				voltage = s.raw_to_voltage(raw_data)
				weight = s.voltage_to_weight(voltage)
				print(f"Raw data: {raw_data}, Voltage:{voltage:.6f} V, Weight: {abs(weight):.4f} g")
				num_pills = round(weight / pill_weight_g) if pill_weight_g > 0 else 0
				print(f"Estimated number of pills: {num_pills}")

			time.sleep(1)

	except KeyboardInterrupt:
		print("Exiting...")
		GPIO.cleanup()
if __name__ == "__main__":
	main()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
	

	
