import time
import RPi.GPIO as GPIO
from hx711 import HX711

class Scale():
    def __init__(self):
        self.DT = 4  # Data pin (GPIO4)
        self.SCK = 27  # Clock pin (GPIO27)
        self.hx = HX711(self.DT, self.SCK)
        self.OFFSET = 0
        self.SCALE_FACTOR = 59.15
        self.WEIGHT_OFFSET = -2.9
        self.VREF = 5.0

    def get_raw_data(self):
        raw_data = self.hx.get_raw_data()
        if raw_data:
            return raw_data[-1]
        return None

    def raw_to_voltage(self, raw_data):
        max_raw_value = 8388607
        voltage = (raw_data / max_raw_value) * self.VREF
        return voltage - self.OFFSET

    def voltage_to_weight(self, voltage):
        weight_mg = (voltage * self.SCALE_FACTOR) - self.WEIGHT_OFFSET
        return weight_mg

def main():
    try:
        s = Scale()
        print("Preparing to read weight for 10 seconds...")
        pill_weight_g = float(input("Enter the weight of a single pill (g): "))
        readings = []

        start_time = time.time()
        while time.time() - start_time < 10:
            raw_data = s.get_raw_data()
            if raw_data is not None:
                voltage = s.raw_to_voltage(raw_data)
                weight = s.voltage_to_weight(voltage)
                num_pills = round(weight / pill_weight_g) if pill_weight_g > 0 else 0
                print(f"Raw: {raw_data}, Voltage: {voltage:.6f} V, Weight: {abs(weight):.4f} g, Estimated Pills: {num_pills}")
                readings.append(num_pills)
            time.sleep(1)

        if readings:
            avg_pills = round(sum(readings) / len(readings))
            print(f"\n Average Estimated Number of Pills: {avg_pills}")
        else:
            print("No valid readings received.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
