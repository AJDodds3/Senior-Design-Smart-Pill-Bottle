import subprocess
import sys

def run_combination():
    print("\nRunning Image Capture + Classification...\n")
    subprocess.run(["python3", "combination.py"])
    input("\nDone. Press Enter to return to menu.")

def run_loadcell():
    print("\nRunning Weight Check...\n")
    subprocess.run(["python3", "loadcell.py"])
    input("\nDone. Press Enter to return to menu.")

def run_stepper():
    print("\nRunning Stepper Motor Script...\n")
    subprocess.run(["python3", "stepper_control.py"])
    input("\nDone. Press Enter to return to menu.")

def main_menu():
    while True:
        print("\n--- Raspberry Pi Demo Controller ---")
        print("Press 1 to take an image and classify")
        print("Press 2 to check pill weight")
        print("Press 3 to open container (stepper)")
        print("Press q to quit")
        choice = input("Your choice: ").strip()

        if choice == '1':
            run_combination()
        elif choice == '2':
            run_loadcell()
        elif choice == '3':
            run_stepper()
        elif choice.lower() == 'q':
            print("Exiting controller.")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main_menu()
