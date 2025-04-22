# Pill Verification Project – Raspberry Pi Demo

This project contains scripts used in our pill verification prototype, combining image-based recognition, load cell readings, and motorized dispensing via a stepper motor.

Key Files

 `combination.py`
- Used for image capture and classification of pills using a trained PyTorch CNN model.
- To view directory on raspberry pi use ls
- To edit a file use nano
- To change directory use cd
- It is important to read below and activate the python environment to be able to run any of these scripts
- Double check that the model is in /home/pi as well as combination.py otherwise you need to edit the directory in the code
- For any general questions contact andrewdodds0715@gmail.com
  
`main_controller.py`
- Allows user to run:
  - `combination.py` (image classification)
  - `loadcell.py` (weight measurement via load cell)
  - `stepper_control.py` (lock/unlock pill bottle)

Before running any scripts, activate your python environment. You may have different environments depending on your setup:


source pill_verification_env/bin/activate
source pillenv/bin/activate
This is dependent on which raspberry pi you are using just use type ls in /home/pi and see which environment you have
To check the libraries in the environment use pip list
To make a new environment use venv but you will need to reinstall libraries
In order to actually run a script do python3 name_of_script_here.py

To actually train new models use feb_model_training.py and feel free to change the type of neural network
You will need a python environment in your IDE with all the required libraries listed in the python file
You will also need to change the directory to wherever you are storing the images
It is important to note that the version of torch, torchvision, python, etc all libraries must have the same version in your IDE AND the raspberry pi
