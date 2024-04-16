import os
import subprocess
import sys
from tqdm import tqdm
import time

def install_dependencies():
    try:
        import cv2
    except ImportError:
        print("cv2 is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        print("cv2 has been installed.")

def display_menu():
    print("Vehicle Analysis")
    print("================")
    print("1. Image Classification")
    print("2. Vehicle Detection")
    print("3. Exit")
    print()

def run_classification(input_folder):
    command = f"python vehicle-classification.py {input_folder}"
    subprocess.run(command, shell=True)

def run_detection(input_video):
    command = f"python vehicle-detection.py {input_video}"
    subprocess.run(command, shell=True)

def main():
    install_dependencies()
    
    print("Loading, please wait...")
    for _ in tqdm(range(10)):
        time.sleep(0.1)  
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-3): ")
        print()

        if choice == "1":
            input_folder = input("Enter the path to the image folder: ")
            if os.path.isdir(input_folder):
                run_classification(input_folder)
            else:
                print("Invalid folder path.")
        elif choice == "2":
            input_video = input("Enter the path to the input video: ")
            if os.path.isfile(input_video):
                run_detection(input_video)
            else:
                print("Invalid video file path.")
        elif choice == "3":
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
        print()

if __name__ == "__main__":
    main()
