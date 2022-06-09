import subprocess
import sys
import easyocr

args = sys.argv[1:]

NUMBER_OF_IMAGES = 6

# The purpose of this script is for testing all the images in parallel.
# NOTE: standard output is messy, this script is not intended for marking.
def main():
    script_name = "CS373LicensePlateDetection.py"
    if args != [] and args[0] == '-extension':
        model = easyocr.Reader(['en'])
        script_name = "CS373_extension.py"
    
    commands = []

    for i in range(NUMBER_OF_IMAGES):
        commands.append("python " + script_name + " numberplate" + str(i + 1) + ".png")

    processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
    for p in processes: 
        p.wait()


if __name__ == "__main__":
    main()
