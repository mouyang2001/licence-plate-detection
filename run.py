import subprocess
import sys

args = sys.argv[1:]

NUMBER_OF_IMAGES = 6

def main():
    script_name = "CS373LicensePlateDetection.py"
    if args != [] and args[0] == '-extension':
        script_name = "CS373_extension.py"
    
    commands = []

    for i in range(NUMBER_OF_IMAGES):
        commands.append("python " + script_name + " numberplate" + str(i + 1) + ".png")

    processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
    for p in processes: 
        p.wait()


if __name__ == "__main__":
    main()
