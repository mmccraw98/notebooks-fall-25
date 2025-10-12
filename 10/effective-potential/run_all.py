import subprocess

if __name__ == "__main__":
    subprocess.run([
        "python",
        "jam_disk.py"
    ], check=True)

    subprocess.run([
        "python",
        "jam.py"
    ], check=True)