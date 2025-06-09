import subprocess
import time

scripts = [
    # "sa.py",
    "test_3.py",
    "ga.py",
    "grasp.py"
]

def run_script(script):
    print(f"\n▶️ Running {script}...")
    start = time.time()
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ {script} completed in {round(time.time() - start, 2)}s")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} failed with error:\n{e}")

if __name__ == "__main__":
    print("🚀 Starting full execution pipeline...")
    for script in scripts:
        run_script(script)
    print("\n🎯 All codes executed.")
