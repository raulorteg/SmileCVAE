from Pathlib import Path

def setup_directories():
    """
    Create all directories the other functions expect to work with.
    Execute by calling python setup_directories.py
    """
    queue_to_make = ["results/axis", "results/changed", "results/sampled", "results/iterations",
                    "results/plots", "model_savepoints"]
    [Path(directory).mkdir(parents=True, exist_ok=True) for directory in queue_to_make]

if __name__ == "__main__":
    setup_directories()
    print("Done! Directories are now created.")