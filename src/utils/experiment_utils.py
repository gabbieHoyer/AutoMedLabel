# src/utils/experiment_utils.py
import os

# ------- FUNCTION TO DETERMINE EXPERIMENT RUN OUTPUT DIRECTORY ------- #

def determine_run_directory(base_dir, task_name, group_name=None):
    """
    Determines the next run directory for storing experiment data.
    """
    if group_name !=None:
        base_path = os.path.join(base_dir, task_name, group_name)
    else:
        base_path = os.path.join(base_dir, task_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Filter for directories that start with 'Run_' and are followed by an integer
    existing_runs = []
    for d in os.listdir(base_path):
        if d.startswith('Run_') and os.path.isdir(os.path.join(base_path, d)):
            parts = d.split('_')
            if len(parts) == 2 and parts[1].isdigit():  # Check if there is a number after 'Run_'
                existing_runs.append(d)
    
    if existing_runs:
        # Sort by the integer value of the part after 'Run_'
        existing_runs.sort(key=lambda x: int(x.split('_')[-1]))
        last_run_num = int(existing_runs[-1].split('_')[-1])
        next_run_num = last_run_num + 1
    else:
        next_run_num = 1
    
    run_directory = f'Run_{next_run_num}'
    full_run_path = os.path.join(base_path, run_directory)
    os.makedirs(full_run_path, exist_ok=True)
    
    return full_run_path

def in_notebook():
    """Return True if running inside a Jupyter notebook, False otherwise."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        # If shell is ZMQInteractiveShell, then it's likely a Jupyter notebook or qtconsole.
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        # Other shell types: 'SpyderShell', etc.
        return False
    except ImportError:
        return False  # Probably standard Python interpreter
    except AttributeError:
        return False  # get_ipython() is None or shell is None
