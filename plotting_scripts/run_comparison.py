import subprocess
import os
import sys

def run_all_plots():
    """
    This script runs the synoptic comparison plot for all weather types from 0 to 10.
    It calls the 'plot_comparison.py' script for each weather type.
    """
    
    # Define the path to the main plotting script
    plotting_script = "plot_comparison.py"
    
    # Check if the plotting script exists
    if not os.path.exists(plotting_script):
        print(f"Error: The plotting script '{plotting_script}' was not found.")
        print("Please make sure both scripts are in the same directory.")
        return

    # Define the output directory and create it if it doesn't exist
    output_dir = "./figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Loop through all weather types ---
    weather_types_to_run = range(11) # This will create a range from 0 to 10
    
    print("--- Starting to generate all synoptic comparison plots ---")
    
    for wt in weather_types_to_run:
        print(f"==> Generating plot for Weather Type: {wt}")
        
        # Construct the command to be executed
        command = [
            sys.executable,             # Use the same python interpreter that is running this script
            plotting_script,
            "--weather_type", str(wt),
            "--output_dir", output_dir
        ]
        
        try:
            # Execute the command
            # We use subprocess.run to wait for the command to complete
            process = subprocess.run(
                command, 
                check=True,              # This will raise an exception if the script returns a non-zero exit code
                capture_output=True,     # Capture stdout and stderr
                text=True                # Decode stdout/stderr as text
            )
            
            # Print the output from the script, which includes the "Saved figure" message
            print(process.stdout)
            if process.stderr:
                print("Stderr:", process.stderr)

        except FileNotFoundError:
            print(f"Error: Could not find the command '{sys.executable}'. Make sure Python is in your PATH.")
            break
        except subprocess.CalledProcessError as e:
            # This block will execute if the plotting script fails for any reason
            print(f"---!!! An error occurred while running for Weather Type: {wt} !!!---")
            print(f"Return code: {e.returncode}")
            print("Stdout:")
            print(e.stdout)
            print("Stderr:")
            print(e.stderr)
            print("---!!! Aborting the process. Please check the error message above. !!!---")
            break # Stop the loop if one of the plots fails
            
    else:
        # This 'else' block runs only if the loop completes without 'break'
        print("\n--- All plots generated successfully! ---")

if __name__ == "__main__":
    run_all_plots()
