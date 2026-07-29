import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_filename):
    print(f"Reading notebook '{notebook_filename}'...")
    with open(notebook_filename, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Configure Execution preprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    print("Running all notebook cells. This might take a couple of minutes...")
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
        print("All notebook cells executed successfully without any error!")

        # Save executed notebook to see outputs in the file
        output_filename = notebook_filename.replace(".ipynb", "_executed.ipynb")
        with open(output_filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Saved executed notebook to '{output_filename}'.")

        # Rename executed notebook back to neural_network_tutorial.ipynb to keep the repo clean & ready
        os.replace(output_filename, notebook_filename)
        print(f"Overwrote '{notebook_filename}' with the fully executed version.")
        return True
    except Exception as e:
        print(f"Error executing the notebook: {e}")
        return False

if __name__ == "__main__":
    success = run_notebook("neural_network_tutorial.ipynb")

    # Verify outputs exist
    loss_chart_exists = os.path.exists("loss_comparison_chart.png")
    scatter_chart_exists = os.path.exists("prediction_scatters.png")
    video_exists = os.path.exists("nn_learning_process.mp4")

    print("\n--- Output Verification ---")
    print(f"loss_comparison_chart.png exists: {loss_chart_exists}")
    print(f"prediction_scatters.png exists: {scatter_chart_exists}")
    print(f"nn_learning_process.mp4 exists: {video_exists}")

    if success and loss_chart_exists and scatter_chart_exists and video_exists:
        print("\nSUCCESS: All verifications passed!")
        exit(0)
    else:
        print("\nFAILURE: One or more verifications failed.")
        exit(1)
