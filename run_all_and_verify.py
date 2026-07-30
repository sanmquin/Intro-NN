import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_filename):
    print(f"Reading notebook '{notebook_filename}'...")
    with open(notebook_filename, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Configure Execution preprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    print(f"Running all cells in '{notebook_filename}'. This might take a couple of minutes...")
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
        print(f"All cells in '{notebook_filename}' executed successfully without any error!")

        # Save executed notebook to see outputs in the file
        output_filename = notebook_filename.replace(".ipynb", "_executed.ipynb")
        with open(output_filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Saved executed notebook to '{output_filename}'.")

        # Rename executed notebook back to notebook_filename to keep the repo clean & ready
        os.replace(output_filename, notebook_filename)
        print(f"Overwrote '{notebook_filename}' with the fully executed version.")
        return True
    except Exception as e:
        print(f"Error executing the notebook: {e}")
        return False

if __name__ == "__main__":
    # Standard Notebook
    success_std = run_notebook("neural_network_tutorial.ipynb")
    # Deep Notebook
    success_deep = run_notebook("deep_neural_network_tutorial.ipynb")
    # Architecture Variation Notebook
    success_arch = run_notebook("architecture_variation_tutorial.ipynb")

    # Verify standard outputs exist
    loss_chart_exists = os.path.exists("loss_comparison_chart.png")
    scatter_chart_exists = os.path.exists("prediction_scatters.png")
    video_exists = os.path.exists("nn_learning_process.mp4")

    # Verify deep outputs exist
    deep_loss_chart_exists = os.path.exists("deep_loss_comparison_chart.png")
    deep_scatter_chart_exists = os.path.exists("deep_prediction_scatters.png")
    deep_video_exists = os.path.exists("deep_nn_learning_process.mp4")

    # Verify architecture variation outputs exist
    arch_loss_chart_exists = os.path.exists("architecture_loss_comparison.png")
    arch_cost_chart_exists = os.path.exists("architecture_cost_metrics.png")
    arch_scatter_chart_exists = os.path.exists("architecture_prediction_scatters.png")
    arch_video_exists = os.path.exists("architecture_learning_process.mp4")

    print("\n--- Output Verification ---")
    print(f"loss_comparison_chart.png exists: {loss_chart_exists}")
    print(f"prediction_scatters.png exists: {scatter_chart_exists}")
    print(f"nn_learning_process.mp4 exists: {video_exists}")
    print()
    print(f"deep_loss_comparison_chart.png exists: {deep_loss_chart_exists}")
    print(f"deep_prediction_scatters.png exists: {deep_scatter_chart_exists}")
    print(f"deep_nn_learning_process.mp4 exists: {deep_video_exists}")
    print()
    print(f"architecture_loss_comparison.png exists: {arch_loss_chart_exists}")
    print(f"architecture_cost_metrics.png exists: {arch_cost_chart_exists}")
    print(f"architecture_prediction_scatters.png exists: {arch_scatter_chart_exists}")
    print(f"architecture_learning_process.mp4 exists: {arch_video_exists}")

    all_success = success_std and success_deep and success_arch
    all_files_exist = (loss_chart_exists and scatter_chart_exists and video_exists and
                       deep_loss_chart_exists and deep_scatter_chart_exists and deep_video_exists and
                       arch_loss_chart_exists and arch_cost_chart_exists and arch_scatter_chart_exists and arch_video_exists)

    if all_success and all_files_exist:
        print("\nSUCCESS: All verifications passed!")
        exit(0)
    else:
        print("\nFAILURE: One or more verifications failed.")
        exit(1)
