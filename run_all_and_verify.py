import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_filename):
    print(f"\n========================================================")
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

        # Rename executed notebook back to maintain clean & ready repo
        os.replace(output_filename, notebook_filename)
        print(f"Overwrote '{notebook_filename}' with the fully executed version.")
        return True
    except Exception as e:
        print(f"Error executing the notebook '{notebook_filename}': {e}")
        return False

if __name__ == "__main__":
    # 1. Run Standard Neural Network Tutorial
    success_std = run_notebook("neural_network_tutorial.ipynb")

    # 2. Run Deep Neural Network Tutorial
    success_deep = run_notebook("deep_neural_network_tutorial.ipynb")

    # 3. Run Parameter Study Tutorial
    success_param = run_notebook("parameter_study_tutorial.ipynb")

    # Verify standard outputs exist
    loss_chart_exists = os.path.exists("loss_comparison_chart.png")
    scatter_chart_exists = os.path.exists("prediction_scatters.png")
    video_exists = os.path.exists("nn_learning_process.mp4")

    # Verify deep outputs exist
    deep_loss_chart_exists = os.path.exists("deep_loss_comparison_chart.png")
    deep_scatter_exists = os.path.exists("deep_prediction_scatters.png")
    deep_std_scatter_exists = os.path.exists("deep_prediction_scatters_std.png")
    deep_video_exists = os.path.exists("deep_nn_learning_process.mp4")

    # Verify parameter study outputs exist
    param_heatmaps_exist = os.path.exists("parameter_study_heatmaps.png")
    param_pareto_exist = os.path.exists("parameter_study_pareto_frontiers.png")
    param_csv_exists = os.path.exists("parameter_study_sweep_results.csv")

    print("\n--- Output Verification ---")
    print(f"neural_network_tutorial.ipynb success: {success_std}")
    print(f"loss_comparison_chart.png exists: {loss_chart_exists}")
    print(f"prediction_scatters.png exists: {scatter_chart_exists}")
    print(f"nn_learning_process.mp4 exists: {video_exists}")
    print()
    print(f"deep_neural_network_tutorial.ipynb success: {success_deep}")
    print(f"deep_loss_comparison_chart.png exists: {deep_loss_chart_exists}")
    print(f"deep_prediction_scatters.png exists: {deep_scatter_exists}")
    print(f"deep_prediction_scatters_std.png exists: {deep_std_scatter_exists}")
    print(f"deep_nn_learning_process.mp4 exists: {deep_video_exists}")
    print()
    print(f"parameter_study_tutorial.ipynb success: {success_param}")
    print(f"parameter_study_heatmaps.png exists: {param_heatmaps_exist}")
    print(f"parameter_study_pareto_frontiers.png exists: {param_pareto_exist}")
    print(f"parameter_study_sweep_results.csv exists: {param_csv_exists}")

    all_success = (
        success_std and loss_chart_exists and scatter_chart_exists and video_exists and
        success_deep and deep_loss_chart_exists and deep_scatter_exists and deep_std_scatter_exists and deep_video_exists and
        success_param and param_heatmaps_exist and param_pareto_exist and param_csv_exists
    )

    if all_success:
        print("\nSUCCESS: All verifications passed!")
        exit(0)
    else:
        print("\nFAILURE: One or more verifications failed.")
        exit(1)
