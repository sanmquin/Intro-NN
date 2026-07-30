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
    success_std = run_notebook("0.neural_network_tutorial.ipynb")

    # 2. Run Deep Neural Network Tutorial
    success_deep = run_notebook("1.deep_neural_network_tutorial.ipynb")

    # 3. Run Parameter Study Tutorial
    success_param = run_notebook("2.parameter_study_tutorial.ipynb")

    # 4. Run Small 6-Neuron Network Interpretability Tutorial
    success_small = run_notebook("4.small_network_analysis_tutorial.ipynb")

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

    # Verify small network outputs exist
    small_loss_exists = os.path.exists("small_loss_comparison.png")
    small_scatters_exists = os.path.exists("small_prediction_scatters.png")
    small_heatmaps_exists = os.path.exists("small_connectivity_heatmaps.png")
    small_influences_exists = os.path.exists("small_neuron_influence_scatters.png")

    print("\n--- Output Verification ---")
    print(f"0.neural_network_tutorial.ipynb success: {success_std}")
    print(f"loss_comparison_chart.png exists: {loss_chart_exists}")
    print(f"prediction_scatters.png exists: {scatter_chart_exists}")
    print(f"nn_learning_process.mp4 exists: {video_exists}")
    print()
    print(f"1.deep_neural_network_tutorial.ipynb success: {success_deep}")
    print(f"deep_loss_comparison_chart.png exists: {deep_loss_chart_exists}")
    print(f"deep_prediction_scatters.png exists: {deep_scatter_exists}")
    print(f"deep_prediction_scatters_std.png exists: {deep_std_scatter_exists}")
    print(f"deep_nn_learning_process.mp4 exists: {deep_video_exists}")
    print()
    print(f"2.parameter_study_tutorial.ipynb success: {success_param}")
    print(f"parameter_study_heatmaps.png exists: {param_heatmaps_exist}")
    print(f"parameter_study_pareto_frontiers.png exists: {param_pareto_exist}")
    print(f"parameter_study_sweep_results.csv exists: {param_csv_exists}")
    print()
    print(f"4.small_network_analysis_tutorial.ipynb success: {success_small}")
    print(f"small_loss_comparison.png exists: {small_loss_exists}")
    print(f"small_prediction_scatters.png exists: {small_scatters_exists}")
    print(f"small_connectivity_heatmaps.png exists: {small_heatmaps_exists}")
    print(f"small_neuron_influence_scatters.png exists: {small_influences_exists}")

    all_success = (
        success_std and loss_chart_exists and scatter_chart_exists and video_exists and
        success_deep and deep_loss_chart_exists and deep_scatter_exists and deep_std_scatter_exists and deep_video_exists and
        success_param and param_heatmaps_exist and param_pareto_exist and param_csv_exists and
        success_small and small_loss_exists and small_scatters_exists and small_heatmaps_exists and small_influences_exists
    )

    if all_success:
        print("\nSUCCESS: All verifications passed!")
        exit(0)
    else:
        print("\nFAILURE: One or more verifications failed.")
        exit(1)
