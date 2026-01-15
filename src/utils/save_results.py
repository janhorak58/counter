import os
import csv


def save_results_to_csv(results, video_filename, model_name, output_folder=None):
    base_filename = os.path.splitext(os.path.basename(video_filename))[0]
    csv_filename = f"{base_filename}_{model_name}_results.csv"

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = os.path.join(output_folder, csv_filename)

    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["line_name", "in_count", "out_count"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for line_name, counts in results.items():
            writer.writerow(
                {
                    "line_name": line_name,
                    "in_count": counts["in"],
                    "out_count": counts["out"],
                }
            )

    print(f"Results saved to {csv_filename}")
