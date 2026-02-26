import json
import os

def generate_samples(input_file="query_data.json", output_dir="examples"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, "r") as f:
        data = json.load(f)

    # Sort data by runtime to get a variety of samples
    sorted_data = sorted(data, key=lambda x: x['runtime'])

    # Pick indices for: Fastest, 25th percentile, Median, 75th percentile, and Slowest
    indices = [
        0,
        len(sorted_data) // 4,
        len(sorted_data) // 2,
        (3 * len(sorted_data)) // 4,
        -1
    ]

    names = ["very_fast", "fast", "medium", "slow", "very_slow"]

    for idx, name in zip(indices, names):
        sample = sorted_data[idx]
        file_path = os.path.join(output_dir, f"{name}_plan.json")

        # We only save the 'plan' key because predict.py shouldn't see the actual runtime
        with open(file_path, "w") as f:
            json.dump(sample['plan'], f, indent=2)

        print(f"Generated {file_path} | True Runtime: {sample['runtime']:.2f}ms")

if __name__ == "__main__":
    generate_samples()