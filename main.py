# main.py
import argparse
from data.collect_data import collect_dataset
from data.dataset_report import run_master_report
from model.train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="GNN Query Runtime Pipeline")

    # Flags to control which parts of the pipeline run
    parser.add_argument('--collect', action='store_true', help='Run data collection')
    parser.add_argument('--train', action='store_true', help='Run training and evaluation')

    # Parameters
    parser.add_argument('--size', type=int, default=20000, help='Queries to collect')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')

    args = parser.parse_args()

    if args.collect:
        print("Collecting Data:")
        collect_dataset(size=args.size)
        print("Dataset Report:")
        run_master_report("query_data.json")


    if args.train:
        print("Training Model:")
        run_training_pipeline(epochs=args.epochs)

if __name__ == "__main__":
    main()