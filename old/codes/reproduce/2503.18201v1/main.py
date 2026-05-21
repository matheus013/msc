import argparse
from training.train_base import train_base_model
from training.train_marl import train_marl_model
from training.train_imarl import train_imarl_model
from training.train_gnn import train_sarl_gnn_model, train_marl_gnn_model

def main():
    parser = argparse.ArgumentParser(description="MEIO Optimization via DRL")
    parser.add_argument("--model", type=str, required=True,
                        choices=["sarl", "sarl+gnn", "marl", "marl+gnn", "imarl"],
                        help="Which model to train")
    parser.add_argument("--scenario", type=str, default="A1",
                        help="Scenario ID (e.g., A1, B3, D1)")
    args = parser.parse_args()

    if args.model == "sarl":
        train_base_model(args.scenario)
    elif args.model == "sarl+gnn":
        train_sarl_gnn_model(args.scenario)
    elif args.model == "marl":
        train_marl_model(args.scenario)
    elif args.model == "marl+gnn":
        train_marl_gnn_model(args.scenario)
    elif args.model == "imarl":
        train_imarl_model(args.scenario)
    else:
        raise ValueError("Unknown model selected")

if __name__ == "__main__":
    main()
