from main import parse, train
from src.model import DRModel

if __name__ == "__main__":
    args = parse(print_help=True)
    args.train_fill_unknown = True,
    args.dataset_name = "Fdataset"
    args.disease_neighbor_num = 3
    args.drug_neighbor_num = 3
    args.epochs = 64
    args.train_fill_unknown = False


    train(args, DRModel)


