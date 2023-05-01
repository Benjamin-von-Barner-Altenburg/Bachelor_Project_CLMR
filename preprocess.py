import argparse
from tqdm import tqdm
from clmr.datasets import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="audio")
    parser.add_argument("--dataset_dir", type=str, default="data/Edm_Data")
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    train_dataset = get_dataset(args.dataset, args.dataset_dir + "/train", subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir + "/val", subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir + "/test", subset="test")

    for i in tqdm(range(len(train_dataset))):
        train_dataset.preprocess(i, args.sample_rate)

    for i in tqdm(range(len(valid_dataset))):
        valid_dataset.preprocess(i, args.sample_rate)

    for i in tqdm(range(len(test_dataset))):
        test_dataset.preprocess(i, args.sample_rate)
