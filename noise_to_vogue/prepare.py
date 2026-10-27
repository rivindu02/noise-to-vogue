"""Download and verify Fashion-MNIST before training or using the viewer."""
from .data import load_data


def main():
    for train in (True, False):
        x, y = load_data(train=train)
        print(f"{'train' if train else 'test'}: {tuple(x.shape)}, {len(y)} labels, range [{x.min().item()}, {x.max().item()}]")


if __name__ == "__main__":
    main()
