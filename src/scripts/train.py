from aml4cv.model import load_model


def train() -> None:
    print("Training from aml4cv scripts!")
    model = load_model()


if __name__ == "__main__":
    train()
