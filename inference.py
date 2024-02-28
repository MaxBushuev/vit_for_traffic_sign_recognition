import argparse

from transformers import pipeline


def main(args):
    pipe = pipeline(task="image-classification", model=args.model_name_or_path)

    print(pipe(args.image_path)[0]["label"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--image_path", type=str, default="stop.jpg")

    args = parser.parse_args()

    main(args)
