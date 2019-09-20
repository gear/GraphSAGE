import argparse

def main():
    parser = argparse.ArgumentParser("test parser")
    parser.add_argument('--hiddens', type=list)

    args = parser.parse_args()

    print(args.hiddens)

if __name__ == "__main__":
    main()
