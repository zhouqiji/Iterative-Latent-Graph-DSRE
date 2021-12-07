import argparse



def main(args):
    config = load_config(args.config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'test'])
    parser.add_argument('--show_example', action='store_true', help='Show an example')
    args = parser.parse_args()
    main(args)
