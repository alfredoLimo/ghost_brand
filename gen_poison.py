import os
import glob
import sys
from PIL import Image
import glob
import argparse
import pickle
from torchvision import transforms
from opt import PoisonGeneration


def main():
    poison_generator = PoisonGeneration(target_concept=args.target_name, device="cuda:1", eps=args.eps)
    # all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_data_paths = sorted(
    glob.glob(os.path.join(args.directory, '*.p')),
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    all_imgs = [Image.fromarray(img) for img in all_imgs]

    all_result_imgs = poison_generator.generate_all(all_imgs, args.target_name)
    os.makedirs(args.outdir, exist_ok=True)

    for idx, cur_img in enumerate(all_result_imgs):
        cur_data = {"text": all_texts[idx], "img": cur_img}
        pickle.dump(cur_data, open(os.path.join(args.outdir, "{}.p".format(idx)), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help="", default='clean_data/')
    parser.add_argument('-od', '--outdir', type=str, help="", default='poisoned_data/')
    # parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('-e', '--eps', type=float, default=0.1)

    parser.add_argument('-t', '--target_name', type=str, default="cat")
    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    main()
