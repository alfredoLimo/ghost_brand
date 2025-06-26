import os
import sys

import torch
from PIL import Image
import glob
import pickle
import argparse
from torchvision import transforms
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel


def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


class CLIP(object):
    def __init__(self):
        self.device = "cuda:1"
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

    def text_emb(self, text_ls):
        if isinstance(text_ls, str):
            text_ls = [text_ls]
        inputs = self.processor(text=text_ls, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

    def img_emb(self, img):
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features

    def __call__(self, image, text, softmax=False):
        text_input = [text] if isinstance(text, str) else text
        image_input = image if isinstance(image, list) else [image]
        inputs = self.processor(text=text_input, images=image_input, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            if softmax:
                return outputs.logits_per_image.softmax(dim=-1).cpu().numpy()
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        return similarity[0][0]


def main():
    clip_model = CLIP()
    data_dir = args.directory

    source_concept = args.concept
    os.makedirs(args.outdir, exist_ok=True)
    all_data = glob.glob(os.path.join(data_dir, "*.p"))
    res_ls = []
    for idx, cur_data_f in enumerate(all_data):
        cur_data = pickle.load(open(cur_data_f, "rb"))
        cur_img = Image.fromarray(cur_data["img"])
        cur_text = cur_data["text"]

        cur_img = crop_to_square(cur_img)
        score = clip_model(cur_img, "a photo of a {}".format(source_concept))
        if score > 0.24:
            res_ls.append((cur_img, cur_text))

    if len(res_ls) < args.num:
        Exception("Not enough data from the source concept to select from. Please add more in the folder. ")

    all_prompts = [d[1] for d in res_ls]
    text_emb = clip_model.text_emb(all_prompts)
    text_emb_target = clip_model.text_emb("a photo of a {}".format(source_concept))
    text_emb_np = text_emb.cpu().float().numpy()
    text_emb_target_np = text_emb_target.cpu().float().numpy()
    res = cosine_similarity(text_emb_np, text_emb_target_np).reshape(-1)
    candidate = np.argsort(res)[::-1][:300]
    random_selected_candidate = random.sample(list(candidate), args.num)
    final_list = [res_ls[i] for i in random_selected_candidate]
    for i, data in enumerate(final_list):
        img, text = data
        cur_data = {
            "img": np.array(img),
            "text": text,
        }
        pickle.dump(cur_data, open(os.path.join(args.outdir, "{}.p".format(i)), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str,
                        help="", default='data/')
    parser.add_argument('-od', '--outdir', type=str,
                        help="", default='clean_data/')
    parser.add_argument('-n', '--num', type=int,
                        help="", default=5)
    parser.add_argument('-c', '--concept', type=str, default="dog",
                        help="")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()

# download the sample from https://mirror.cs.uchicago.edu/fawkes/files/resources/example-data.zip