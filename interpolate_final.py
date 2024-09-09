import pdb
import sys
from PIL import Image
import os, pickle
from distutils.dir_util import copy_tree

osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64M"
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
import cm

INPUT_IMG_DIR="final_png/"
OUTPUT_IMG_DIR="final_png_out/"
DRIVE_DIR="/content/drive/MyDrive/interpolation/final_png_out/"

img_paths=sorted(os.listdir(INPUT_IMG_DIR))
if len(img_paths)==0:
    raise ValueError("No images found in input dir!")
img_paths=[p for p in img_paths if ".jpg" in p]


CM = cm.ContextManager(version="1.5")
prompt = 'analog photo, water reflection, caustics, high resolution, highly detailed, ultra HD, 4k, warm monochrome image'
n_prompt = 'lowres, disfigured, messy, poorly drawn, low quality, blurry'
qc_prompt = 'analog photo, water reflection, caustics, high resolution, highly detailed, ultra HD, 4k, warm monochrome image'
qc_neg_prompt = 'lowres, disfigured, messy, poorly drawn, low quality, blurry'

for i, _ in enumerate(img_paths[:-1]):
    if i==5:
        pair_name = f"{img_paths[i].split('.')[0]}_{img_paths[i+1].split('.')[0]}"
        print(f"Processing {pair_name}")
        src_img=INPUT_IMG_DIR + img_paths[i]
        target_img=INPUT_IMG_DIR + img_paths[i+1]
        img1 = Image.open(src_img).convert('RGB').resize((768, 768))
        img2 = Image.open(target_img).convert('RGB').resize((768, 768))
        
        out_dir=OUTPUT_IMG_DIR + pair_name
        CM.interpolate_qc(img1, img2, n_choices=8, qc_prompts=(qc_prompt, qc_neg_prompt), prompt=prompt, n_prompt=n_prompt, min_steps=.25, max_steps=.45, optimize_cond=400, ddim_steps=200, num_frames=17, guide_scale=10, ddim_eta=.1, schedule_type='linear', out_dir=out_dir)
        copy_tree(OUTPUT_IMG_DIR, DRIVE_DIR)
