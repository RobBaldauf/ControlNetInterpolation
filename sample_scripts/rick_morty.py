import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/rick1.jpeg').resize((768, 640))
img2 = Image.open('data/rick2.jpeg').resize((768, 640))

pose_path = 'data/rick_poses.pk'
# os.remove(pose_path)
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    p1, pose1 = CM.get_pose(img1, return_metadata=True, filter_largest=False)
    Image.fromarray(p1).save('pose1.png')

    canny2 = CM.get_canny(img2, lower_bound=250)
    prompt = 'photo of two people driving a car, photorealistic'
    n_prompt = 'blurry, cartoon, painting'
    Image.fromarray(canny2).save('canny2.png')
    out2 = CM.generate(control=canny2, prompt=prompt, n_prompt=n_prompt, mode='canny', guide_scale=15, ctrl_scale=.6, ddim_steps=30)
    # out2 = CM.img2img(init_img=img2, prompt=prompt, n_prompt=n_prompt, guide_scale=15, ctrl_scale=.6, ddim_steps=30)
    Image.fromarray(out2).save('out2.png')
    p2, pose2 = CM.get_pose(out2, return_metadata=True, filter_largest=False)
    Image.fromarray(p2).save('pose2.png')
    
    pickle.dump((pose1, pose2), open(pose_path, 'wb'))
# CM.visualize_poses(poses=(pose1, pose2), num_frames=17, out_dir='rick_poses')

prompt = 'rick and morty, high quality, large eyes, expressive faces, 8k wallpaper, artstation, highly detailed'
n_prompt = 'text, signature, logo, distorted, cursed, scary, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'

# CM.interpolate(img1, img2, cond_path=None, prompt=prompt, n_prompt=n_prompt, optimize_cond=0, ddim_steps=250, num_frames=9, guide_scale=10, bias=.1, schedule_type='linear', out_dir='rickmorty_noopt')
# CM.interpolate(img1, img2, cond_path='data/rick300.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=300, min_steps=.3, ddim_steps=250, num_frames=17, guide_scale=10, bias=.1, retroactive_interp=False, schedule_type='linear', out_dir='rickmorty_300_1')
qc_prompt = 'rick and morty, faces, eyes, high quality, masterful cartoon, highly detailed, masterpiece, trending on artstation'
qc_neg_prompt = 'text, signature, logo, distorted, cursed, blurry, asymmetrical face, weird hands, messy, lopsided, disfigured, bad art, poorly drawn, low quality'
CM.interpolate_qc(img1, img2, controls=(pose1, pose2), scale_control=False, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/rick300.pt', optimize_cond=300, min_steps=.3, max_steps=.55, ddim_steps=200, num_frames=17, guide_scale=10, ddim_eta=.1, schedule_type='linear', out_dir='rickmorty_clip')
# CM.interpolate_then_diffuse(img1, img2, cond_path='data/rick200.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=200, ddim_steps=250, num_frames=17, guide_scale=10, schedule_type='linear', out_dir='rickmorty_itd')

# baseline
# CM.interpolate_imgs(img1, img2, prompt=prompt, n_prompt=n_prompt,
#                     optimize_cond=0, num_frames=17, guide_scale=10, out_dir='rickmorty_baseline')