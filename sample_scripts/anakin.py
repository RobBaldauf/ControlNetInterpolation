import sys
from PIL import Image
import os, pickle
import pdb
osp = os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/ana1.png').resize((768, 768))
img2 = Image.open('data/ana2.png').resize((768, 768))
# pose_path = 'data/ana_poses.pk'
# if osp.exists(pose_path):
#     pose1, pose2 = pickle.load(open(pose_path, 'rb'))
# else:
#     p1, pose1 = CM.get_pose(img1, return_metadata=True)
#     Image.fromarray(p1).save('pose1.png')

#     canny2 = CM.get_canny(img2, lower_bound=500)
#     Image.fromarray(canny2).save('canny2.png')

#     prompt = 'close-up portrait of a person, face, headshot, photorealistic, detailed face'
#     n_prompt = 'watermark, text, writing, signature, jpeg, artifacts, blurry, cartoon, painting'
#     out2 = CM.img2img(control=canny2, prompt=prompt, n_prompt=n_prompt, init_img=img2, mode='canny', time_frac=0.98, ddim_steps=50)
#     Image.fromarray(out2).save('out2.png')

#     p2, pose2 = CM.get_pose(out2, return_metadata=True)
#     Image.fromarray(p2).save('pose2.png')
#     pickle.dump((pose1, pose2), open(pose_path, 'wb'))

prompt = 'portrait of anakin, darth vader, ultra hd, high quality, 8k wallpaper, cinematic, artstation, star wars, highly detailed'
n_prompt = 'text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'
# CM.interpolate(img1, img2, controls=(pose1, pose2), cond_path='data/ana.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=200, max_steps=200, ddim_steps=250, num_frames=17, guide_scale=10, out_dir='anakin')

# baseline
# CM.interpolate_imgs(img1, img2, prompt=prompt, n_prompt=n_prompt,
#                     optimize_cond=0, num_frames=17, guide_scale=10, out_dir='anakin_baseline')
qc_prompt = 'portrait, centered, hyperrealistic, unreal engine, cinematic'
qc_neg_prompt = 'text, signature, logo, distorted, ugly, bad anatomy, weird face, weird eyes, asymmetrical face, bad anatomy, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/ana100.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=100, ddim_steps=200, num_frames=17, guide_scale=10, schedule_type='linear', out_dir='anakin_clip')
