import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/armstrong1.png').resize((768, 640))
img2 = Image.open('data/armstrong2.jpeg').resize((768, 640))

pose_path = 'data/armstrong_poses.pk'
if osp.exists(pose_path):
    os.remove(pose_path)
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    p1, pose1 = CM.get_pose(img1, return_metadata=True)
    Image.fromarray(p1).save('a_pose1.png')

    p2, pose2 = CM.get_pose(img2, return_metadata=True, filter_largest=False)
    pose2['subset'] = pose2['subset'][1:]
    # Image.fromarray(p2).save('a_pose2.png')
    # canny2 = CM.get_canny(img2, lower_bound=250)
    # prompt = 'photo of a person playing the trumpet, photorealistic'
    # n_prompt = 'blurry, cartoon, painting'
    # Image.fromarray(canny2).save('a_canny2.png')
    # out2 = CM.img2img(control=canny2, prompt=prompt, n_prompt=n_prompt, init_img=img2, mode='canny', time_frac=0.9)
    # Image.fromarray(out2).save('a_out2.png')

    pickle.dump((pose1, pose2), open(pose_path, 'wb'))


prompt = 'photo of a person playing the trumpet, photorealistic, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, messy, weird face, lopsided, disfigured, low quality'

qc_prompt = 'portrait, face, high quality photo, high resolution, ultra HD, detailed'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, low quality'
CM.interpolate_qc(img1, img2, controls=(pose1, pose2), qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/armstrong500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.35, max_steps=.5, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='armstrong_clip')
