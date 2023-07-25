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
img1 = Image.open('data/phoen1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/phoen2.jpeg').convert('RGB').resize((768, 768))

# CM.interpolate_naive(img1, img2, num_frames=17, out_dir='phoenix_naive')
# exit()

pose_path = 'data/phoenix_poses.pk'
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    p1, pose1 = CM.get_pose(img1, return_metadata=True)
    # Image.fromarray(p1).save('pose1.png')
    p2, pose2 = CM.get_pose(img2, return_metadata=True)
    # Image.fromarray(p2).save('pose2.png')
    
    pickle.dump((pose1, pose2), open(pose_path, 'wb'))
# CM.visualize_poses(poses=(pose1, pose2), num_frames=17, out_dir='phoenix_poses')

prompt = 'Joaquin Phoenix, male actor, joker, cinematic portrait, detailed ultra HD, dramatic lighting'
n_prompt = 'text, signature, logo, distorted, ugly, asymmetrical, messy, complicated, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, 2D'

qc_prompt = 'cinematic portrait, symmetrical face, high quality' #movie character, 
qc_neg_prompt = 'text, signature, logo, distorted, ugly, poorly drawn, low quality, 2D, multiple faces'
CM.interpolate_qc(img1, img2, controls=(pose1, pose2), scale_control=3.5, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/phoenix400.pt', prompt=prompt, min_steps=.33, max_steps=.55, n_prompt=n_prompt, optimize_cond=400, ddim_steps=250, num_frames=65, guide_scale=10, ddim_eta=0.1, schedule_type='linear', out_dir='phoenix_clip')
