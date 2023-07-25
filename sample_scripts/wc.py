import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/wc1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/wc2.png').convert('RGB').resize((768, 768))

pose_path = 'data/wc_poses.pk'
# os.remove(pose_path)
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    canny1 = CM.get_canny(img1)
    prompt = 'photo of a couple smiling, facing the camera, standing side by side, photorealistic'
    n_prompt = 'blurry, cartoon, painting, sign, logo'
    Image.fromarray(canny1).save('canny1.png')
    out1 = CM.generate(control=canny1, prompt=prompt, n_prompt=n_prompt, mode='canny', guide_scale=50, eta=0.2, ctrl_scale=.5)
    Image.fromarray(out1).save('out1.png')
    p1, pose1 = CM.get_pose(out1, return_metadata=True, filter_largest=False)
    Image.fromarray(p1).save('pose1.png')

    p2, pose2 = CM.get_pose(img2, return_metadata=True, filter_largest=False)
    Image.fromarray(p2).save('pose2.png')
    
    pickle.dump((pose1, pose2), open(pose_path, 'wb'))
CM.visualize_poses(poses=(pose1, pose2), num_frames=17, out_dir='wc_poses')
    
prompt = 'man and woman'
n_prompt = 'text, signature, distorted, lowres, messy, lopsided, disfigured, low quality'

qc_prompt = 'two people, symmetrical'
qc_neg_prompt = 'text, signature, distorted, ugly'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/wc500.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=500, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='wc_clip')
