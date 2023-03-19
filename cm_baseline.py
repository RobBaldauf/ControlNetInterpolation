import pdb
import shutil
from share import *

import os
import numpy as np
import torch
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from controlnet.annotator.openpose import util

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
        
    return interp

def interp_poses(pose_md1, pose_md2, alpha = 0.5):
    candidate = []
    subset = [-1] * 20
    for i in range(18):
        j = int(pose_md1['subset'][0][i])
        k = int(pose_md2['subset'][0][i])
        if j == -1 or k == -1:
            candidate.append([-1,-1,0,i])
            subset[i] = -1
            continue
        candidate.append([pose_md1['candidate'][j][0] * alpha + pose_md2['candidate'][k][0] * (1-alpha),
            pose_md1['candidate'][j][1] * alpha + pose_md2['candidate'][k][1] * (1-alpha),
            0,i])
        subset[i] = i
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, np.array(candidate), np.array([subset]))
    return canvas

class ContextManager:
    def __init__(self):
        self.filters = {}
        self.mode = None
        self.model = create_model('./controlnet/models/cldm_v15.yaml').cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def change_mode(self, mode):
        if self.mode == mode:
            return
        
        if mode not in self.filters:
            if mode == 'pose':
                self.filters[mode] = OpenposeDetector()
            elif mode == 'canny':
                self.filters[mode] = CannyDetector()
            elif mode == 'seg':
                self.filters[mode] = UniformerDetector()
        
        if mode == 'pose':
            self.model.load_state_dict(load_state_dict('./controlnet/models/control_sd15_openpose.pth', location='cuda'))
        elif mode == 'canny':
            self.model.load_state_dict(load_state_dict('./controlnet/models/control_sd15_canny.pth', location='cuda'))
        elif mode == 'seg':
            self.model.load_state_dict(load_state_dict('./controlnet/models/control_sd15_seg.pth', location='cuda'))

    def get_canny(self, image, lower_bound=220, upper_bound=255):
        self.change_mode('canny')
        canny = self.filters['canny'](HWC3(np.array(image)), lower_bound, upper_bound)
        return canny
        
    def get_pose(self, image, return_metadata=False):
        self.change_mode('pose')
        pred_pose, metadata = self.filters['pose'](HWC3(np.array(image)))
        if return_metadata:
            return pred_pose, metadata
        return pred_pose
        
    @torch.no_grad()
    def interpolate_pose(self, img1, pose_md1, img2, pose_md2, prompt, n_prompt,
                    ddim_steps=100, num_frames=17, steps_per_level=5, guide_scale=7.5):
        """
        ddim_steps: number of steps in DDIM sampling
        num_frames: includes endpoints (both original images)
        steps_per_level: each successive level adds this many more ddim steps
        """
        if isinstance(img1, Image.Image):
            img1 = torch.tensor(np.array(img1)).float().cuda() / 127.5 - 1.0
            img2 = torch.tensor(np.array(img2)).float().cuda() / 127.5 - 1.0
        ldm = self.model
        self.ddim_sampler.make_schedule(ddim_steps)
        num_levels = int(np.log2(num_frames-1)) # does not include endpoints
        assert np.log2(num_frames-1) % 1 < 1e-5
        assert ddim_steps >= num_levels * steps_per_level
        timesteps = self.ddim_sampler.ddim_timesteps[steps_per_level:num_levels * steps_per_level+1:steps_per_level]

        self.change_mode('pose')
        H = W = 512
        shape = (4, H // 8, W // 8)
        ldm.control_scales = [1] * 13
        cond = {"c_crossattn": [ldm.get_learned_conditioning([prompt])]}
        un_cond = {"c_crossattn": [ldm.get_learned_conditioning([n_prompt])]}

        latents1, latents2 = self.get_latent_stack(img1, img2, timesteps)
        latents = [None] * num_frames
        df = num_frames - 1
        t_by_frame = [None] * num_frames
        start_ix = (num_levels-1) * steps_per_level
        end_ix = num_levels * steps_per_level
        for level in range(num_levels):
            level = (num_frames-1)//df
            latents[0] = latents1[-level]
            latents[-1] = latents2[-level]
            df //= 2

            for frame_ix in range(df, num_frames-1, df*2):
                t_by_frame[frame_ix] = timesteps[level]
                frac = .5
                if frame_ix-df == 0:
                    frac -= .1
                if frame_ix+df == num_frames-1:
                    frac += .1
                latents[frame_ix] = interpolate_spherical(latents[frame_ix-df], latents[frame_ix+df], frac)

            if level == 2:
                latents[num_frames//2] = interpolate_spherical(latents[num_frames//4], latents[3*num_frames//4], .5)
            
            for frame_ix in range(df, num_frames-1, df): # exclude endpoints
                t = torch.tensor([t_by_frame[frame_ix]], dtype=torch.long, device='cuda')
                frac = frame_ix/(num_frames-1)
                pose_img = interp_poses(pose_md1, pose_md2, alpha=frac).transpose(2,0,1)
                control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0

                cond["c_concat"] = [control]
                un_cond["c_concat"] = [control]
                latents[frame_ix], _ = self.ddim_sampler.sample(ddim_steps, 1,
                    shape, cond, verbose=False, x_T=latents[frame_ix], timesteps=t.item(),
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)
                
                time_range = np.flip(timesteps[start_ix:end_ix])
                for i, step in enumerate(time_range):
                    index = end_ix - i - 1
                    ts = torch.tensor([step], device='cuda', dtype=torch.long)

                    latents[frame_ix] = self.ddim_sampler.p_sample_ddim(latents[frame_ix], cond, ts, index=index, unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)[0]

            start_ix -= steps_per_level
            end_ix -= steps_per_level 

        for frame_ix in range(1,num_frames-1):
            x_samples = ldm.decode_first_stage(latents[frame_ix]).permute(0, 2, 3, 1)
            x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:02d}.png')
    
    def get_latent_stack(self, img1, img2, timesteps):
        ldm = self.model
        latents1 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.permute(2,0,1).unsqueeze(0)))]
        latents2 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.permute(2,0,1).unsqueeze(0)))]
        
        t_prev = None
        for t_now in timesteps:
            noise = torch.randn_like(latents1[-1])
            latents1.append(self.add_more_noise(latents1[-1], noise, t_now, t_prev))
            latents2.append(self.add_more_noise(latents2[-1], noise, t_now, t_prev))
            t_prev = t_now
        return latents1[::-1], latents2[::-1]
    
    @torch.no_grad()
    def img2img(self, control, prompt, n_prompt, init_img=None, latents=None, mode=None, time_frac=0.3,
                ddim_steps=50, ctrl_scale=1, guide_scale=7.5, eta=0):
        if mode is not None:
            self.change_mode(mode)
        elif self.mode is None:
            print('no mode set')
            return
        
        control = torch.from_numpy(control).float().cuda().unsqueeze(0) / 255.0
        if len(control.shape) == 3:
            control = control.tile(1, 3, 1, 1)
        ldm = self.model
        if init_img is not None:
            if isinstance(init_img, Image.Image):
                init_img = torch.tensor(np.array(init_img)).float().cuda() / 127.5 - 1.0
            latents = ldm.get_first_stage_encoding(ldm.encode_first_stage(init_img.permute(2,0,1).unsqueeze(0)))
        
        T = int(time_frac * ldm.num_timesteps)
        t = torch.tensor([T], dtype=torch.long, device='cuda')
        noise = torch.randn_like(latents)
        x_T = (extract_into_tensor(ldm.sqrt_alphas_cumprod, t, latents.shape) * latents +
            extract_into_tensor(ldm.sqrt_one_minus_alphas_cumprod, t, latents.shape) * noise)
            
        cond = {"c_concat": [control], "c_crossattn": [ldm.get_learned_conditioning([prompt])]}
        un_cond = {"c_concat": [control], "c_crossattn": [ldm.get_learned_conditioning([n_prompt])]}

        H = W = 512
        shape = (4, H // 8, W // 8)

        ldm.control_scales = [ctrl_scale] * 13
        samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
            shape, cond, verbose=False, eta=eta, x_T=x_T, timesteps=int(time_frac * ddim_steps),
            unconditional_guidance_scale=guide_scale,
            unconditional_conditioning=un_cond)

        x_samples = ldm.decode_first_stage(samples).permute(0, 2, 3, 1)
        x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        return x_samples[0]

    @torch.no_grad()
    def generate(self, control, prompt, n_prompt, num_samples=1,
                ddim_steps=50, ctrl_scale=1, guide_scale=7.5, eta=0):
        control = torch.from_numpy(control).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = control.permute(0, 3, 1, 2)

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}

        H = W = 512
        shape = (4, H // 8, W // 8)

        self.model.control_scales = [ctrl_scale] * 13
        samples, _ = self.ddim_sampler.sample(ddim_steps, num_samples,
                                shape, cond, verbose=False, eta=eta,
                                unconditional_guidance_scale=guide_scale,
                                unconditional_conditioning=un_cond)

        x_samples = self.model.decode_first_stage(samples).permute(0, 2, 3, 1)
        x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        return [x_samples[i] for i in range(num_samples)]

def main():
    cm = ContextManager()
    img1 = Image.open('05.png').convert('RGB').resize((512, 512))
    img2 = Image.open('j1.jpeg').convert('RGB').resize((512, 512))
    pose1 = cm.get_pose(img1)
    pose2 = cm.get_pose(img2)

    prompt = 'mulan, disney cartoon, extremely detailed, ultra hd'
    num_samples = 1
    strength = 1
    ddim_steps = 60
    scale = 9
    n_prompt = 'long body, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, bad art, poorly drawn, low quality'
    cm.interpolate()
