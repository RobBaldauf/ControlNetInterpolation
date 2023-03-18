import pdb
import shutil
from share import *

import os
import numpy as np
import torch
from PIL import Image

from tqdm import trange
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
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """ 
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

def to_pil(rgb):
    if len(rgb.shape) == 3:
        return Image.fromarray((rgb*255).detach().cpu().numpy().astype('uint8'))
    return Image.fromarray((rgb[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8'))

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
        candidate.append([pose_md1['candidate'][j][0] * (1-alpha) + pose_md2['candidate'][k][0] * alpha,
            pose_md1['candidate'][j][1] * (1-alpha) + pose_md2['candidate'][k][1] * alpha,
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
        
    def interpolate_pose(self, img1, pose_md1, img2, pose_md2, prompt, n_prompt, num_frames,
                    ddim_steps=100, guide_scale=7.5, optimize_cond=0): #steps_per_frame=10, 
        """
        ddim_steps: number of steps in DDIM sampling
        num_frames: includes endpoints (both original images)
        steps_per_frame: each successive level adds this many more ddim steps
        """
        if isinstance(img1, Image.Image):
            img1 = torch.tensor(np.array(img1)).float().cuda() / 127.5 - 1.0
            img2 = torch.tensor(np.array(img2)).float().cuda() / 127.5 - 1.0
        # if ddim_steps < num_frames*steps_per_frame/2:
        #     steps_per_frame = 2*ddim_steps // num_frames
        #     print(f'lowering steps_per_frame to {steps_per_frame}')
        
        self.change_mode('pose')
        ldm = self.model
        H = W = 512
        shape = (4, H // 8, W // 8)
        # schedules include endpoints
        self.ddim_sampler.make_schedule(ddim_steps, verbose=False)
        step_schedule = [int(8*x**.5) for x in np.linspace(0, 70, (num_frames+1)//2)]
        timestep_schedule = [self.ddim_sampler.ddim_timesteps[s] for s in step_schedule]
        latents1, latents2 = self.get_latent_stack(img1, img2, timestep_schedule)
        latents = [None] * num_frames
        latents[0] = latents1[0]
        latents[-1] = latents2[0]
        
        cond_base = ldm.get_learned_conditioning([prompt])
        uncond_base = ldm.get_learned_conditioning([n_prompt])
        cond = {"c_crossattn": [cond_base], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

        if optimize_cond:
            T = np.random.randint(10,20)
            eta = 0.
            self.ddim_sampler.make_schedule(T, ddim_eta=eta, verbose=False)
            uncond_base.requires_grad_(True)
            cond1 = cond_base
            cond2 = cond_base.clone()
            cond1.requires_grad_(True)
            cond2.requires_grad_(True)
            optimizer = torch.optim.Adam([cond1, cond2, uncond_base], lr=1e-5)
            for cur_iter in range(optimize_cond):
                with torch.autocast('cuda'):
                    cond["c_crossattn"] = [cond1]
                    t_now = self.ddim_sampler.ddim_timesteps[T]
                    t_prev = self.ddim_sampler.ddim_timesteps[T-1]
                    x_t = ldm.sqrt_alphas_cumprod[t_prev] * latents[0] + \
                        ldm.sqrt_one_minus_alphas_cumprod[t_prev] * torch.randn_like(latents[0])
                    x_T = self.add_more_noise(x_t, torch.randn_like(x_t), t_now, t_prev)
                    ts = torch.full((1,), t_now, device='cuda', dtype=torch.long)
                    pdb.set_trace()
                    outs = self.ddim_sampler.p_sample_grad(img, cond, ts, index=u,  unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                    img, pred_x0 = outs
                    loss1 = (latents[0] - samples).abs().mean()
                    loss1.backward()

                    # samples, _ = self.ddim_sampler.sample_grad(T, 1, shape, cond, verbose=False, eta=eta, x_T=x_T, timesteps=u, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                    # # grad = latents[0] - samples
                    # # samples.backward(grad)
                    # loss1 = (latents[0] - samples).abs().mean()
                    # loss1.backward()

                    cond["c_crossattn"] = [cond2]
                    u = np.random.randint(T//3, 2*T//3)
                    t = self.ddim_sampler.ddim_timesteps[u]
                    x_T = ldm.sqrt_alphas_cumprod[t] * latents[-1] + ldm.sqrt_one_minus_alphas_cumprod[t] * torch.randn_like(latents[-1])
                    samples, _ = self.ddim_sampler.sample_grad(T, 1, shape, cond, verbose=False, eta=eta, x_T=x_T, timesteps=u, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                    loss2 = (latents[-1] - samples).norm()
                    loss2.backward()

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if cur_iter % 3 == 0:
                        print(f'iter {cur_iter}: {loss1.item()}, {loss2.item()}')

            cond1.requires_grad_(False)
            cond2.requires_grad_(False)
            uncond_base.requires_grad_(False)

        with torch.no_grad():
            self.ddim_sampler.make_schedule(ddim_steps, verbose=False)
            for frame_ix in trange(1,num_frames-1):
                frac = frame_ix/(num_frames-1)
                f = min(frame_ix, num_frames - frame_ix - 1)
                ldm.control_scales = [0.5 + 3*f/num_frames] * 13 # range from 0.5 to 2
                latents[frame_ix] = interpolate_spherical(latents1[f], latents2[f], frac)
                pose_img = interp_poses(pose_md1, pose_md2, alpha=frac).transpose(2,0,1)
                control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0

                cond["c_concat"] = [control]
                un_cond["c_concat"] = [control]
                if optimize_cond:
                    cond["c_crossattn"] = [cond1 * (1-frac) + cond2 * frac]
                samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
                    shape, cond, verbose=False,
                    x_T=latents[frame_ix], timesteps=step_schedule[f],
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)

                x_samples = ldm.decode_first_stage(samples).permute(0, 2, 3, 1)
                x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(x_samples[0]).save(f'blend/{frame_ix:02d}.png')

    
    def get_latent_stack(self, img1, img2, timesteps):
        ldm = self.model
        latents1 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.permute(2,0,1).unsqueeze(0)))]
        latents2 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.permute(2,0,1).unsqueeze(0)))]
        
        shutil.rmtree('blend', ignore_errors=True)
        os.makedirs('blend')
        t_prev = None
        for t_now in timesteps[1:]:
            noise = torch.randn_like(latents1[-1])
            latents1.append(self.add_more_noise(latents1[-1], noise, t_now, t_prev))
            latents2.append(self.add_more_noise(latents2[-1], noise, t_now, t_prev))
            t_prev = t_now
        return latents1, latents2
    
    def add_more_noise(self, latents, noise, t2, t1=None):
        ldm = self.model
        if t1 is None:
            return ldm.sqrt_alphas_cumprod[t2] * latents + \
                ldm.sqrt_one_minus_alphas_cumprod[t2] * noise

        a1 = ldm.sqrt_alphas_cumprod[t1]
        sig1 = ldm.sqrt_one_minus_alphas_cumprod[t1]
        a2 = ldm.sqrt_alphas_cumprod[t2]
        sig2 = ldm.sqrt_one_minus_alphas_cumprod[t2]

        scale = a2/a1
        sigma = (sig2**2 - (scale * sig1)**2).sqrt()
        return scale * latents + sigma * noise
    
    @torch.no_grad()
    def img2img(self, control, prompt, n_prompt, init_img=None, noise=None, mode=None, time_frac=0.3,
                ddim_steps=50, ctrl_scale=1, guide_scale=7.5, eta=0):
        if mode is not None:
            self.change_mode(mode)
        elif self.mode is None:
            print('no mode set')
            return
        
        if not isinstance(control, torch.Tensor):
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
        noisy_latents = (extract_into_tensor(ldm.sqrt_alphas_cumprod, t, latents.shape) * latents +
            extract_into_tensor(ldm.sqrt_one_minus_alphas_cumprod, t, latents.shape) * noise)
            
        cond = {"c_concat": [control], "c_crossattn": [ldm.get_learned_conditioning([prompt])]}
        un_cond = {"c_concat": [control], "c_crossattn": [ldm.get_learned_conditioning([n_prompt])]}

        H = W = 512
        shape = (4, H // 8, W // 8)

        ldm.control_scales = [ctrl_scale] * 13
        samples, _ = self.ddim_sampler.sample_grad(ddim_steps, 1,
            shape, cond, verbose=False, eta=eta, x_T=noisy_latents, timesteps=int(time_frac * ddim_steps),
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
