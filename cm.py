import pdb
import shutil
from share import *

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import yaml
from tqdm import trange
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from controlnet.annotator.openpose import util

def get_step_schedule(max_steps, num_levels, schedule_type='convex'):
    if schedule_type == 'concave':
        return [int(max_steps * x**.5 / 10) for x in np.linspace(0, 100, num_levels+1)]
    elif schedule_type == 'convex':
        return [int(max_steps * x**2 / 16) for x in np.linspace(0, 4, num_levels+1)]
    elif schedule_type == 'linear':
        return [int(x) for x in np.linspace(0, max_steps, num_levels+1)]

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def interpolate_linear(p0,p1, frac):
    return p0 + (p1 - p0) * frac

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
        candidate.append([interpolate_linear(pose_md1['candidate'][j][0], pose_md2['candidate'][k][0], alpha),
            interpolate_linear(pose_md1['candidate'][j][1], pose_md2['candidate'][k][1], alpha),
            0,i])
        subset[i] = i
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, np.array(candidate), np.array([subset]))
    return canvas

class ContextManager:
    def __init__(self, version='2.1'):
        self.filters = {}
        self.mode = None
        self.version = version
        if version == '2.1':
            self.model = create_model('./controlnet/models/cldm_v21.yaml').cuda()
        else:
            self.model = create_model('./controlnet/models/cldm_v15.yaml').cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def init_mode(self):
        if self.mode is None:
            self.change_mode('pose')

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
            if self.version == '2.1':
                self.model.load_state_dict(load_state_dict('./controlnet/models/openpose-sd21.ckpt', location='cuda'))
            else:
                self.model.load_state_dict(load_state_dict('./controlnet/models/control_sd15_openpose.pth', location='cuda'))
        elif mode == 'canny':
            if self.version == '2.1':
                self.model.load_state_dict(load_state_dict('./controlnet/models/canny-sd21.ckpt', location='cuda'))
            else:
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
        
    def learn_conditioning(self, img1, img2, cond, cond_base, uncond_base, ddim_steps, guide_scale, num_iters=200, cond_lr=1e-4):
        # augment = transforms.TrivialAugmentWide(num_magnitude_bins=20)
        augment = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=(512,512), scale=(0.7,1.0)),
        ])

        cond = {"c_crossattn": [cond_base], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        ldm = self.model
        uncond_base.requires_grad_(True)
        cond1 = cond_base
        cond2 = cond_base.clone()
        cond1.requires_grad_(True)
        cond2.requires_grad_(True)
        optimizer = torch.optim.Adam([cond1, cond2, uncond_base], lr=cond_lr) #
        T = ddim_steps
        self.ddim_sampler.make_schedule(T, verbose=False)
        for cur_iter in range(num_iters):
            L1 = ldm.get_first_stage_encoding(ldm.encode_first_stage(augment(img1).float() / 127.5 - 1.0))
            L2 = ldm.get_first_stage_encoding(ldm.encode_first_stage(augment(img2).float() / 127.5 - 1.0))
            with torch.autocast('cuda'):
                u = np.random.randint(T//3, 2*T//3)
                t_u = self.ddim_sampler.ddim_timesteps[u]
                tu = torch.tensor([t_u], device='cuda', dtype=torch.long)

                cond["c_crossattn"] = [cond1]
                noise = torch.randn_like(L1)
                x_t_u = ldm.sqrt_alphas_cumprod[t_u] * L1 + \
                    ldm.sqrt_one_minus_alphas_cumprod[t_u] * noise
                eps = self.ddim_sampler.pred_eps(x_t_u, cond, tu, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                loss1 = (eps - noise).pow(2).mean()
                loss1.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                cond["c_crossattn"] = [cond2]
                noise = torch.randn_like(L2)
                x_t_u = ldm.sqrt_alphas_cumprod[t_u] * L2 + \
                    ldm.sqrt_one_minus_alphas_cumprod[t_u] * noise
                eps = self.ddim_sampler.pred_eps(x_t_u, cond, tu, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                loss2 = (eps - noise).pow(2).mean()
                loss2.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # if cur_iter % 50 == 0:
                #     print(f'iter {cur_iter}: {loss1.item()}, {loss2.item()}')

        cond1.requires_grad_(False)
        cond2.requires_grad_(False)
        uncond_base.requires_grad_(False)
        return cond1, cond2, uncond_base

    def interpolate_naive(self, img1, img2, num_frames, out_dir='blend'):
        if isinstance(img1, Image.Image):
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()
        ldm = self.model
        L1 = ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        L2 = ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        for frame_ix in trange(1,num_frames-1):
            frac = frame_ix/(num_frames-1)
            latent = interpolate_spherical(L1, L2, frac)
            x_samples = ldm.decode_first_stage(latent).permute(0, 2, 3, 1)
            x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:03d}.png')

    def interpolate_then_diffuse(self, img1, pose_md1, img2, pose_md2, num_frames, cond_path=None, cond_lr=1e-4, prompt=None, n_prompt=None, ddim_steps=250, guide_scale=7.5, schedule_type='concave', optimize_cond=0, out_dir='blend'): #steps_per_frame=10, 
        """
        each successive frame has more noise than the previous
        """
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{num_frames-1:03d}.png')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()
        
        self.change_mode('pose')
        ldm = self.model
        ldm.control_scales = [1] * 13
        H = W = 512
        shape = (4, H // 8, W // 8)

        if cond_path and os.path.exists(cond_path):
            optimize_cond = True
            cond1, cond2, uncond_base = torch.load(cond_path)
            cond = {}
            un_cond = {"c_crossattn": [uncond_base]}
        else:
            cond_base = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])
            cond = {"c_crossattn": [cond_base], 'c_concat': None}
            un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

            if optimize_cond:
                cond1, cond2, uncond_base = self.learn_conditioning(img1, img2, cond, cond_base, uncond_base, ddim_steps, guide_scale=guide_scale, num_iters=optimize_cond, cond_lr=cond_lr)
                if cond_path:
                    torch.save((cond1, cond2, uncond_base), cond_path)

        img1 = img1.float() / 127.5 - 1.0
        img2 = img2.float() / 127.5 - 1.0
        # schedules include endpoints
        self.ddim_sampler.make_schedule(ddim_steps, verbose=False)
        step_schedule = get_step_schedule(.7*ddim_steps, (num_frames+1)//2, schedule_type=schedule_type)
        timestep_schedule = [self.ddim_sampler.ddim_timesteps[s] for s in step_schedule]
        latents1, latents2 = self.get_latent_stack(img1, img2, timestep_schedule)
        latents = [None] * num_frames
        latents[0] = latents1[0]
        latents[-1] = latents2[0]
        
        with torch.no_grad():
            for frame_ix in trange(1,num_frames-1):
                frac = frame_ix/(num_frames-1)
                f = min(frame_ix, num_frames - frame_ix - 1)
                # ldm.control_scales = [0.5 + 3*f/num_frames] * 13 # range from 0.5 to 2
                latents[frame_ix] = interpolate_spherical(latents1[f], latents2[f], frac)
                pose_img = interp_poses(pose_md1, pose_md2, alpha=frac).transpose(2,0,1)
                control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0

                cond["c_concat"] = [control]
                un_cond["c_concat"] = [control]
                if optimize_cond:
                    cond["c_crossattn"] = [interpolate_spherical(cond1, cond2, frac)]
                samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
                    shape, cond, verbose=False,
                    x_T=latents[frame_ix], timesteps=step_schedule[f],
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)

                x_samples = ldm.decode_first_stage(samples).permute(0, 2, 3, 1)
                x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:03d}.png')

        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, step_schedule=step_schedule)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))

    def interpolate(self, img1, img2, controls=None, control_type='pose', cond_path=None, cond_lr=1e-4, prompt=None, n_prompt=None, max_steps=200, ddim_steps=250, num_frames=17, guide_scale=7.5, schedule_type='concave', optimize_cond=0, bias=0, retroactive_interp=True, out_dir='blend'):
        """
        ddim_steps: number of steps in DDIM sampling
        num_frames: includes endpoints (both original images)
        """
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{num_frames-1:03d}.png')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()

        if controls is None:
            self.init_mode()
        else:
            self.change_mode(control_type)

        ldm = self.model
        ldm.control_scales = [1] * 13
        H = W = 512

        if cond_path and os.path.exists(cond_path):
            optimize_cond = True
            cond1, cond2, uncond_base = torch.load(cond_path)
            cond = {'c_concat': None}
            un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        else:
            cond_base = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])
            cond = {"c_crossattn": [cond_base], 'c_concat': None}
            un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

            if optimize_cond:
                cond1, cond2, uncond_base = self.learn_conditioning(img1, img2, cond, cond_base, uncond_base, ddim_steps, guide_scale=guide_scale, num_iters=optimize_cond, cond_lr=cond_lr)
                if cond_path:
                    torch.save((cond1, cond2, uncond_base), cond_path)

        img1 = img1.float() / 127.5 - 1.0
        img2 = img2.float() / 127.5 - 1.0
        # schedules include endpoints
        num_levels = int(np.log2(num_frames-1)) # does not include endpoints
        assert np.log2(num_frames-1) % 1 < 1e-5
        self.ddim_sampler.make_schedule(ddim_steps, verbose=False)
        step_schedule = get_step_schedule(max_steps, num_levels, schedule_type=schedule_type)
        timesteps = self.ddim_sampler.ddim_timesteps
        timestep_schedule = [timesteps[s] for s in step_schedule]
        latents1, latents2 = self.get_latent_stack(img1, img2, timestep_schedule)
        latents = [None] * num_frames
        latents[0] = latents1[0]
        latents[-1] = latents2[0]
        
        if controls is not None:
            if control_type == 'pose':
                pose_md1, pose_md2 = controls
            else:
                raise NotImplementedError
        for level in trange(1,num_levels+1):
            cur_ix = step_schedule[-level]
            prev_ix = step_schedule[-level-1]
            latents[0] = latents1[-level]
            latents[-1] = latents2[-level]
            df = 2**(num_levels-level)

            for frame_ix in range(df, num_frames-1, df*2):
                frac = .5
                if frame_ix-df == 0:
                    frac -= bias
                if frame_ix+df == num_frames-1:
                    frac += bias
                latents[frame_ix] = interpolate_spherical(latents[frame_ix-df], latents[frame_ix+df], frac)

            if retroactive_interp:
                if level == 2:
                    latents[num_frames//2] = interpolate_spherical(latents[num_frames//4], latents[3*num_frames//4], .5)
                
                if level == 3:
                    latents[num_frames//4] = interpolate_spherical(latents[num_frames//8], latents[3*num_frames//8], .5)
                    latents[num_frames//2] = interpolate_spherical(latents[3*num_frames//8], latents[5*num_frames//8], .5)
                    latents[3*num_frames//4] = interpolate_spherical(latents[5*num_frames//8], latents[7*num_frames//8], .5)
            
            for frame_ix in range(df, num_frames-1, df): # exclude endpoints
                frac = frame_ix/(num_frames-1)

                if controls is not None:
                    pose_img = interp_poses(pose_md1, pose_md2, alpha=frac).transpose(2,0,1)
                    control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0
                    cond["c_concat"] = un_cond["c_concat"] = [control]

                if optimize_cond:
                    cond["c_crossattn"] = [interpolate_spherical(cond1, cond2, frac)]
                
                for i, t in enumerate(np.flip(timesteps[prev_ix:cur_ix])):
                    index = cur_ix - i - 1
                    ts = torch.tensor([t], device='cuda', dtype=torch.long)

                    latents[frame_ix] = self.ddim_sampler.p_sample_ddim(latents[frame_ix], cond, ts, index=index, unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)[0]

        for frame_ix in range(1,num_frames-1):
            x_samples = ldm.decode_first_stage(latents[frame_ix]).permute(0, 2, 3, 1)
            x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:03d}.png')
        
        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, step_schedule=step_schedule, bias=bias, retroactive_interp=retroactive_interp)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
    
    def get_latent_stack(self, img1, img2, timesteps):
        ldm = self.model
        latents1 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img1))]
        latents2 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img2))]
        
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
