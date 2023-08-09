# Interpolating between Images with Diffusion Models

[Project](https://clintonjwang.github.io/interpolation) | [Paper](https://arxiv.org/abs/2307.12560)

![Interpolate between diverse input images](https://github.com/clintonjwang/ControlNet/blob/main/github_teaser.png?raw=true)

Fork of [ControlNet](https://github.com/lllyasviel/ControlNet) adapted for generating videos that interpolate between two arbitrary given images, as described in ["Interpolating between Images with Diffusion Models"](https://clintonjwang.github.io/interpolation). See installation instructions in the original ControlNet repository.

Images and prompts used in our paper are contained in `sample_imgs` and `sample_scripts`. Image copyrights belong to their original owners. You may get slightly different results as the random seed was not fixed.

If you use this work in your research, please use the following citation:
```
@misc{wang2023interpolating,
      title={Interpolating between Images with Diffusion Models}, 
      author={Clinton J. Wang and Polina Golland},
      year={2023},
      eprint={2307.12560},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
