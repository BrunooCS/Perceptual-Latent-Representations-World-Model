# Perceptual Latent Representations for World Models

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.01758-b31b1b.svg)](https://arxiv.org/abs/2510.01758)

Research code for **Perceptual Latent Representations: Improving Visual Fidelity and Autonomous Control in World Models**, a Final Degree Work in Data Science and Engineering at Universidade da Coruña.

The project studies a central bottleneck in generative World Models: the visual latent representation used by the memory model and the controller. Standard VAE-based vision modules often produce blurry reconstructions and latent states that are difficult to predict over time. This repository implements a compact alternative, **DDS+VAE**, that combines unsupervised dynamic feature selection, hierarchical spatial compression, and an internal perceptual loss to produce sharper and more useful latent representations without scaling model size.

## Highlights

| Result in CarRacing-v3 | Baseline VAE | DDS+VAE |
|---|---:|---:|
| Reconstruction MSE ↓ | 0.00165 | **0.00039** |
| FID ↓ | 59.46 | **25.35** |
| FVD ↓ | 239 | **176** |
| Agent reward ↑ | 734.96 ± 162.75 | **818.58 ± 147.05** |

Main takeaways:

- **Sharper latent vision**: 76.4% lower reconstruction MSE in CarRacing-v3.
- **Better imagined futures**: 57.4% lower FID and 26.4% lower FVD for generated dream sequences.
- **Better control**: 11.4% higher average reward with the same linear controller setup.
- **Compact model**: 4.04M parameters, below the 4.35M VAE baseline and far below the 6.69M MAE+VAE baseline used for comparison.

## Research Outputs

- Paper: [Unsupervised Dynamic Feature Selection for Robust Latent Spaces in Vision Tasks](https://arxiv.org/abs/2510.01758)
- Official TFG record at Universidade da Coruña: [RUC publication page](https://ruc.udc.es/entities/publication/adabf2d2-ee3d-4c3c-b734-486f68a0669d)
- TFG defense: Facultade de Informática, Universidade da Coruña, 2 July 2025.

If you use this work, please cite:

```bibtex
@misc{corcuera2025unsupervised,
  title={Unsupervised Dynamic Feature Selection for Robust Latent Spaces in Vision Tasks},
  author={Bruno Corcuera and Carlos Eiras-Franco and Brais Cancela},
  year={2025},
  eprint={2510.01758},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Method Overview

World Models split autonomous behavior into three modules:

1. **Vision**: compress image observations into a latent vector `z_t`.
2. **Memory**: predict future latent states with an MDN-RNN.
3. **Controller**: map the latent state and recurrent state to an action.

This project focuses on the vision module, because its latent space determines how much useful information reaches both memory and control.

### DDS+VAE Vision Module

DDS+VAE replaces a monolithic VAE with a hierarchical representation pipeline:

1. **UNet1 + DDS** selects instance-specific visual features without labels.
2. **Downscaling/Upscaling** converts sparse selected pixels into a dense compact spatial representation.
3. **Mini-VAE** regularizes the representation into a normalized latent space suitable for MDN-RNN prediction.
4. **Internal perceptual loss** compares activations inside the frozen Upscaling module, avoiding an external pretrained perceptual network.

Conceptually:

```python
mask = DDS(unet1(x))
x_sparse = mask * x
h = downscale(x_sparse)
z = minivae.encode(h)
h_hat = minivae.decode(z)
x_hat = unet2(upscale(h_hat))
```

The DDS module typically preserves only a small fraction of pixels, but learns to keep high-value structures such as road borders, the vehicle, texture transitions, and spatial cues needed by the controller.

## Experimental Setup

### Environments

- `CarRacing-v3`: main benchmark, with procedural track generation and continuous control.
- `SuperMarioBros-v0`: secondary visual-generation benchmark with discrete actions and deterministic levels.

### Compared Vision Models

| Model | Parameters | Training |
|---|---:|---|
| VAE baseline | 4.35M | 1 epoch |
| MAE+VAE baseline | 6.69M | 1 + 2 epochs |
| DDS+VAE | 4.04M | 1 + 2 epochs |

### Metrics

- **MSE** for image reconstruction.
- **FID** for frame-level realism of generated dream sequences.
- **FVD** for temporal coherence and video-level similarity.
- **Cumulative reward** for downstream control in CarRacing-v3.

## Results

| Environment | Model | MSE ↓ | FID ↓ | FVD ↓ | 
|---|---|---:|---:|---:|---:|
| CarRacing-v3 | VAE | 0.00165 | 59.46 | 239 | 
| CarRacing-v3 | MAE+VAE | 0.00220 | 54.84 | 312 | 
| CarRacing-v3 | DDS+VAE | **0.00039** | **25.35** | **176** |
| SuperMarioBros-v0 | VAE | 0.00134 | 64.06 | 412 | 
| SuperMarioBros-v0 | MAE+VAE | 0.00114 | **60.21** | 465 |
| SuperMarioBros-v0 | DDS+VAE | **0.00105** | 61.08 | **338** | 

The strongest evidence comes from CarRacing-v3, where procedural generation reduces memorization and directly tests whether the latent space generalizes to unseen tracks. DDS+VAE improves both visual quality and downstream control, which indicates that the representation is not only sharper but also more actionable.

## Repository Guide

Expected project layout:

```text
.
├── src/
│   ├── rollouts.py
│   ├── train_dds_vae_stage1.py
│   ├── train_dds_vae_stage2.py
│   ├── train_memory.py
│   ├── train_controller.py
│   ├── models/
│   │   ├── dds_vae.py
│   │   ├── mdn_rnn.py
│   │   ├── controller.py
│   │   ├── dds/
│   │   ├── minivae/
│   │   ├── unet/
│   │   └── UpscaleDownscale/
│   └── utils/
├── notebooks (self-explanatory)/
│   ├── 1-Rollouts.ipynb
│   ├── 2-DDS+VAE.ipynb
│   ├── 3-Memory.ipynb
│   └── 4-Controller.ipynb
├── resources/
└── README.md
```

The notebooks provide the most direct route for inspecting data generation, architecture behavior, memory prediction, and controller training.

## Running the Notebooks

```bash
jupyter lab "notebooks (self-explanatory)/"
```

Recommended reading order:

1. `1-Rollouts.ipynb`: data collection and environment interaction.
2. `2-DDS+VAE.ipynb`: vision architecture and two-stage training.
3. `3-Memory.ipynb`: latent dynamics with MDN-RNN.
4. `4-Controller.ipynb`: CMA-ES controller optimization and evaluation.

## Scope and Limitations

This work is a research prototype, not a production reinforcement-learning framework. The main contribution is the visual latent representation and its effect on generated futures and control. The current validation is strongest in 2D visual environments; extension to 3D domains, robotics, longer-horizon prediction, and modern memory architectures remains future work.

## Author

**Bruno Corcuera Sánchez**  
Universidade da Coruña, Facultade de Informática  
Academic email: [bruno.sanchez1@udc.es](mailto:bruno.sanchez1@udc.es)  
Personal email: [brunocorcueras@gmail.com](mailto:brunocorcueras@gmail.com)

## Acknowledgments

Advisors: **Brais Cancela Barizo** and **Carlos Eiras Franco**.  
This project builds on the World Models paradigm introduced by David Ha and Jürgen Schmidhuber and on the broader PyTorch, Gymnasium, and reinforcement-learning research ecosystem.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
