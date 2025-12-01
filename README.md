# SAR-ULRTC-CR

## Unrolled Low-Rank Tensor Completion for SAR-Guided Cloud Removal in Hyperspectral Images

Deep unrolled network solving low-rank tensor completion (LRTC) with SAR-guided detail injection for hyperspectral cloud removal.

### Overview

This repository contains the implementation of **SAR-ULRTC-CR** (SAR-guided Unrolled Low-Rank Tensor Completion for Cloud Removal), a deep learning approach that combines traditional optimization-based methods with neural networks to address cloud removal in hyperspectral imagery using Synthetic Aperture Radar (SAR) guidance.

### Key Features

- **Unrolled Low-Rank Tensor Completion**: Leverages the low-rank structure of hyperspectral images through an algorithm unrolling framework, converting iterative optimization steps into learnable network layers.
- **SAR-Guided Detail Injection**: Utilizes SAR imagery, which is unaffected by cloud cover, to guide the reconstruction process and preserve spatial details in cloud-affected regions.
- **Hyperspectral Cloud Removal**: Specifically designed for removing clouds and restoring missing information in hyperspectral remote sensing images.

### Method

The proposed method addresses the cloud removal problem by:
1. Formulating cloud removal as a low-rank tensor completion problem
2. Unrolling the optimization algorithm into a deep network architecture
3. Incorporating SAR data as auxiliary information to guide the completion process
4. Learning the network parameters end-to-end for improved reconstruction quality

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
