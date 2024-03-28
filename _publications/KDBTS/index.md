---
layout: page
permalink: publications/KDBTS/
date: 2024_03_29 # determines sorting just take the date of the first publication as YYYY_MM_DD
image: assets/teaser.png
image_mouseover: assets/teaserfigure.pdf # assets/header_vid.mp4

title: "Boosting Self-Supervision for Single View Scene Completion via Knowledge Distillation"
venue: CVPR, 2024
authors:
  - name: keonheehan
    affiliations: "1,2"
    equal_contribution: True
  - name: dominikmuhle
    affiliations: "1,2"
    equal_contribution: True
  - name: felixwimbauer
    affiliations: "1,2"
  - name: danielcremers
    affiliations: "1,2"
affiliations:
  - name: tum
    length: short
  - name: mcml
    length: long

description: "We propose to boost single-view scene completion by exploiting additional information from multiple images."

links:
    - name: Project Page
      link: publications/KDBTS/
    - name: Paper
      link: https://arxiv.org/abs/2305.09527
      style: "bi bi-file-earmark-richtext"
    - name: Code
      link: https://github.com/keonhee-han/KDBTS
      style: "bi bi-github"
    - name: Video
      link: https://www.youtube.com/watch?v=_wDUresP6v8&t=23s
      style: "bi bi-youtube"

citation: '@article{muhle2023dnls_covs,
  title={Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares},
  author={Dominik Muhle and Lukas Koestler and Krishna Murthy Jatavallabhula and Daniel Cremers},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}'

acknowledgements: 'This work was supported by the ERC Advanced Grant SIMULACRON, by the Munich Center for Machine Learning and by the EPSRC Programme Grant VisualAI EP/T028572/1.'
# citation: "@{ASDF}"
---

<video width="100%" autoplay muted loop>
  <source src="./assets/header_vid.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

***Knowledge Distillation from Multi-View to Single-View.** We propose to boost single-view scene completion by exploiting additional information from multiple images. a) we first train a novel multi-view scene reconstruction algorithm that is able to fuse density fields from multiple images in a fully self-supervised manner. b) we then employ knowledge distillation to directly supervise a state-of-the-art single-view reconstruction model in 3D to boost its performance*

# Abstract

Inferring scene geometry from images via Structure from Motion is a long-standing and fundamental problem in computer vision. While classical approaches and, more recently, depth map predictions only focus on the visible parts of a scene, the task of scene completion aims to reason about geometry even in occluded regions. With the popularity of neural radiance fields (NeRFs), implicit representations also became popular for scene completion by predicting so-called density fields. Unlike explicit approaches e.g. voxel-based methods, density fields also allow for accurate depth prediction and novel-view synthesis via image based rendering. In this work, we propose to fuse the scene reconstruction from multiple images and distill this knowledge into a more accurate single-view scene reconstruction. To this end, we propose Multi-View Behind the Scenes(MVBTS) to fuse density fields from multiple posed images, trained fully self-supervised only from image data. Using knowledge distillation, we use MVBTS to train a single view scene completion network via direct supervision called KDBTS. It achieves state-of-the-art performance on occupancy prediction, especially in occluded regions.
