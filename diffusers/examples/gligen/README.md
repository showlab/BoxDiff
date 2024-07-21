# GLIGEN: Open-Set Grounded Text-to-Image Generation (CVPR 2023)

[Yuheng Li](https://yuheng-li.github.io/), [Haotian Liu](https://hliu.cc), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Fangzhou Mu](https://pages.cs.wisc.edu/~fmu/), [Jianwei Yang](https://jwyang.github.io/), [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/), [Chunyuan Li*](https://chunyuan.li/), [Yong Jae Lee*](https://pages.cs.wisc.edu/~yongjaelee/) (*Co-senior authors)

[[Project Page](https://gligen.github.io/)] [[Paper](https://arxiv.org/abs/2301.07093)] [[Demo](https://huggingface.co/spaces/gligen/demo)] [[YouTube Video](https://youtu.be/-MCkU7IAGKs)]

## :fire: News

* **[2023.03.22]** Our fork on diffusers with support of text-box-conditioned generation and inpainting is released.  We are working on integrating it into official diffusers code base.  More conditions and a new demo is on the way.  Stay tuned!

## Installation
```Shell
conda create -n gligen_diffusers python=3.10
conda activate gligen_diffusers
pip install torch torchvision transformers accelerate

git clone git@github.com:gligen/diffusers.git
pip install -e .
```

## Usage
```Shell
cd examples/gligen

# for text-box conditioned generation.
python generation_text_box.py

# for text-box conditioned inpainting.
python inpainting_text_box.py
```
