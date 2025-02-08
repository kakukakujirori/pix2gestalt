# Pix2Gestalt: Amodal Segmentation by Synthesizing Wholes
### CVPR 2024 (Highlight)
### [Project Page](https://gestalt.cs.columbia.edu/)  | [Paper](https://arxiv.org/pdf/2401.14398.pdf) | [arXiv](https://arxiv.org/abs/2401.14398) | [Weights](https://huggingface.co/cvlab/pix2gestalt-weights) | [Citation](https://github.com/cvlab-columbia/pix2gestalt#citation)

[pix2gestalt: Amodal Segmentation by Synthesizing Wholes](https://gestalt.cs.columbia.edu/)
 [Ege Ozguroglu](https://egeozguroglu.github.io/)<sup>1</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/)<sup>1</sup>, [Dídac Surís](https://www.didacsuris.com/)<sup>1</sup>, [Dian Chen](https://scholar.google.com/citations?user=zdAyna8AAAAJ&hl=en)<sup>2</sup>, [Achal Dave](https://www.achaldave.com/)<sup>2</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup> <br>
 <sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute

 This is a reimplementation of Pix2Gestalt using the Diffusers library.

 ##  Installation
```bash
conda create -n pix2gestalt python=3.12
conda activate pix2gestalt
pip install -r requirements.txt
```

### Dataset
```bash
wget https://gestalt.cs.columbia.edu/assets/pix2gestalt_occlusions_release.tar.gz
tar -xvf pix2gestalt_occlusions_release.tar.gz
```
Please refer to the original repo for the data license and disclaimer.

### Training

Change the ```--train_data_dir``` in ```train.sh``` to point to your dataset path, and run:
```
bash train.sh
```
