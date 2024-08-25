<p align="center">
  <img src="./assets/sapiens_animation.gif" alt="Sapiens" title="Sapiens" width="500"/>
</p>

<p align="center">
   <h2 align="center">Foundation for Human Vision Models</h2>
   <p align="center">
      <a href="https://rawalkhirodkar.github.io/"><strong>Rawal Khirodkar</strong></a>
      ·
      <a href="https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&hl=en"><strong>Timur Bagautdinov</strong></a>
      ·
      <a href="https://una-dinosauria.github.io/"><strong>Julieta Martinez</strong></a>
      ·
      <a href="https://about.meta.com/realitylabs/"><strong>Su Zhaoen</strong></a>
      ·
      <a href="https://about.meta.com/realitylabs/"><strong>Austin James</strong></a>
      <br>
      <a href="https://www.linkedin.com/in/peter-selednik-05036499/"><strong>Peter Selednik</strong></a>
      .
      <a href="https://scholar.google.fr/citations?user=8orqBsYAAAAJ&hl=ja"><strong>Stuart Anderson</strong></a>
      .
      <a href="https://shunsukesaito.github.io/"><strong>Shunsuke Saito</strong></a>
   </p>
   <h3 align="center">ECCV 2024 (Oral)</h3>
</p>

<p align="center">
   <a href="https://arxiv.org/abs/2408.12569">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
   </a>
   <a href='https://about.meta.com/realitylabs/codecavatars/sapiens/'>
      <img src='https://img.shields.io/badge/Sapiens-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'>
   </a>

</p>

Sapiens offers a comprehensive suite for human-centric vision tasks (e.g., 2D pose, part segmentation, depth, normal, etc.). The model family is pretrained on 300 million in-the-wild human images and shows excellent generalization to unconstrained conditions. These models are also designed for extracting high-resolution features, having been natively trained at a 1024 x 1024 image resolution with a 16-pixel patch size.

## 🚀 Getting Started

### Clone the Repository
   ```bash
   git clone https://github.com/facebookresearch/sapiens.git
   export SAPIENS_ROOT=/path/to/sapiens
   ```

### Recommended: Lite Installation (Inference-only)
   For users setting up their own environment primarily for running existing models in inference mode, we recommend the [Sapiens-Lite installation](lite/README.md).\
   This setup offers optimized inference (4x faster) with minimal dependencies (only PyTorch + numpy + cv2).

### Full Installation
   To replicate our complete training setup, run the provided installation script. \
   This will create a new conda environment named `sapiens` and install all necessary dependencies.

   ```bash
   cd $SAPIENS_ROOT/_install
   ./conda.sh
   ```

   Please download the checkpoints from [hugging-face](https://huggingface.co/facebook/sapiens/tree/main/sapiens_host). \
   You can be selective about only downloading the checkpoints of interest.\
   Set `$SAPIENS_CHECKPOINT_ROOT` to be the path to the `sapiens_host` folder. Checkpoint directory structure:
   ```plaintext
   sapiens_host/
   ├── detector/
   │   └── checkpoints/
   │       └── rtmpose/
   ├── pretrain/
   │   └── checkpoints/
   │       ├── sapiens_0.3b/
   │       ├── sapiens_0.6b/
   │       ├── sapiens_1b/
   │       └── sapiens_2b/
   ├── pose/
   └── seg/
   └── depth/
   └── normal/
   ```

## 🌟 Human-Centric Vision Tasks
We finetune sapiens for multiple human-centric vision tasks. Please checkout the list below.

- ###  [Image Encoder](docs/PRETRAIN_README.md) <sup><small><a href="lite/docs/PRETRAIN_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### [Pose Estimation](docs/POSE_README.md) <sup><small><a href="lite/docs/POSE_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### [Body Part Segmentation](docs/SEG_README.md) <sup><small><a href="lite/docs/SEG_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### [Depth Estimation](docs/DEPTH_README.md) <sup><small><a href="lite/docs/DEPTH_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### [Surface Normal Estimation](docs/NORMAL_README.md) <sup><small><a href="lite/docs/NORMAL_README.md" style="color: #FFA500;">[lite]</a></small></sup>

## 🎯 Easy Steps to Finetuning Sapiens
Finetuning our models is super-easy! Here is a detailed training guide for the following tasks.
- ### [Coming Soon] Pose/Seg/Depth
- ### [Surface Normal Estimation](docs/finetune/NORMAL_README.md)

## 🤝 Acknowledgements & Support & Contributing
We would like to acknowledge the work by [OpenMMLab](https://github.com/open-mmlab) which this project benefits from.\
For any questions or issues, please open an issue in the repository.\
See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## License
This project is licensed under [LICENSE](LICENSE).\
Portions of the project derived from open-source projects are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## 📚 Citation
If you use Sapiens in your research, please use the following BibTeX entry.
```bibtex
@misc{khirodkar2024_sapiens,
    title={Sapiens: Foundation for Human Vision Models},
    author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
    year={2024},
    eprint={2408.12569},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2408.12569}
}
```
