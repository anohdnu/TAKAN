##  TAKAN
PyTorch code for the  paper:\
**Out-of-Distribution Detection in Continual Learning**\
<p align="center">
<img src="main_fig.png" width="90%">
</p>

## Setup and Datasets Preparation
Please following the setup steps in [CODA-Prompt]
### Setup
 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environment w/ python 3.8, ex: `conda create --name coda python=3.8`
 * `conda activate coda`
 * `sh install_requirements.sh`
 * <b>NOTE: this framework was tested using `torch == 2.0.0` but should work for previous versions</b>
 
### Datasets
 * Create a folder `data/`
 * **CUB200**: retrive from: https://www.kaggle.com/datasets/cyizhuo/cub-200-2011-by-classes-folder
 * **ImageNet-R**: retrieve from: https://github.com/hendrycks/imagenet-r
 * **DomainNet**: retrieve from: http://ai.bu.edu/M3SDA/

## Training
All commands should be run under the project root directory.

```bash
sh experiments/cub.sh
sh experiments/imagenet-r.sh
sh experiments/domainnet.sh
```

The methods with a linear classifier and with TA-KAN classifier will be evaluated.

## Results
Results will be saved in a folder named `outputs/`. To get the final average accuracy, AUROC and FPR95, retrieve the final number in the file `outputs/**/results-<metric>/global.yaml`

## Citation
**If you found our work useful for your research, please cite our work**:
    

## Thanks
The code is developed based on [KAC] and [CODA-Prompt], and the implementation of KAC follows [Fast-KAN].

[KAC]: https://github.com/Ethanhuhuhu/KAC

[CODA-Prompt]: https://github.com/GT-RIPL/CODA-Prompt

[Fast-KAN]: https://github.com/ZiyaoLi/fast-kan
