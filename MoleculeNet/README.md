# Disclaimer and Instructions

This codebase is built upon a fork from [PAR](https://github.com/tata1661/PAR-NeurIPS21) repositories. The README file, license, etc are copied and modified from there. The original README file from the PAR repository can be found at the bottom. Please follow their instructions to download the four MoleculeNet datasets (tox21, sider, muv, toxcast) and install required dependencies. 

# Instruction for meta-training and meta-testing ADKF-IFT on MoleculeNet

The `./example_script_train_adkf-ift.sh` script contains example commands for running ADKF-IFT and Pre-ADKF-IFT on the four MoleculeNet datasets, which can be executed by running:

```bash
bash example_script_train_adkf-ift.sh 
```

---
---

Below is the original README file from the PAR repository.

This is the PyTorch implementation of ["Property-Aware Relation Networks (PAR) for Few-Shot Molecular Property Prediction (spotlight)"](https://papers.nips.cc/paper/2021/hash/91bc333f6967019ac47b49ca0f2fa757-Abstract.html) published in NeurIPS 2021 as a spotlight paper. 
The PaddlePaddle implementation is a part of PaddleHelix, which can be reached [here](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/fewshot_molecular_property).

<p align="center"><img src="PAR-thumbnail.png" alt="logo" width="400px" />

Please cite our paper if you find it helpful. Thanks. 
```
@InProceedings{wang2021property,
  title={Property-Aware Relation Networks for Few-Shot Molecular Property Prediction},
  author={Wang, Yaqing and Abuduweili, Abulikemu and Yao, Quanming and Dou, Dejing},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2021},
}
```

## Environment  
We used the following Python packages for core development. We tested on `Python 3.7`.
```
- pytorch 1.7.0
- torch-geometric 1.7.0
```

## Datasets 
Tox21, SIDER, MUV and ToxCast are previously downloaded from [SNAP](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). You can download the data [here](https://drive.google.com/file/d/1K3c4iCFHEKUuDVSGBtBYr8EOegvIJulO/view?usp=sharing), unzip the file and put the resultant ``muv, sider, tox21, and toxcast" in the data folder. 

## Experiments
To run the experiments, use the command (please check and tune the hyper-parameters in [parser.py](parser.py):
```
python main.py
```

If you want to quickly run PAR method on tox21 dataset, please use the command:
```
bash script_train.sh
```