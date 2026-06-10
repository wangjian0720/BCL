## Environment
We recommend the following environment configuration:
```bash
Python == 3.12
PyTorch == 2.4.0
CUDA == 12.4
```
---


## Dataset Preparation
Please download the following datasets before training:

| Dataset | Link |
|---|---|
| ACDC | [ACDC Dataset](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) |
| MSC-CMRSeg | [MSC-CMRSeg Dataset](https://zmiclab.github.io/zxh/0/mscmrseg19/) |
| PROMISE12 | [PROMISE12 Dataset](https://promise12.grand-challenge.org/Download/) |
| Kvasir | [Kvasir-Seg Dataset](https://datasets.simula.no/kvasir-seg/) |

After downloading the datasets, please follow the preprocessing instructions in: ./data

---


## Inference

Additional hyperparameters can be adjusted in the corresponding `.sh` scripts or configuration files.

To evaluate a trained model:

```bash
sh test_acdc.sh
```
Before testing, please ensure that:
- the checkpoint path is correctly specified,
- the dataset path is properly configured.
---