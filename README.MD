# GrAMMI: Graph Based Adversarial Modeling with Mutual Information

### [Arxiv](https://arxiv.org/pdf/2306.11168.pdf) - IROS 2023

GrAMMI is an adversarial tracking method that uses a graph neural network along with a regularized gaussian mixture model that is regularized using mutual information. The method is designed for tracking adversarial targets in sparse partially observable environments. 

## Training Models
  
### 1. Create Conda environment

```
conda env create -f requirements.yaml
conda activate grammi
```

### 2. Download Datasets

Visit [the Google Drive](https://drive.google.com/file/d/1uLlv_08dQX7PPxVsemM--EQlxdE8qcgs/view?usp=sharing) to download the datasets. We provide datasets for the Smuggler Domain (high and low visibility) and Prisoner datasets (high, medium, and low visibility). Download the datasets to `grammi_datasets` in the same directory for the default options.

### 3. Train 

The models can be trained using [run_multiple.py](run_multiple.py). The following command trains multiple seeds for a single model type and a single time horizon prediction. 

### 4. Evaluate the model

Once the models are trained, you can evaluate the models using [evaluate/evaluate_models_mi.py](evaluate/evaluate_models_mi.py). This will record the metrics for the models based on the test set.

**[Details of configuration](/configs)**

## Citation

If you find our code or paper is useful, please consider citing:

```bibtex
@inproceedings{ye2023grammi,
  title={Learning Models of Adversarial Agent Behavior under Partial
Observability},
  author={Ye, Sean and Natarajan, Manisha and Wu, Zixuan and Paleja, Rohan and Chen, Letian and Gombolay, Matthew},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

## License

This code is distributed under an [MIT LICENSE](LICENSE).