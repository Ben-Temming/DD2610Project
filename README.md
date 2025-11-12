<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- <br /> -->
<!-- PROJECT LOGO -->
  <!-- <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->


<h1 align="center">Updated Instructions</h1>

### Installation

1.  Create the conda environment:
    ```bash
    conda env create -f conda_environment.yaml
    ```
2.  Activate the environment:
    ```bash
    conda activate ssr
    ```
3.  Test CUDA installation:
    ```bash
    python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
    ```

### Testing

Run the training script with the following command:

```bash
python train.py --gpu 0 --c config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml
```

**NOTE:** Before running this locally on Windows, you may need to make the following changes in the `config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml` file:
*   Set `num_workers` to `0`. This is to avoid a multi-threading issue on Windows that can prevent data from being loaded.
*   Set `num_log_iter` to `1` (or a lower number). The default is `256`, which can be very slow on a laptop GPU. This will allow you to see the TensorBoard output more quickly.

### Tensorboard

To view the Tensorboard graph, run:
```bash
tensorboard --logdir=./saved_models/classic_cv/rankup_utkface_lb250_s0/tensorboard/
```

### Wandb

For logging, you can use `wandb` as an alternative to TensorBoard.

#### Offline Mode

To log experiments locally, you can use `wandb` in offline mode. This is useful for tracking experiments without an internet connection.

1.  In your `.yaml` configuration file, set `use_wandb` to `True`.
2.  Set `wandb_mode` to `offline`.

Example configuration:
```yaml
use_wandb: True
wandb_mode: offline
```

Your logs will be saved locally in the `wandb` directory under the `saved_models` directory.


#### Online Mode

To sync your logs with the Weights & Bienses cloud service (https://wandb.ai/site/), use the online mode. This allows you to view and manage your experiments from anywhere through their web interface.

1.  **Create a `.env` file** in the root of the project.

    For Linux or macOS, use this command:
    ```bash
    cp .env.template .env
    ```

    For Windows, use this command:
    ```bash
    copy .env.template .env
    ```
2.  **Add your WANDB_API_KEY** to the `.env` file:
    ```
    WANDB_API_KEY=your_api_key_here
    ```
3.  **Enable `wandb` in your configuration file**:
    In your `.yaml` configuration file (e.g., `rankup_utkface_lb250_s0.yaml`), set `use_wandb` to `True` and ensure `wandb_mode` is set to `online` 

    Example configuration:
    ```yaml
    use_wandb: True
    wandb_mode: online
    ```
    > **Note:** For online mode, you can also specify the project and run name in the `.yaml` file. This helps organize your experiments in the `wandb` dashboard.
    ```yaml
    wandb_online_logging:
      project: DD2610Project
      name: classic_cv_rankup_utkface_lb250_s0
    ```
    > The `project` should be the same for all runs within the same project, and the `name` should be unique for each experiment.

Now, when you run your training, the logs will be synced to your `wandb` account.




<h1 align="center">🎋 Semi-Supervised Regression</h1>

<p align="center">
  Official repository for the paper <br />
  <strong>"RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier"</strong>,<br />
  by Pin-Yen Huang, Szu-Wei Fu, and Yu Tsao. Accepted at NeurIPS 2024.
</p>

<p align="center">
  Explore the following resources:
  <br />
  <a href="https://arxiv.org/abs/2410.22124">📄 arXiv Paper</a>
  ·
  <a href="./results/README.md">🗃️ Experiment Logs</a>
  ·
  <a href="./visualization/README.md">✨ Feature Visualizations</a>
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>📋 Table of Contents</strong></summary>
  <ol>
    <li><a href="#1-introduction">Introduction</a></li>
    <li><a href="#2-getting-started">Getting Started</a></li>
    <li><a href="#3-usage">Usage</a></li>
    <li><a href="#4-benchmark-results">Benchmark Results</a></li>
    <li><a href="#5-license">License</a></li>
    <li><a href="#6-contact">Contact</a></li>
    <li><a href="#7-citing-this-work">Citing This Work</a></li>
    <li><a href="#8-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## 1. Introduction

<div align="center">
  <img src="https://pm25.github.io/research-figures/rankup/figures/rankup-illustration.png" alt="RankUp Illustration" height="250">
  <br/>
  Paper Link: <a href="https://arxiv.org/abs/2410.22124">https://arxiv.org/abs/2410.22124</a>
</div>
<br/>

This repository contains the official code and documentation for the paper _"RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier"_. Here, you’ll find implementations of the proposed methods (RankUp and RDA), alongside related works, experiment logs, feature visualizations, datasets, and hyperparameter settings used in our experiments. The code is adapted from the popular semi-supervised classification framework, [USB](https://github.com/microsoft/semi-supervised-learning). We are deeply grateful to the USB team for their excellent framework, which played a crucial role in the development of our research and made our work possible!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- NEWS & UPDATES -->

## 2. News and Updates

-   [Jan 01, 2025] Simplified the code by removing classification algorithms.
-   [Oct 23, 2024] Released the code for the RankUp paper.
-   [Sep 25, 2024] The _RankUp_ paper has been accepted into the NeurIPS 2024 Main Track!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## 3. Getting Started

This section guides you through setting up the repository locally. Follow these steps to get your environment ready.

### 💽 Clone Repository

Clone this repository with the following command:

```bash
git clone https://github.com/pm25/semi-supervised-regression.git
```

### 🪛 Prerequisites

This repository is built on PyTorch, along with `torchvision`, `torchaudio`, and `transformers`.

To install the required packages, first create a Conda environment:

```bash
conda create --name ssr python=3.8
```

Activate the environment:

```bash
conda activate ssr
```

Then, use `pip` to install the required packages:

```bash
python -m pip install -r requirements.txt
```

Once the installation is complete, you can start using this repository by running:

```bash
python train.py --gpu 0 --c config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml
```

### ⚙️ Configuration Files

All configuration files are stored in the [config](./config) folder.

The configuration file paths follow this naming format:

```python
./config/{data_type}/{method}/{method}_{dataset}_lb{num_labeled}_s{seed}.yaml
```

For example, the configuration file for the RankUp method on the UTKFace dataset with 250 labeled samples and seed 0 is located at:

```bash
./config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml
```

### 📊 Prepare Datasets

No manual dataset preparation is required. The datasets will be automatically downloaded and processed when you run the code. For more details on the dataset splits and processing, visit the [pm25/Regression-Datasets](https://github.com/pm25/regression-datasets).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## 4. Usage

This repository is designed to be user-friendly and extendable. Below are some examples to help you get started with training, evaluation, and feature visualization.

### 🏋️ Training

To train RankUp on the UTKFace dataset with 250 labels on GPU 0, use the following command. You can modify the configuration file to try other algorithms, datasets, or label settings:

```bash
python train.py --gpu 0 --c config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml
```

### 🎯 Evaluation

After training, you can either view the evaluation results in the training logs or run the evaluation script:

```bash
python eval.py --gpu 0 --c config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml
```

### ✨ Features Visualization

Once training is complete, you can extract hidden features of the evaluation data using:

```bash
python eval.py --gpu 0 --save_features \
    --c config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml
```

To visualize these features, you can project them into 2D or 3D space using t-SNE or UMAP:

```bash
python visualization/plot.py --methods tsne umap --output_dim 2 \
    --load_path visualization/features/rankup_utkface_lb250_s0.npy
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- BENCHMARK RESULTS -->

## 5. Benchmark Results

Please refer to [Results](./results) for benchmark results and experiment logs on different tasks and labeled settings.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## 6. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## 7. Contact

-   Pin-Yen Huang (pyhuang97@gmail.com), Arizona State University
-   Szu-Wei Fu (szuweif@nvidia.com), NVIDIA
-   Yu Tsao (yu.tsao@citi.sinica.edu.tw), Academia Sinica

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Citing -->

## 8. Citing this Work

If you find this repository useful, please consider citing our paper or giving this repository a star. Your support is greatly appreciated!

```
@inproceedings{
  huang2024rankup,
  title={RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier},
  author={Pin-Yen Huang and Szu-Wei Fu and Yu Tsao},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024},
  url={https://openreview.net/forum?id=d2lPM1Aczs}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## 9. Acknowledgments

-   [USB](https://github.com/microsoft/semi-supervised-learning)
-   [TorchSSL](https://github.com/TorchSSL/TorchSSL)
-   [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[linkedin-url]: https://linkedin.com/in/py-huang
