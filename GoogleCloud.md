# Read me for Google Cloud

## 1) Setup  a Google Cloud account
- create free account https://cloud.google.com/
    <br>(requires method payment) -> 300$ of credits for 90 days
- upgrade to full account https://console.cloud.google.com/welcome/ <br>(as long as you have credits you will not be charged, I think 😊)
- request 1 GPU 
    - go to https://console.cloud.google.com/iam-admin/quotas
    - filter `GPUs (all regions)` 
    - click three dots and `Edit Quotas` 
    - fill the form, request limit increase to 1 for `GPUs (all regions)`
        <br>just 1 at first for fast approval, can request more later
    - wait for approval email (should be fast a few min/hours)
- monitor (see below)

- other 200$ credits can be obtained later with the KTH coupon (check course canvas) if needed


## Monitoring
- https://console.cloud.google.com/welcome/
- monitor quota https://console.cloud.google.com/iam-admin/quotas
- monitor billing/credits https://console.cloud.google.com/billing/011BF1-EDE8E9-4926F8/credits?hl=it&organizationId=0&supportedpurview=project 
    - create budget alert (too be safe)


## 2) Create a PyTorch Deep Learning VM instance
https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning 
<br>(or from https://docs.cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)

- `Launch`
- fill deployment form
    - choose `deployment name` ("-vm" will be appended automatically)
    - `Deployment Service Account`
        - `New Account` -> choose `Service account name`
        - if the first deploy fails could be because the `Deployment Service Account` was still being created, try again, this time select `Existing Account`
    - `Zone` -> choose a zone where to allocate the VM (see where GPUs could be available https://docs.cloud.google.com/compute/docs/regions-zones/accelerator-zones) (I was able to allocate a L4 GPU in us-west1-a)
    - `Machine type` -> GPUS
        - `GPU type` 
        - `Machine type` 
        - `GPUs` -> Add GPU
            - `Number of GPUs` -> 1
            - `GPU type` -> NVIDIA L4 (also T4 should work, but L4 is newer and faster, P4 is older and slower, other gpus require special quota permissions)

- you will see the VM instance in the solution deployment page https://console.cloud.google.com/products/solutions/deployments


## 3) VM setup

Using the PyTorch Deep Learning VM the cuda drivers are already installed. If using a GPU NVIDIA L4 (probably also with T4) it's enough to install requirements.txt as is (see later).

- from `Solution>Solution deployment` select the VM instance created
- SSH connection (can be done in the browser)
- generate ssh for connection to your github account
    ```bash 
    ssh-keygen -t ed25519 -C "your_email@example.com"
    cat ~/.ssh/id_ed25519.pub
    # copy the output, paste it in your github account (Settings>SSH and GPG keys>New SSH key)

    ```
- clone repo
    ```bash 
    git clone repository_ssh_link
    ```
- install miniconda (https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation)
    ```bash 
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ~/Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc # or restart terminal
    ```

- create conda environment
    ```bash 
    conda create -name ssr python=3.10 
    conda activate ssr
    ```

- install requirements
    ```bash 
    python -m pip install -r requirements.txt
    ```

- create `.env` file with wandb api key
    ```bash 
    cp .env.template .env
    nano .env
    ```

    ```env
    WANDB_API_KEY=your_wandb_api_key
    ```


## 4) Training

- see available config files 

    ```bash 
    cd DD2610Project
    tree config/*/rankup*
    ```

- choose which training to run and edit the config file accordingly
    ```bash 
    nano config/domain/architecture/your_chosen_config.yaml
    ```

    ```yaml
    # change
    use_tensorboard: False 
    use_wandb: True
    wandb_mode: online #offline 

    # add
    wandb_online_logging: 
        project: DD2610Project
        name: your_experiment_name # unique name for each run

    # use original num_log_iter and num_workers values
    ```

- check that the GPU is available (nvidia-smi) (in case of issue sudo reboot should fix it)
    ```bash 
    nvidia-smi
    # if GPU driver mismatch
    sudo reboot
    ```

- create tmux session
    ```bash 
    tmux new -s session_name
    ```
- activate conda env
    ```bash 
    conda activate ssr
    ```

- run training script 
    ```bash 
    python train.py --gpu 0 --c config/domain/architecture/your_chosen_config.yaml
    ```

- see that training has started properly (wandb online dashboard, training logs)

- detach tmux session
    ```
    ctrl+B (also : if from web tab) then D
    ```

- ssh connection to the VM can be closed 
    ```bash 
    exit
    # or simply close the terminal
    ```

- monitor training with wandb https://wandb.ai/site/
- when training is finished
- connect again via ssh to the VM
- reattach tmux session (tmux attach)
    ```bash
    tmux attach -t session_name
    ```

- terminate tmux session
    ```bash
    exit
    ```

- eventually create multiple tmux sessions for different trainings in parallel (each training will probably be slower)