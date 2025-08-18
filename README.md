## 1. Requirements

We recommend using **conda** to create the environment.

```bash
conda env create -f environment.yml
conda activate gnsp
```

The provided `environment.yml` is a **slim version** containing only the core dependencies required to reproduce the experiments.
If you encounter dependency issues, please refer to the full version `environment_full.yml` for an exact environment match.

---

## 2. Directory Structure

After downloading the code and dataset, the folder structure should look like:

```
GNSP
├── config
│   ├── mtil_order_I.json
│   └── mtil_order_II.json
├── data
│   ├── ImageNet1K             
│   │   ├── ILSVR2012_img_train
│   │   │   ├── n01440764
│   │   │   └── ...
│   │   └── labels.txt
│   └── mtil                        
│       ├── Aircraft
│       ├── Caltech101
│       └── ...
├── project_config.json
└── run_mtil.py
```

---

## 3. Usage

### Step 1: Download source code

Clone or download the repository into a folder named `GNSP`.

### Step 2: Prepare dataset

All datasets of **MTIL** including Aircraft, Caltech101, CIFAR10, CIFAR100, DTD, 
EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397. 
You can use `gather_datasets.py` to download directly through pytorch api.
```
python gather_datasets.py
```
Note that due to some links becoming invalid, gather_datasets.cpy may not be able to retrieve the entire dataset. 
Some of these datasets can be downloaded in the following ways:
```
# caltech101:
wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1
python -c "import zipfile; zipfile.ZipFile('caltech-101.zip','r').extractall('/home/liuyuyang/ptt/GIFT_CL/data/mtil/caltech101')"

# EuroSAT
wget https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1
wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1

# StanfordCars
git clone https://github.com/jhpohovey/StanfordCars.git
mv StanfordCars/stanford_cars ./stanford_cars
```

**ImageNet1K** we only need the train split, you can download by
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```
`labels.txt` can be found in <https://github.com/fh295/semanticCNN/blob/master/imagenet_labels/labels.txt>


### Step 3: Configure experiment settings

Some commonly used experimental settings are in config file `GNSP/configs/mtil_order_I.json`
You can change the `exp_name` to whatever you like.

### Step 4: Run training

```bash
cd GNSP
python run_mtil.py --config_file=./configs/mtil_order_I.json
```

* Training will start automatically.
* After training completes, the script will automatically perform evaluation.
* You need to manually compute the **Transfer**, **Last**, and **Avg** metrics from the evaluation results.

---

## 5. Output Files

* **Checkpoints**
  Saved in:

  ```
  GNSP/checkpoint/exp_name/
  ```

  Contains model weights for each experiment.

* **Logs**
  Saved in:

  ```
  GNSP/log/exp_name/
  ```

  Contains detailed training and evaluation logs.

* **Gram Matrices**
  Saved in:

  ```
  GNSP/gram_matrix/exp_name/
  ```

  Contains the computed Gram matrices for each dataset.



