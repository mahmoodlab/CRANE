CRANE <img src="Logo1.png" width="150px" align="right" />
===========
### Deep learning enabled assessment of cardiac allograft rejection from endomyocardial biopsies
*Nature Medicine*

Read Link | [Journal Link](https://www.nature.com/articles/s41591-022-01709-2) | [Interactive Demo](http://crane.mahmoodlab.org) | [Cite](#reference) 

*This study is currently under a press embargo until 03/21/22 11am ET, address all questions to fmahmood@fas.harvard.edu*

CRANE  is  a  high-throughput,  interpretable,  multi-task  framework  that  simultaneously  address the main diagnostic tasks in endomyocardial biopsy screening, detection of: acute cellular rejection, antibody-mediated rejection and quilty B lesions, as well as their concurrent appearances (e.g.cellular rejection with quilty lesions). For the detected rejection, the model further estimates the rejection grade.


Thanks  to  the  weakly-supervised  formulation,  the  model  can  be  trained  using  the  patient  diagnosis  as  the only label, surpassing the needs and limitations of manually annotated diagnostic regions for each task.  The architecture of the CRANE model is depicted in Figure 2. CRANE takes as input digitized H&E stained WSI, which represents the gold standard in endomyocardial biopsy assessment.

<img src="EMB.png" width="500px" align="center" />\
**Figure 1: Endomyocardial biopsy assessment.** 
*Fragments of endomyocardial tissue biopsied from the right ventricle underwent formalin fixation and paraffin embedding (FFPE). Each paraffin block was cut into slides with three consecutive levels and stained with hematoxylin and eosin(H&E). Each slide was digitized and served as an input for the model. Ground truth diagnoses to include rejection type and severity were distilled from each pathology report.*

<img src="Model.png" width="1000px" align="center" />\
**Figure 2: Crane model.** 
*A weakly-supervised multi-task, multi-label network was constructed to simultaneously identify healthy tissue and different rejection conditions (cellular, antibody and/or quilty lesions).  A multiple instance learning approach is used to enable model training from the patient’s diagnosis as the only labels, surpassing the need for slide-level diagnosis or pixel-level annotations of relevant image regions.  A separate classifier is trained to estimate the rejection grade.  The model assigns attention scores to each image region reflecting its relevance for the diagnosis. The high-attention regions can be used to interpret the model’s prediction and validate the accuracy of the diagnosis. Shown attention heat map for the cellular rejection task.*



## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 16)
* Python (3.7.5), h5py (2.10.0), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), openslide (3.4.1), pandas (0.25.3), pillow (7.0.0), PyTorch (1.5.0), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.4.2).

### Installation Guide for Linux (using anaconda)
[Installation Guide](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md)


### Data Preparation
To process the WSI data we used the publicly available [CLAM WSI-analysis toolbox](https://github.com/mahmoodlab/CLAM). First, the tissue regions in each biopsy slide are segmented. Then 256x256 patches (without spatial overlapping) are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained truncated ResNet50 is used to encode each raw image patch into a 1024-dim feature vector. In the CLAM toolbox, the features are saved as matrices of torch tensors of size N x 1024, where N is the number of patches from each WSI (varies from slide to slide). The extracted features then serve as input to the network. The following folder structure is assumed for the extracted features vectors:    
```bash
DATA_ROOT_DIR/
    └──features/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```
DATA_ROOT_DIR is the base directory of all datasets (e.g. the directory to your SSD) and features is a folder containing features from each slide stored as a .pt files. Please refer to [CLAM](https://github.com/mahmoodlab/CLAM) for examples on tissue segmentation and feature extraction. 

### Datasets
The model takes as an input list of data in the form of a csv file containing the following columns: **case_id**, **slide_id**. Each **case_id** is a unique identifier for a patient, while the **slide_id** is a unique identifier for a slide that corresponds to the name of an extracted feature .pt file. In this way, multiple slides from a patient can be easily tracked. The remaining columns in the csv file correspond to the labels stored under the following headers: **label_cell, label_amr, label_quilty**, or **label_grade**. The first three labels describe the state of EMB, used by the multi-task network, while the labels for the Grading network are specified in **label_grade**. We provide dummy examples of the dataset csv files in the **dataset_csv** folder called *CardiacDummy_MTL.csv* (for the EMB assessment network) and *CardiacDummy_Grade.csv* (for the grading network). You are free to input the labels for your data in any way as long as you specify the appropriate dictionary maps under the label_dicts argument of the dataset object's constructor (see below). For demonstration purposes, we use 'low' and 'high' for the grade labels. In the multi-task problem, the state of cellular rejection, represented by **label_cell** is encoded as 'no_cell' and 'cell' to express the absence or presence of cellular rejection. Similarly, the antibody-mediated rejections, specified in **label_amr** are marked as 'no_amr' and 'amr', while quilty lesions (**label_quilty**) are specified as 'no_quilty' and 'quilty'.

Dataset objects used for actual training/validation/testing can be constructed using the *Generic_MIL_MTL_Dataset* Class (for the multi-task problem) and *Generic_MIL_Dataset* (for the grading task), defined in *datasets/dataset_mtl.py* and *datasets/dataset_generic.py*. Examples of such dataset objects passed to the models can be found in both **main.py** and **eval.py**: 

```python
if args.task == 'cardiac-grade':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CardiacDummy_Grade.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'low':0, 'high':1},
                            label_cols=['label_grade']
                            patient_strat=False,
                            ignore=[])


elif args.task == 'cardiac-mtl':
    args.n_classes=[2,2,2]
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/CardiacDummy_MTL.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dicts = [{'no_cell':0, 'cell':1},
                                            {'no_amr':0, 'amr':1},
                                            {'no_quilty':0, 'quilty':1}],
                            label_cols=['label_cell','label_amr','label_quilty'],
                            patient_strat=False,
                            ignore=[])

```
In addition, the following arguments need to be specified:
* csv_path (str): Path to the dataset csv file
* data_dir (str): Path to saved .pt features for the dataset
* label_dicts (list of dict): List of dictionaries with key, value pairs for converting str labels to int for each label column
* label_cols (list of str): List of column headings to use as labels and map with label_dicts

Finally, the user should add this specific 'task' specified by this dataset object to be one of the choices in the --task arguments as shown below:

```python
parser.add_argument('--task', type=str, choices=['cardiac-mtl, cardiac-grade'])
```



### Training Splits
For evaluating the algorithm's performance, we randomly partitioned our dataset into training, validation and test splits. The split is constructed automatically based on the dataset provided in the csv file. To ensure balance proportion of diagnosis across all splits, an additional csv dataset file can be constructed to account for all the unique combinations of diagnosis (e.g. antibody-mediated rejection alone or antibody-mediated rejection with quilty lesions). The folder **dataset_csv** contains two dummy csv files: *CardiacDummy_MTLSplit.csv* and *CardiacDummy_GradeSplit.csv* which illustrates all possible combinations of diagnosis for the EMB and for the grading network, respectively. The csv file has to contain the following three entries: **case_id**, **slide_id** and **label**. The splits are constructed based on the patient's unique identifier specified in **case_id**, so all the slides from the given patient are presented in the same split. The **label** marks the possible diagnosis that should be considered during the split. We consider 14 distinct classes for the EMB assessment network as shown in the *label_dict* of the **create_splits.py** script:
```python
if args.task == 'cardiac-mtl':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacDummy_MTLSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'healthy'                 :0,
                                          'quilty'                  :1,
                                          'cell_only_low'           :2,
                                          'cell_only_high'          :3,
                                          'cell_low_quilty'         :4,
                                          'cell_high_quilty'        :5,
                                          'amr_only_low'            :6,
                                          'amr_only_high'           :7,
                                          'amr_low_quilty'          :8,
                                          'amr_high_quilty'         :9,
                                          'cell_amr_low'            :10,
                                          'cell_amr_high'           :11,
                                          'cell_amr_quilty_low'     :12,
                                          'cell_amr_quilty_high'    :13},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% of data for test-set (held out set), remaining 70% will be used for training
```
where p_val and p_test control the percentage of samples that should be used for validation and testing, respectively. A similar approach is used to split the data for the Grading network (see create_splits.py).

The splits can be then created using the **create_splits.py** script as follows:
``` shell
python create_splits.py --task cardiac-mtl --seed 1 --k 1
python create_splits.py --task cardiac-grade --seed 1 --k 1
```
where k is the number of folds. 


### Training
To following command can be used to train the multi-task model:
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code cardiac_output  --task cardiac-mtl --mtl --log_data  --data_root_dir DATA_ROOT_DIR --drop_out
```
while the grading network training can be performed as follows:
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code cardiac_output  --task cardiac-grade --log_data  --data_root_dir DATA_ROOT_DIR --drop_out
```
The GPU id(s) to be used can be specified using CUDA_VISIBLE_DEVICES, in the example command, the 1st GPU is used. Other arguments such as --drop_out, --early_stopping, --lr, --reg, and --max_epochs can be specified to customize your experiments. 
For information on each argument, see:
``` shell
python main.py -h
```

By default, results will be saved to **results/exp_code** corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the argument toggle --log_data), the user can go into the results folder for the particular experiment, run:
``` shell
tensorboard --logdir=.
```
This should open a browser window and show the logged training/validation statistics in real-time.


### Evaluation
User also has the option of using the evaluation script to test the performances of trained models. Examples corresponding to the models trained above are provided below:
``` shell
CUDA_VISIBLE_DEVICES=0 python eval.py --task cardiac-mtl --results_dir results --models_exp_code cardiac-mtl_100_s1 --drop_out --k 1  --data_root_dir DATA_ROOT_DIR --mtl 
 
CUDA_VISIBLE_DEVICES=0 python eval.py --task cardiac-grade --results_dir results --models_exp_code cardiac-grade_100_s1 --drop_out --k 1 --data_root_dir DATA_ROOT_DIR 
```
For information on each commandline argument, see:
``` shell
python eval.py -h
```

<img src="Heatmaps.png" width="1000px" align="center" />\
**Figure 3: Attention maps of regions used to make classification determinations.**  
*Cellular Rejection. The  highest  attention  regions  (red)  identified  regions  with  increased  interstitial  lymphocytic  infiltrates  andassociated  myocyte  injury,  while  the  adjacent,  lower  attention  regions  (blue)  identified  healthier  myocyteswithout injury. b.Antibody Mediated Rejection.  The highest attention regions identified regions with edema within the interstitial spaces in addition to increased mixed inflammatory infiltrate, comprised of eosinophils,neutrophils,  and  lymphocytes.  The  adjacent  lower  attention  regions  (blue)  identified  background  fibrosis,stroma, and healthier myocytes. c.  Quilty B Lesion.  The highest attention regions (red) identified a single, benign focus lymphocytes within the endocardium, without injury or damage to the myocardium.  The lower attention regions (blue) identified background, healthy myocytes. d.Cellular Grade.  The highest attention regions (red) identified diffuse, prominent interstitial lymphocytic infiltrates with associated myocyte injury, representing severe rejection. The lower attention regions (blue) identified background fibrosis and unaffected, healthier myocytes. Additional examples of multi-label attention maps can be viewed in the interactive (demo)[http://crane.mahmoodlab.org]*


## Issues
- Please report all issues on the public forum.

## License
© [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper:

Lipkova, J., Chen, T.Y., Lu, M.Y., Shady, M., Williams, M., Wang, J., Mitchell, R.N., Turan, M., Coskun, G., Demir, D. and Nart, D. et al. Deep learning-enabled assessment of cardiac allograft rejection from endomyocardial biopsies. Nature Medicine (2022). https://doi.org/10.1038/s41591-022-01709-2

```
@article{lu2021data,
  title={Deep learning-enabled assessment of cardiac allograft rejection from endomyocardial biopsies.},
  author={Lipkova, Jana and Chen, Tiffany Y and Lu, Ming Y and Shady, Maha and Williams, Mane and Wang, Jingwen and Mitchell, Richard N and Turan, Mehmet and Coskun, Gulfize and Demir, Derya and others},
  journal={Nature Medicine},
  year={2022},
  publisher={Nature Publishing Group}
}
```

