import pdb
import os
import pandas as pd
import numpy as np
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
                    help='fraction of labels (default: [0.25, 0.5, 0.75, 1.0])')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str,
choices=['camelyon_40x_cv',
'cardiac_normal_rejction',
'cardiac-amr-grades',
'cardiac-cell-grades',
'cardiac-cell-amr-grades',
'cardiac-mtl',
'cardiac-mtl-noQ',
'cardiac-stl'])

parser.add_argument('--hold_out_test', action='store_true', default=False,
                    help='hold-out the test set for each split')
parser.add_argument('--split_code', type=str, default=None)

args = parser.parse_args()

if args.task == 'camelyon_40x_cv':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/camelyon_clean.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set


elif args.task == 'cardiac_normal_rejction':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacFullSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'healthy':0, 'quilty':1,
					  'cell_only_low':2, 'cell_only_high':3,'cell_low_quilty':4, 'cell_high_quilty':5,
					  'amr_only_low':6, 'amr_only_high':7, 'amr_low_quilty':8, 'amr_high_quilty':9,
					  'cell_amr_low':10,'cell_amr_high':11, 'cell_amr_quilty_low':12, 'cell_amr_quilty_high':13},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set


elif args.task == 'cardiac-amr-grades':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacAmrGradesSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'amr_only_low'  		: 0,
			    		  'amr_only_high' 		: 1,
                                          'amr_low_quilty'              : 2,
                                          'amr_high_quilty'             : 3,
                                          'cell_amr_low'                : 4,
                                          'cell_amr_high'               : 5,
                                          'cell_amr_quilty_low'         : 6,
                                          'cell_amr_quilty_high'        : 7},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set

elif args.task == 'cardiac-cell-grades':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacCellGradesSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'cell_only_low'		: 0,
			    		  'cell_only_high'		: 1,
                                          'cell_low_quilty'             : 2,
                                          'cell_high_quilty'            : 3,
                                          'cell_amr_low'                : 4,
                                          'cell_amr_high'               : 5,
                                          'cell_amr_quilty_low'         : 6,
                                          'cell_amr_quilty_high'        : 7},
                            patient_strat= True,
                            ignore=[])
    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set



elif args.task == 'cardiac-cell-amr-grades':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacCellAmrGradesSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'cell_only_low'		: 0,
			    		  'cell_only_high'		: 1,
					  'cell_low_quilty'		: 2,
					  'cell_high_quilty'		: 3,
					  'amr_only_low' 		: 4,
					  'amr_only_high'		: 5,
					  'amr_low_quilty'		: 6,
					  'amr_high_quilty'		: 7,
					  'cell_amr_low'		: 8,
					  'cell_amr_high'		: 9,
					  'cell_amr_quilty_low'		: 10,
					  'cell_amr_quilty_high'	: 11},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # to use hold-out test set set p_test = 0


# split based on unique classes in data not the same as classes in multi-task part
elif args.task == 'cardiac-mtl':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacFullSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'healthy':0, 'quilty':1,
                                          'cell_only_low':2, 'cell_only_high':3,'cell_low_quilty':4, 'cell_high_quilty':5,
                                          'amr_only_low':6, 'amr_only_high':7, 'amr_low_quilty':8, 'amr_high_quilty':9,
                                          'cell_amr_low':10,'cell_amr_high':11, 'cell_amr_quilty_low':12, 'cell_amr_quilty_high':13},
                            patient_strat= True,
                            ignore=[])
			    
			    

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set


# split based on unique classes in data not the same as classes in multi-task part
elif args.task == 'cardiac-mtl-noQ':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacFullSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
 			    label_dict = {'healthy':0, 'quilty':1,
                                          'cell_only_low':2, 'cell_only_high':3,'cell_low_quilty':4, 'cell_high_quilty':5,
                                          'amr_only_low':6, 'amr_only_high':7, 'amr_low_quilty':8, 'amr_high_quilty':9,
                                          'cell_amr_low':10,'cell_amr_high':11, 'cell_amr_quilty_low':12, 
					  'cell_amr_quilty_high':13},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set


# split based on unique classes in data not the same as classes in multi-task part
elif args.task == 'cardiac-stl':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CardiacFullSplit.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'healthy':0, 'quilty':1,
                                          'cell_only_low':2, 'cell_only_high':3,'cell_low_quilty':4, 'cell_high_quilty':5,
                                          'amr_only_low':6, 'amr_only_high':7, 'amr_low_quilty':8, 'amr_high_quilty':9,
                                          'cell_amr_low':10,'cell_amr_high':11, 'cell_amr_quilty_low':12, 'cell_amr_quilty_high':13},
                            patient_strat= True,
                            ignore=[])

    p_val  = 0.1   # use 10% of data in validation
    p_test = 0.2   # use 20% data for test set


else:
    raise NotImplementedError


# splits
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.floor(num_slides_cls * p_val).astype(int)      # use 10% data in validation
test_num = np.floor(num_slides_cls * p_test).astype(int)     # use 20% for test set
print("---------------------------------")
print(f"validation set size = {val_num} ")
print(f"test set size       = {test_num}")
print("---------------------------------")


if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.25, 0.5, 0.75, 1.0]

    if args.hold_out_test:
        custom_test_ids = dataset.sample_held_out(test_num=test_num)
    else:
        custom_test_ids = None

    for lf in label_fracs:
        if args.split_code is not None:
            split_dir = 'splits/'+ str(args.split_code) + '_{}'.format(int(lf * 100))
        else:
            split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))

        os.makedirs(split_dir, exist_ok=True)
        #pdb.set_trace()
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, custom_test_ids=custom_test_ids)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val','test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val','test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))

