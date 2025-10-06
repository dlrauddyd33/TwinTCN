import numpy as np
import os
import pickle
from glob import glob
import torch
import random
from tqdm import tqdm


# Get current directory
current_dir = os.getcwd()
train_path = current_dir + '/battery_brand1/train'
test_path = current_dir + '/battery_brand1/test'


train_pkl_files = sorted(glob(train_path + '/*.pkl'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
test_pkl_files = sorted(glob(test_path + '/*.pkl'), key=lambda x: int(x.split("/")[-1].split(".")[0]))


ind_pkl_files = []
ood_pkl_files = []
car_num_list = []


ood_car_num_list = set()
ind_car_num_list = set()


all_car_dict = {}
car_data_dict = {}
for each_path in tqdm(train_pkl_files + test_pkl_files):
    this_pkl_file = torch.load(each_path, weights_only=False)
    this_car_number = this_pkl_file[1]['car']
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
        ind_car_num_list.add(this_car_number)
    else:
        ood_pkl_files.append(each_path)
        ood_car_num_list.add(this_car_number)
    car_num_list.append(this_pkl_file[1]['car'])
    if this_car_number not in car_data_dict:
        car_data_dict[this_car_number] = []
    car_data_dict[this_car_number].append(this_pkl_file)


    if this_car_number not in all_car_dict:
        all_car_dict[this_car_number] = []
        all_car_dict[this_car_number].append(each_path)
    else:
        all_car_dict[this_car_number].append(each_path)


# Create save directory
save_dir = os.path.join(current_dir, 'five_fold_utils/battery_brand1')
os.makedirs(save_dir, exist_ok=True)
# Save data for each car
for car_number, data_list in car_data_dict.items():
    save_path = os.path.join(save_dir, f'car_{car_number}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)


print(f"Successfully saved data for {len(car_data_dict)} cars.")




# shuffle
random.seed(0)
ind_sorted = sorted(ind_car_num_list)
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list)
random.shuffle(ood_sorted)
print(ood_sorted)
print(ind_car_num_list, len(ind_car_num_list))
print(ood_car_num_list, len(ood_car_num_list))
ind_odd_dict = {}
ind_odd_dict["ind_sorted"], ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs(current_dir + '/five_fold_utils', exist_ok=True)
np.save(current_dir + '/five_fold_utils/ind_odd_dict1.npz', ind_odd_dict)


train_path = current_dir + '/battery_brand2/train'
test_path = current_dir + '/battery_brand2/test'


train_pkl_files = sorted(glob(train_path + '/*.pkl'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
test_pkl_files = sorted(glob(test_path + '/*.pkl'), key=lambda x: int(x.split("/")[-1].split(".")[0]))


ind_pkl_files = []
ood_pkl_files = []
car_num_list = []


ood_car_num_list = set()
ind_car_num_list = set()



car_data_dict = {}
for each_path in tqdm(train_pkl_files + test_pkl_files):
    #     print(each_path)
    this_pkl_file = torch.load(each_path, weights_only=False)
    this_car_number = this_pkl_file[1]['car']
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
        ind_car_num_list.add(this_car_number)
    else:
        ood_pkl_files.append(each_path)
        ood_car_num_list.add(this_car_number)
    car_num_list.append(this_pkl_file[1]['car'])
    if this_car_number not in car_data_dict:
        car_data_dict[this_car_number] = []
    car_data_dict[this_car_number].append(this_pkl_file)


    if this_car_number not in all_car_dict:
        all_car_dict[this_car_number] = []
        all_car_dict[this_car_number].append(each_path)
    else:
        all_car_dict[this_car_number].append(each_path)


# Create save directory
save_dir = os.path.join(current_dir, 'five_fold_utils/battery_brand2')
os.makedirs(save_dir, exist_ok=True)


# Save data for each car
for car_number, data_list in car_data_dict.items():
    save_path = os.path.join(save_dir, f'car_{car_number}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)


print(f"Successfully saved data for {len(car_data_dict)} cars.")


print(ind_car_num_list, len(ind_car_num_list))
print(ood_car_num_list, len(ood_car_num_list))
# shuffle
random.seed(0)
ind_sorted = sorted(ind_car_num_list)
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list)
random.shuffle(ood_sorted)
print(ood_sorted)
ind_odd_dict = {}
ind_odd_dict["ind_sorted"], ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs(current_dir + '/five_fold_utils', exist_ok=True)
np.save(current_dir + '/five_fold_utils/ind_odd_dict2.npz', ind_odd_dict)



data_path = current_dir + '/battery_brand3/data'


data_pkl_files = sorted(glob(data_path + '/*.pkl'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
ind_pkl_files = []
ood_pkl_files = []
car_num_list = []


ood_car_num_list = set()
ind_car_num_list = set()


car_data_dict = {}
for each_path in tqdm(data_pkl_files):
    #     print(each_path)
    this_pkl_file = torch.load(each_path, weights_only=False)
    this_car_number = this_pkl_file[1]['car']
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
        ind_car_num_list.add(this_car_number)
    else:
        ood_pkl_files.append(each_path)
        ood_car_num_list.add(this_car_number)
    car_num_list.append(this_pkl_file[1]['car'])
    if this_car_number not in car_data_dict:
        car_data_dict[this_car_number] = []
    car_data_dict[this_car_number].append(this_pkl_file)


    if this_car_number not in all_car_dict:
        all_car_dict[this_car_number] = []
        all_car_dict[this_car_number].append(each_path)
    else:
        all_car_dict[this_car_number].append(each_path)


# Create save directory
save_dir = os.path.join(current_dir, 'five_fold_utils/battery_brand3')
os.makedirs(save_dir, exist_ok=True)


# Save data for each car
for car_number, data_list in car_data_dict.items():
    save_path = os.path.join(save_dir, f'car_{car_number}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)


print(f"Successfully saved data for {len(car_data_dict)} cars.")




print(ind_car_num_list, len(ind_car_num_list))
print(ood_car_num_list, len(ood_car_num_list))
# shuffle
random.seed(0)
ind_sorted = sorted(ind_car_num_list)
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list)
random.shuffle(ood_sorted)
print(ood_sorted)
ind_odd_dict = {}
ind_odd_dict["ind_sorted"], ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs(current_dir + '/five_fold_utils', exist_ok=True)
np.save(current_dir + '/five_fold_utils/ind_odd_dict3.npz', ind_odd_dict)



# save all the three brands path information
np.save(current_dir + '/five_fold_utils/all_car_dict.npz', all_car_dict)
