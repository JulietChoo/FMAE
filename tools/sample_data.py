import os
import numpy as np
import shutil

def read_file(path):
    file_list = os.listdir(path)
    file_path_list = [os.path.join(path, img) for img in file_list]
    file_path_list.sort()
    return file_path_list

def sample_dataset_loc(datasets, dst_dir):
    # only sample tampered images
    path = '/data/zhujy/IMDL'
    dst_tp_dir = os.path.join(dst_dir, 'tampered')
    dst_mk_dir = os.path.join(dst_dir, 'mask')
    os.makedirs(dst_tp_dir, exist_ok=True)
    os.makedirs(dst_mk_dir, exist_ok=True)
    for dataset in datasets:
        tp_data_dir = os.path.join(path, dataset, 'tampered')
        mk_data_dir = os.path.join(path, dataset, 'mask')
        tp_list = read_file(tp_data_dir)
        mk_list = read_file(mk_data_dir)
        N_dataset = range(len(os.listdir(tp_data_dir)))
        N_sample = 100
        sample_index = np.random.choice(N_dataset, size=N_sample, replace=False)
        print(len(sample_index))
        for index in sample_index:
            img_name = tp_list[index]
            mask_name = mk_list[index]
            print(img_name)
            shutil.copy(img_name, dst_tp_dir)
            shutil.copy(mask_name, dst_mk_dir)

def sample_dataset_det(datasets, dst_dir):
    # sample authentic and tampered images
    path = '/data/zhujy/IMDL'
    dst_tp_dir = os.path.join(dst_dir, 'tampered')
    dst_mk_dir = os.path.join(dst_dir, 'mask')
    dst_au_dir = os.path.join(dst_dir, 'authentic')
    os.makedirs(dst_tp_dir, exist_ok=True)
    os.makedirs(dst_mk_dir, exist_ok=True)
    os.makedirs(dst_au_dir, exist_ok=True)
    for dataset in datasets:
        tp_data_dir = os.path.join(path, dataset, 'tampered')
        mk_data_dir = os.path.join(path, dataset, 'mask')
        au_data_dir = os.path.join(path, dataset, 'authentic')
        tp_list = read_file(tp_data_dir)
        mk_list = read_file(mk_data_dir)
        au_list = read_file(au_data_dir)
        N_tp = range(len(os.listdir(tp_data_dir)))
        N_dataset = len(os.listdir(tp_data_dir)) + len(os.listdir(au_data_dir))
        N_tp_sample = round(300*len(os.listdir(tp_data_dir))/N_dataset)
        N_au = range(len(os.listdir(au_data_dir)))
        N_au_sample = round(300*len(os.listdir(au_data_dir))/N_dataset)
        sample_index_tp = np.random.choice(N_tp, size=N_tp_sample, replace=False)
        sample_index_au = np.random.choice(N_au, size=N_au_sample, replace=False)
        print(len(sample_index_tp))
        print(len(sample_index_au))
        for index in sample_index_tp:
            img_name = tp_list[index]
            mask_name = mk_list[index]
            print(img_name)
            shutil.copy(img_name, dst_tp_dir)
            shutil.copy(mask_name, dst_mk_dir)
        for index in sample_index_au:
            img_name = au_list[index]
            print(img_name)
            shutil.copy(img_name, dst_au_dir)

if __name__ == '__main__':
    datasets = ['Casiav1', 'Columbia', 'IMD2020', 'Autosplice']
    dst_dir = '/data/zhujy/IMDL/SampleDataset_DF_det_300'
    sample_dataset_det(datasets, dst_dir)

    datasets = ['Casiav1', 'Columbia', 'NIST', 'IMD2020', 'DSO-1', 'Korus', 'Autosplice', 'OpenForensics']
    dst_dir = '/data/zhujy/IMDL/SampleDataset_DF_loc_100'
    sample_dataset_loc(datasets, dst_dir)