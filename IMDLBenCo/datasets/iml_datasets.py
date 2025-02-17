import json
import os


from ..registry import DATASETS
from .abstract_dataset import AbstractDataset
    
@DATASETS.register_module()
class ManiDataset(AbstractDataset):
    def _init_dataset_path(self, path):
        path = path
        tp_dir = os.path.join(path, 'tampered')
        gt_dir = os.path.join(path, 'mask')
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        # Use sort mathod to keep order, to make sure the order is the same as the order in the tp_list and gt_list
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'tampered', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'mask', gt_list[index]) for index in range(len(gt_list))]
        return t_tp_list, t_gt_list
    
@DATASETS.register_module()
class ManiAuDataset(AbstractDataset):
    def _init_dataset_path(self, path):
        path = path
        tp_dir = os.path.join(path, 'tampered')
        gt_dir = os.path.join(path, 'mask')
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        # Use sort mathod to keep order, to make sure the order is the same as the order in the tp_list and gt_list
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'tampered', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'mask', gt_list[index]) for index in range(len(gt_list))]
        
        au_dir = os.path.join(path, 'authentic')
        if not os.path.exists(au_dir):
            raise AttributeError('no authentic images')
        else:
            au_list = os.listdir(au_dir)
            au_list.sort()
            a_au_list = [os.path.join(path, 'authentic', au_list[index]) for index in range(len(au_list))]
            a_gt_list = ["Negative" for index in range(len(au_list))]
            t_tp_list = t_tp_list + a_au_list
            t_gt_list = t_gt_list + a_gt_list
            
        return t_tp_list, t_gt_list

    
@DATASETS.register_module()
class JsonDataset(AbstractDataset):
    """ init from a json file, which contains all the images path
        file is organized as:
            [
                ["./Tp/6.jpg", "./Gt/6.jpg"],
                ["./Tp/7.jpg", "./Gt/7.jpg"],
                ["./Tp/8.jpg", "Negative"],
                ......
            ]
        if path is "Neagative" then the image is negative sample, which means ground truths is a totally black image, and its label == 0.
        
    Args:
        path (_type_): _description_
        transform_albu (_type_, optional): _description_. Defaults to None.
        mask_edge_generator (_type_, optional): _description_. Defaults to None.
        if_return_shape
    """
    def _init_dataset_path(self, path):
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            if os.path.isfile(record[0]):
                tp_list.append(record[0])
                gt_list.append(record[1])
            else: 
                raise TypeError("Not a file in Json Dataset Error. Try other dataset")
        return tp_list, gt_list

