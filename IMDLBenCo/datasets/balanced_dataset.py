import random
from torch.utils.data import Dataset, DataLoader

from .iml_datasets import JsonDataset, ManiDataset, ManiAuDataset

from ..transforms import get_albu_transforms
from .utils import pil_loader, denormalize

from IMDLBenCo.registry import DATASETS
@DATASETS.register_module()
class BalancedDataset(Dataset):
    """The BalancedDataset manages multiple iml_datasets, so it does not inherit from AbstractDataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, 
                 sample_number = 7568,
                 *args, 
                 **kwargs
                ) -> None:
        self.sample_number = sample_number
        self.settings_list = [
                ['/data/zhujy/IMDL/Casiav2_revised', ManiAuDataset],
                ['/data/zhujy/IMDL/FantasticReality', ManiAuDataset], 
        ]

        self.dataset_list = [self._get_dataset(path, dataset_type, *args, **kwargs) for path, dataset_type in self.settings_list]
        
        
    def _get_dataset(self, path, dataset_type, *args, **kwargs):
        return dataset_type(path, *args, **kwargs)
        
        
    def __len__(self):
        return self.sample_number * len(self.settings_list)    
    
    def __getitem__(self, index):
        dataset_index = index // self.sample_number

        selected_dataset = self.dataset_list[dataset_index]
        length = len(selected_dataset)
        selected_item = random.randint(0, length - 1)
        return selected_dataset[selected_item]

@DATASETS.register_module()
class UnBalancedDataset(Dataset):
    """The BalancedDataset manages multiple iml_datasets, so it does not inherit from AbstractDataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, 
                 # path = None, 
                 sample_number = 5123,
                 all_number = 7172, 
                 *args, 
                 **kwargs
                ) -> None:
        self.sample_number = sample_number
        self.all_number = all_number
        # Defalut
        self.settings_list = [
            ['/data/zhujy/IMDL/Casiav2_revised', ManiDataset],
            ['/data/zhujy/IMDL/FantasticReality', ManiDataset],       
        ]
            

        self.dataset_list = [self._get_dataset(path, dataset_type, *args, **kwargs) for path, dataset_type in self.settings_list]
        
        
    def _get_dataset(self, path, dataset_type, *args, **kwargs):
        return dataset_type(path, *args, **kwargs)
        
        
    def __len__(self):
        return self.all_number  
    
    def __getitem__(self, index):
        if index < self.sample_number:
            dataset_index = 0
            selected_dataset = self.dataset_list[dataset_index]
            length = len(selected_dataset)
            selected_item = index % self.sample_number
            return selected_dataset[selected_item]
        else:
            dataset_index = 1
            selected_dataset = self.dataset_list[dataset_index]
            length = len(selected_dataset)
            selected_item = random.randint(0, length - 1)
            return selected_dataset[selected_item]
        
        
@DATASETS.register_module()
class UnBalancedAuDataset(Dataset):
    """The BalancedDataset manages multiple iml_datasets, so it does not inherit from AbstractDataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, 
                 # path = None, 
                 sample_number = 12614,
                 all_number = 17660, 
                 *args, 
                 **kwargs
                ) -> None:
        self.sample_number = sample_number
        self.all_number = all_number
        # Defalut
        self.settings_list = [
            ['/data/zhujy/IMDL/Casiav2_revised', ManiAuDataset],
            ['/data/zhujy/IMDL/FantasticReality', ManiAuDataset],       
        ]
            

        self.dataset_list = [self._get_dataset(path, dataset_type, *args, **kwargs) for path, dataset_type in self.settings_list]
        
        
    def _get_dataset(self, path, dataset_type, *args, **kwargs):
        return dataset_type(path, *args, **kwargs)
        
        
    def __len__(self):
        return self.all_number  
    
    def __getitem__(self, index):
        if index < self.sample_number:
            dataset_index = 0
            selected_dataset = self.dataset_list[dataset_index]
            length = len(selected_dataset)
            selected_item = index % self.sample_number
            return selected_dataset[selected_item]
        else:
            dataset_index = 1
            selected_dataset = self.dataset_list[dataset_index]
            length = len(selected_dataset)
            selected_item = random.randint(0, length - 1)
            return selected_dataset[selected_item]