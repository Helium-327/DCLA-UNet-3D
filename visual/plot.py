import os
import random
import numpy as np
import pandas as pd 
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
# %matplotlib inline

COLOR_MAP = ListedColormap([(0,0,0), (1,0,0), (0,1,0), (0,0,1)], name='custom_discrete', N=4)
BOUNDARIES = [0, 1, 2, 3, 4]
NORM = BoundaryNorm(BOUNDARIES, COLOR_MAP.N)
MODALITIES = ['flair', 't1', 't2', 't1ce', 'seg']

DATA_DIR = "/root/workspace/BraTS_Solution/data/brats21/BraTS2021_Training_Data"

class MultiModalityLoader():
    def __init__(self, data_dir, modalities, transform=None):
        self.data_dir = data_dir
        self.modals = modalities
        
    def get_patient_ids(self):
        return [d.split('_')[-1] for d in os.listdir(self.data_dir) if d.startswith('BraTS2021')]
        
    def get_id_paths(self, patient_id):
        paths_dict = {}
        paths_dict[patient_id] = {}
        for modal in self.modals:
            path = os.path.join(self.data_dir, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_{modal}.nii.gz")
            paths_dict[patient_id][modal] = path
        return paths_dict
    
    def get_all_paths(self, length=100):
        all_paths_dict = {}
        for patient_id in self.get_patient_ids()[:length] if length else self.get_patient_ids():
            all_paths_dict[patient_id] = self.get_id_paths(patient_id)
        return all_paths_dict
    

    def __getitem__(self, patient_id):
        data = {}
        data[patient_id] = {}
        for modality in self.modals:
            path = os.path.join(self.data_dir, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_{modality}.nii.gz")
            if modality == 'seg':
                data[patient_id][modality] = nib.load(path).get_fdata().transpose((2, 0, 1)).astype(np.int8)
            else:
                data[patient_id][modality] = nib.load(path).get_fdata().transpose((2, 0, 1)).astype(np.float32)
        return data
    
    def __len__(self):
        return len(self.get_patient_ids())
    
    def __iter__(self):
        for patient_id in self.get_patient_ids():
            yield self[patient_id]
    
    
def plot_data(slice_dict, axs_size=3, cmap='bone'):
    for id, id_slices in slice_dict.items():
        figsize= (axs_size*len(id_slices), axs_size)
        fig, axs = plt.subplots(1, len(id_slices), figsize=figsize)
        fig.text(0.1, 0.5, f'ID: {id}', va='center', ha='right', rotation=90, fontsize=8)
        for i, (modal, slice) in enumerate(id_slices.items()):
            if modal == 'seg':
                modal = 'GroudTruth'
                axs[i].imshow(slice, cmap=COLOR_MAP, norm=NORM, alpha=1)
                # axs[i].set_title(f'{modal}')
                axs[i].axis('off')
            else:
                axs[i].imshow(slice, cmap=cmap)
                # axs[i].set_title(f'{modal}')
                axs[i].axis('off')
        plt.show()
        
class MultiModalityPloter:
    def __init__(self, data_dict, cmap='bone'):
        self.data_dict = data_dict
        self.cmap = cmap
        
    def get_slices(self, axis, slice_idx=100, cmap='bone'):
        """获取指定轴和切片的图像数据"""
        
        slice_dict = {}
        for patient_id, id_data in self.data_dict.items():
            slice_dict[patient_id] = {}
            for modal, data in id_data.items():
                slice_dict[patient_id][modal] = self.get_slice_from_axis(data, axis, slice_idx)
        return slice_dict

    def get_slice_from_axis(self, data, axis, index):
    # print(data.shape)
        slices = [slice(None)] * len(data.shape)
        slices[axis] = index    
        slice_index = data[tuple(slices)]
        print(slice_index.shape)            # [slice(None, None, None), slice(None, None, None), slice(None, None, None)]             # [slice(None, None, None), index, slice(None, None, None)]
        return slice_index
    

def plot_data(slice_dict, axs_size=3, cmap='bone'):
    for id, id_slices in slice_dict.items():
        figsize= (axs_size*len(id_slices), axs_size)
        fig, axs = plt.subplots(1, len(id_slices), figsize=figsize)
        fig.text(0.1, 0.5, f'ID: {id}', va='center', ha='right', rotation=90, fontsize=8)
        for i, (modal, slice) in enumerate(id_slices.items()):
            if modal == 'seg':
                modal = 'GroudTruth'

                axs[i].imshow(slice, cmap=cmap, alpha=0)
                # axs[i].set_title(f'{modal}')
                # axs[i].axis('off')
            else:
                axs[i].imshow(slice, cmap=cmap)
                # axs[i].set_title(f'{modal}')
                # axs[i].axis('off')
        plt.show()


if __name__ == "__main__":

    loader = MultiModalityLoader(DATA_DIR, MODALITIES)



    ploter = MultiModalityPloter(loader['00018'])

    # 
    slice_dict = ploter.get_slices(0, 100)



    plot_data(slice_dict, axs_size=0, cmap='bone')


# all_paths = loader.get_all_paths()
# all_intensity = loader.get_all_IntensityRange()
# print(loader['00003']['flair'].max())
# print(loader['00003']['t1'].max())
# print(loader['00003']['t2'].max())
# print(loader['00003']['t1ce'].max())
# all_paths
# all_intensity

# plt.imshow(loader['00018']['00018']['t1'][100])