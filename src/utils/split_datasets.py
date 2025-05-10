# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/03/16 15:33:33
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 数据集划分
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import os
import csv
import random
from typing import Tuple, List



class DatasetSplitter:
    def __init__(self, root_dir:str, ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        self.root_dir = root_dir
        self.ratios = ratios
        self.all_file = []
        
        self.trian_set = []
        self.val_set = []
        self.test_set = []
        
        # 参数校验
        self._validate_ratios()
        
    def _validate_ratios(self):
        if not isinstance(self.ratios, tuple) or len(self.ratios) != 3:
            raise ValueError("比例参数必须是包含三个元素的元组")
        
        if not all(isinstance(x, (int, float)) for x in self.ratios):
            raise TypeError("比例参数必须是数值类型")
        if round(sum(self.ratios), 2) !=1.0:
            raise ValueError("比例总数必须为1")

    def collect_files(self):
        self.all_file.clear()
        for dirpath, _, _ in os.walk(self.root_dir): # 
            if dirpath != self.root_dir:
                self.all_file.append((os.path.basename(dirpath), dirpath))
        print(f"共找到 {len(self.all_file)} 个文件")

    def split(self, random_seed: int=None):
        if not self.all_file:
            raise RuntimeError("请先调用 collect_file() 收集文件")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        shuffled =random.sample(self.all_file, len(self.all_file))
        
        total = len(shuffled)
        train_end = int(total * self.ratios[0])
        val_end = train_end + int(total*self.ratios[1])    
        self.train_set = shuffled[:train_end]
        self.val_set = shuffled[train_end:val_end]
        self.test_set = shuffled[val_end:]
        
    def save_csv(self, output_dir:str, include_all: bool=True):
        os.makedirs(output_dir, exist_ok=True)
        
        if include_all:
            self._save_dataset(
                data =self.all_file,
                path = os.path.join(output_dir, 'all_file.csv')
            )
            
        datasets = [
            (self.train_set, 'train.csv'),
            (self.val_set, 'val.csv'),
            (self.test_set, 'test.csv')
        ]
        
        for data, filename in datasets:
            self._save_dataset(
                data=data,
                path=os.path.join(output_dir, filename)
            )
        print(f"train.csv saved at {os.path.join(output_dir, 'train.csv')}")
        print(f"val.csv saved at {os.path.join(output_dir, 'val.csv')}")
        print(f"test.csv saved at {os.path.join(output_dir, 'test.csv')}")
    
    @staticmethod
    def _save_dataset(data: List[Tuple[str, str]], path:str):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_idx', 'patient_dir'])
            writer.writerows(data)
            
    @staticmethod
    def diff_datasets(train_csv: str, val_csv: str, test_csv: str, logger):
        """比较三个数据集CSV文件中的患者ID是否有重叠"""
        
        def read_patient_ids(csv_path: str) -> set:
            patients = set()
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        patients.add(row['patient_idx'])
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
            return patients
        
        # 读取各数据集的患者ID
        train_ids = read_patient_ids(train_csv)
        val_ids = read_patient_ids(val_csv)
        test_ids = read_patient_ids(test_csv)
        
        # 比较数据集之间的重叠情况
        diff_result = {
            'train_val_overlap': train_ids & val_ids,
            'train_test_overlap': train_ids & test_ids,
            'val_test_overlap': val_ids & test_ids,
            'all_overlap': train_ids & val_ids & test_ids
        }
        
        # 打印比较结果
        logger.info(f"数据集差异分析结果:\n"
               f"训练集样本数: {len(train_ids)}\n"
               f"验证集样本数: {len(val_ids)}\n"
               f"测试集样本数: {len(test_ids)}\n"
               f"{'-' * 40}"
               )
        
        for key, value in diff_result.items():
            assert len(value)==0, logger.error(f"发现重叠项 {key}: {len(value)} 个重复患者\n示例重复ID: {list(value)[:3]}")
            logger.info(f"{key}: 无重叠")
            
        return diff_result
    
if __name__ == "__main__":
    # 使用示例
    splitter = DatasetSplitter(
        root_dir="/root/workspace/BraTS_Solution/data/brats21/BraTS2021_Training_Data",
        ratios=(0.7, 0.2, 0.1)
    )
    
    # 执行操作流程
    splitter.collect_files()
    splitter.split(random_seed=42)
    splitter.save_csv(output_dir="/root/workspace/BraTS_Solution/data/process/output")
    
    # 比较数据集差异
    # csv_dir = "/root/workspace/BraTS_Solution/data/process/output"
    # DatasetSplitter.diff_datasets(
    #     train_csv=os.path.join(csv_dir, 'train.csv'),
    #     val_csv=os.path.join(csv_dir, 'val.csv'),
    #     test_csv=os.path.join(csv_dir, 'test.csv')
    # )