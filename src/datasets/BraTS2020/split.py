from pathlib import Path
import glob
import os
import pandas as pd

def search_instances(data_root, pattern="BraTS19*", csv_saved_dir=None, filename=None):
    """
    Search for instances in the dataset directory that match the given pattern.
    """
    pattern = str(data_root / pattern)
    
    
    csv_saved_path = Path(csv_saved_dir) / filename if filename else Path(csv_saved_dir) / "all_instances.csv"
    
    folder_paths = [
        path for path in glob.glob(pattern)
        if os.path.isdir(path)
    ]
    
    instance_names = [
        Path(p).name for p in folder_paths    
        ]
    
    anlysizes_dict = {
        "patient_id": instance_names,
        "patient_dir": folder_paths
        
    }
    df_results = pd.DataFrame(anlysizes_dict)
    df_results.to_csv(csv_saved_path, index=False)
    print(f"Found {len(df_results)} instances matching pattern '{pattern}'")
    print(f"Saved results to {csv_saved_path}")
    return df_results

def split_datasets(df, ratios=(0.7,0.2,0.1), csv_saved_dir=None):
    """
    Split the dataset into training, validation, and test sets.
    """
    # assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    total_instances = len(df)
    train_end = int(total_instances * ratios[0])
    val_end = int(total_instances * (ratios[0] + ratios[1]))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    if csv_saved_dir:
        train_df.to_csv(Path(csv_saved_dir) / "train.csv", index=False)
        val_df.to_csv(Path(csv_saved_dir) / "val.csv", index=False)
        test_df.to_csv(Path(csv_saved_dir) / "test.csv", index=False)
        print(f"Saved split datasets to {csv_saved_dir}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    data_root = Path("/root/workspace/DCLA-UNet/data/BraTS2020/raw/BraTS2020_TrainingData")
    csv_saved_dir = Path("/root/workspace/DCLA-UNet/data/BraTS2020/raw/")
    
    # Search for instances
    df_instances = search_instances(data_root, pattern="BraTS20*", csv_saved_dir=csv_saved_dir, filename="train.csv")
    
    # Split the dataset
    train_df, val_df, test_df = split_datasets(df_instances, ratios=(0.7, 0.2, 0.1), csv_saved_dir=csv_saved_dir)
    
    # data_root = Path("/root/workspace/DCLA-UNet/data/BraTS2020/raw/BraTS2020_ValidationData")
    # csv_saved_dir = Path("/root/workspace/DCLA-UNet/data/BraTS2020/raw/")
    
    # # Search for instances
    # df_instances = search_instances(data_root, pattern="BraTS20*", csv_saved_dir=csv_saved_dir, filename="train.csv")
    
    # # Split the dataset
    # train_df, val_df, test_df = split_datasets(df_instances, ratios=(0, 0.66, 0.33), csv_saved_dir=csv_saved_dir)