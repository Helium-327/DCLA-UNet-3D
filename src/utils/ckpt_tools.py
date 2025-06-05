# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:11:38
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 模型保存和加载
*      VERSION: v1.0
=================================================
'''

import torch
import os

# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:11:38
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 模型保存和加载
*      VERSION: v2.0
=================================================
'''

import torch
import os

import datetime
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    checkpoint_path:str,
    scheduler=None,
    load_weights_only= True,
    strict=True
):
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=load_weights_only)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 智能优化器加载
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Full optimizer state loaded")
            except ValueError:
                # 参数组不匹配时仅继承学习率
                old_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = old_lr
                print(f"⚠ Partial load: Reset all param_groups lr to {old_lr}")
                
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scheduler.last_epoch = checkpoint['epoch']
            
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # 参数匹配检查
        model_params = set(model.state_dict().keys())
        ckpt_params = set(checkpoint['model_state_dict'].keys())
        print(f"Loaded params: {len(model_params & ckpt_params)}/{len(model_params)} matched")
        print(f"*** Resuming from epoch {start_epoch} | Best val loss: {best_val_loss:.4f}")
        return model, optimizer, scaler, start_epoch, best_val_loss, scheduler
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("⚠ CUDA OOM during loading, trying CPU mode...")
            return load_checkpoint(model, optimizer, scaler, checkpoint_path)
        raise
        

def save_checkpoint(
    logger,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    best_val_loss: float,
    checkpoint_path: str,
    model_arch: str,
    scheduler
):
    date_time = datetime.datetime.now().isoformat()
    pytorch_version = str(torch.__version__)
    scheduler_state_dict = scheduler.state_dict() if scheduler else None
    checkpoint = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_arch': model_arch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler_state_dict,
        'timestamp': date_time,
        'pytorch_version': pytorch_version,  # pytorch_version为包含不被默认允许的TorchVersion全局对象
    }
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✨ Saved checkpoint (epoch {epoch}) to {checkpoint_path}; \
            Size {os.path.getsize(checkpoint_path)/1024**2:.2f} MB")
    except Exception as e:
        logger.error(f"❌ Save failed: {str(e)}")
        raise
    

# def load_checkpoint(model, optimizer, scaler, checkpoint_path):
#     best_val_loss = float('inf')
#     checkpoint = torch.load(checkpoint_path, weights_only=True)
#     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
#     # 仅继承旧优化器的学习率
#     old_optimizer_state = checkpoint['optimizer_state_dict']
#     old_lr = old_optimizer_state['param_groups'][0]['lr']
#     # 遍历新优化器的参数组，更新学习率
#     optimizer = torch.optim.Adam(model.parameters(), lr=old_lr)  # 使用旧学习率
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scaler.load_state_dict(checkpoint['scaler_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     best_val_loss = checkpoint['best_val_loss']
#     # log_path = checkpoint['log_path']
#     print(f"***Resuming training from epoch {start_epoch}...")
#     return model, optimizer, scaler, start_epoch, best_val_loss

# def save_checkpoint(model, optimizer, scaler, epoch, best_val_loss, checkpoint_path, model_arch): #TODO: 添加lr_scheduler
#     checkpoint = {
#         'epoch': epoch,
#         'best_val_loss': best_val_loss,
#         'model_arch': model_arch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scaler_state_dict': scaler.state_dict(),
#         }
#     torch.save(checkpoint, checkpoint_path)
#     print(f"✨Saved {os.path.basename(checkpoint_path)} under {os.path.dirname(checkpoint_path)}")

    
    

    
    