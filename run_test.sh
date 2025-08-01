#!/bin/bash

Training_Models=(
                "DCLA_UNet_final"\
                )

slb_project="test_$(date +%y%m%d)"  #TODO: 填写训练的项目名称 (必填)
# 优先级判断
if [ -z "$slb_project" ]; then  # 脚本变量未设置
    if [ $# -ge 1 ]; then       # 检查命令行参数
        slb_project="$1"
    else                        # 手动输入
        while [[ -z "$slb_project" ]]; do
            read -p "请输入训练的项目名称（必填）: " slb_project
        done
    fi
fi
echo "项目名称已设置: $slb_project"

# 启用nullglob选项防止无匹配时使用字面量
shopt -s nullglob

for model_name in "${Training_Models[@]}"; do
    echo "🔍 正在加载并训练: $model_name 模型................................."  
    # 执行命令
    python src/main.py --model_name "$model_name" \
                   --local \
                   --train_length 210 \
                   --val_length 60 \
                   --test_length 30 \
                   --epochs 10 \
                   --batch_size 1 \
                   --lr 3e-4 \
                   --wd 2e-5 \
                   --cosine_eta_min 1e-6 \
                   --cosine_T_max 100 \
                   --early_stop_patience 5 \
                   --num_workers 8 \
                   --interval 1 \
                   --slb_project $slb_project \
                   --datasets "BraTS2020" \
                   --data_root "/root/workspace/DCLA-UNet/data/BraTS2020/raw" 
done

# 关机(linux)
# echo "✅ 训练完成，正在关机"
# shutdown -h now