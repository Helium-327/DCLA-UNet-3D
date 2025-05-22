#!/bin/bash

Training_Models=(
                "DCLA_UNet_v2_4"\
                # "ResUNetBaseline_S_SLKv2_MSF_v3"\
                # "ResUNetBaseline_S_SLKv2_v3"\
                # "ResUNetBaseline_S_DCLA_SLKv2_v3"\
                # "ResUNetBaseline_S_MSF_v3"\
                # "ResUNetBaseline_S_DCLA_MSF_v3"\
                )


slb_project="0510_DCLA_UNet_v2"  #TODO: 填写训练的项目名称 (必填)
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
                   --slb \
                   --lr 1e-4 \
                   --wd 1e-5 \
                   --cosine_eta_min 1e-6 \
                   --epochs 100 \
                   --cosine_T_max 100 \
                   --early_stop_patience 100 \
                   --batch_size 2 \
                   --num_workers 8 \
                   --interval 1 \
                   --slb_project "$slb_project"                    
done

# 禁用nullglob选项（恢复默认设置）
shopt -u nullglob

shutdown -h now
# 关机(linux)
# echo "✅ 训练完成，正在关机"

