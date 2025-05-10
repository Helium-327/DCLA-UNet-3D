#!/bin/bash

Training_Models=(
                # "Mamba3d" \
                # "MogaNet" \
                # "UNETR" \
                # "UNETR_PP" \
                # "SlimSepUNetv2_LKDCLAv3_Attn" \
                
                # "SlimSepUNetv2_LKDCLAv3" \
                # 'DWResUNetBaseline'\
                # 'DWResUNetBaseline_SLK'\
                # 'DWResUNetBaseline_LiteMSF'\
                # 'DWResUNetBaseline_DCLA'\
                # "ResUNetBaseline_S_SLK_LiteMSF"\

                # "ResUNetBaseline_S_DCLA"\
                # 'ResUNetBaseline_S'\
                # "ResUNetBaseline_S_LiteMSF"\
                "DCLA_UNet_v1"\
                "ResUNetBaseline_S_SLK_MSF_v1"
                "ResUNetBaseline_S_SLK_DCLCA_MSF_v1"
                "ResUNetBaseline_S_SLK_DCLSA_MSF_v1"
                # "ResUNetBaseline_S_SLK"                
                # 'ResUNetBaseline_S_DCLA_SLK'\
                # 'ResUNetBaseline_S_DCLA_LiteMSF'\
                # "ResUNetBaseline_S_SLK_LiteMSF"\
                # "ResUNetBaseline_S_DCLA"\
                # 'ResUNetBaseline_S'\
                # 'ResUNetBaseline_S_SLK'\
                # 'ResUNetBaseline_S_DCLA'\
                # 'ResUNetBaseline_M'\
                # 'ResUNetBaseline_M_SLK'\
                # 'ResUNetBaseline_M_LiteMSF'\
                # 'ResUNetBaseline_M_DCLA'\
                )
 

slb_project="lr2e-4_wd1e-5_mlr1e-5_epochs200_T_max100"  #TODO: 填写训练的项目名称 (必填)
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
    python main.py --model_name "$model_name" \
                   --local \
                   --train_length 210 \
                   --val_length 60 \
                   --test_length 30 \
                   --epochs 10 \
                   --batch_size 1 \
                   --lr 1e-4 \
                   --wd 1e-5 \
                   --cosine_eta_min 1e-5 \
                   --cosine_T_max 100 \
                   --early_stop_patience 5 \
                   --num_workers 4 \
                   --interval 1 \
                   --slb_project $slb_project
done

# 禁用nullglob选项（恢复默认设置）
shopt -u nullglob

# 关机(linux)
# echo "✅ 训练完成，正在关机"
# shutdown -h now