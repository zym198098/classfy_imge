#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/zym/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/zym/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/zym/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/zym/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# conda启动虚拟环境
conda activate d2l

# 在指定的虚拟环境中启动特定脚本
# cd /home/zym/pytorch_proj/classfy_imge
python train.py
exit 0
