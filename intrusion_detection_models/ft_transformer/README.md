# FT-Transformer 使用说明

建议先阅读上级 README 的论文依据、数据协议与指标定义。此文件夹可独立使用，内含所需原始数据。
默认十分类，无 SMOTE、类别加权或少样本抽样。

在 PowerShell 中进入本文件夹后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe train.py --output runs/ft_seed42 --seed 42
.venv/Scripts/python.exe evaluate.py --run runs/ft_seed42 --output evaluations/ft_seed42
```

训练自动检测 GPU；没有可用 CUDA 时使用 CPU。强制 CPU 可加 `--device cpu`。
默认 CPU 线程数 4，GPU 显存不足时降低 `--batch-size 128`；这些调整应记录在实验中。
数据路径相对脚本定位，结果路径相对当前工作目录。路径有空格时加双引号。

可选参数：

```text
--train-csv PATH       默认 data/UNSW_NB15_training-set.csv
--seed 42              划分、初始化、抽样共用的随机种子
--epochs 200           最大训练轮数
--patience 16          连续 17 次无提升时停止
--batch-size 512
--d-token 192          必须能被头数整除
--layers 3
--heads 8
--learning-rate 0.0001
--weight-decay 0.00001
--device auto          auto / cpu / cuda
--threads 4
--smoke                每类最多 32 条训练、16 条验证；仅用于流程检查
```

轻量检查（与正式结果分开）：

```powershell
.venv/Scripts/python.exe train.py --smoke --epochs 2 --d-token 32 --layers 1 --heads 4 --batch-size 64 --output runs/smoke
.venv/Scripts/python.exe evaluate.py --run runs/smoke --limit 1000 --output evaluations/smoke
.venv/Scripts/python.exe audit_data.py --output data_audit.json
```

正式评估不要带 --limit。最佳权重是 model.pt，预处理器是 preprocessor.joblib，必须成对保留。
evaluate.py 还支持 --test-csv、--batch-size、--device 和 --threads。
结果目录已有文件时脚本拒绝覆盖；使用新目录即可。
只加载自己训练的可信模型与 joblib 文件，运行时保持训练和评估依赖版本一致。
PyTorch 设置随机种子和确定性 cuDNN；跨设备/库版本不保证逐位相同。

默认参数是可运行的研究起点。完整训练耗时取决于设备，交付验证不等于已经完成默认配置的收敛训练。
