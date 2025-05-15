# test_mmpose_init.py (修正版)
import os
from pathlib import Path
import mmpose  # 确保 mmpose 本身能导入

print(f"MMPose version via test script: {mmpose.__version__}")
print(f"MMPose __file__ path: {mmpose.__file__}")  # 这是 .../mmpose/__init__.py

# 获取 mmpose 包的根目录
# Path(mmpose.__file__).parent 指向 .../site-packages/mmpose/
mmpose_package_root_dir = Path(mmpose.__file__).resolve().parent
print(f"MMPose package root directory: {mmpose_package_root_dir}")

# 期望的配置文件相对路径 (相对于 mmpose 包的根目录)
config_subpath = 'configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
expected_config_path = mmpose_package_root_dir / config_subpath

print(f"Expected config file path (inside mmpose package): {expected_config_path}")
print(f"Does the config file exist at this location? {expected_config_path.exists()}")

if not expected_config_path.exists():
    print(f"ERROR: The HRNet config file is MISSING from the mmpose package directory ({mmpose_package_root_dir})!")
    print(f"Attempting to list contents of the '{mmpose_package_root_dir / 'configs'}' directory (if it exists):")

    mmpose_configs_dir = mmpose_package_root_dir / 'configs'
    if mmpose_configs_dir.exists() and mmpose_configs_dir.is_dir():
        print(f"Contents of {mmpose_configs_dir}:")
        # 递归地列出一些内容，帮助定位
        for root, dirs, files in os.walk(mmpose_configs_dir, topdown=True):
            # 限制深度和数量以避免过多输出
            if root.count(os.sep) - str(mmpose_configs_dir).count(os.sep) < 3:  # 限制到 configs 下两级子目录
                print(f"  In {root}:")
                for name in dirs[:3]:  # 最多显示3个子目录
                    print(f"    DIR: {name}")
                for name in files[:5]:  # 最多显示5个文件
                    print(f"    FILE: {name}")
            # 停止进一步深入，如果已经列出足够信息
            if root.count(os.sep) - str(mmpose_configs_dir).count(os.sep) >= 1 and \
                    (len(os.listdir(root)) == 0 or (len(dirs) == 0 and len(files) == 0)):
                break  # 如果是空目录或者已经显示了子目录内容则不再深入该分支

    else:
        print(f"The directory '{mmpose_configs_dir}' does NOT exist within the mmpose package.")
        print(f"Contents of the mmpose package root '{mmpose_package_root_dir}':")
        for item in os.listdir(mmpose_package_root_dir)[:10]:  # 列出mmpose包根目录的前10项
            print(f"  - {item}")

else:
    print("HRNet config file FOUND inside the mmpose package directory. This is good!")
    print("The issue is likely how MMPoseInferencer uses the alias to find this path.")