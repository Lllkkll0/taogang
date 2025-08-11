# taogang

本项目用于检测陶钢过滤网。

## 主要功能

- 支持对陶钢过滤网的检测与分析
- 提供示例数据与配置，方便复现和定制
- 主要使用 Jupyter Notebook 进行数据处理和结果展示

## 目录结构

- `Untitled-1.ipynb`：主 Jupyter Notebook，包含核心分析流程
- `inference.py`：推理脚本，可能用于自动化检测
- `import cv2.py`：依赖 OpenCV 的相关脚本
- `requirements.txt`：项目依赖列表
- `config/`：配置文件目录
- `sample_data/`：示例数据目录
- `scripts/`：辅助脚本目录

## 安装依赖

建议使用 Python 3.8 及以上版本。安装依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

1. 下载或克隆本仓库
2. 按需求修改 `config/` 下的配置
3. 通过 Jupyter Notebook 运行 `Untitled-1.ipynb`，或直接运行 `inference.py` 进行检测

## 贡献

欢迎 issue 和 PR！

## 许可

当前项目暂未声明具体开源协议。
