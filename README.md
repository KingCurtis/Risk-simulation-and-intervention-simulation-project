# Risk-simulation-and-intervention-simulation-project
青少年风险模拟与干预推演项目
# 青少年风险模拟与干预推演项目

本项目包含一系列Python脚本，用于演示如何使用统计模型（NB, ZINB, BN）对多维风险/保护因子数据进行建模、模拟和干预情景推演。

## 目录结构

risk-project/
├── requirements.txt
├── README.md
└── scripts/
    ├── risk_simulation.py       (基线：负二项回归模拟)
    ├── zinb_simulation.py       (模型一：零膨胀负二项回归)
    └── bn_do_intervention.py    (模型二：贝叶斯网络与do-干预)

## 安装

1.  创建并激活一个Python虚拟环境 (推荐):
    python -m venv venv
    source venv/bin/activate  # (macOS/Linux)
    .\venv\Scripts\activate   # (Windows)

2.  安装所有依赖:
    pip install -r requirements.txt

## 运行演示

### 1. 负二项回归 (NB) 基线

此脚本演示了基本的计数回归和基于滞后项的前向滚动模拟。

python scripts/risk_simulation.py

### 2. 零膨胀负二项回归 (ZINB)

此脚本用于处理数据中存在大量“结构性零”（即某些保护因子下，违纪次数*必定*为0）的情况。

python scripts/zinb_simulation.py

### 3. 贝叶斯网络 (BN) 与 do-干预

此脚本演示了如何使用贝叶斯网络进行因果推断（do-干预），回答 "what-if" 问题。

python scripts/bn_do_intervention.py
