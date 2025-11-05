"""
scripts/zinb_simulation.py
模型一：零膨胀负二项回归 (ZINB)
适用于 Y (违纪次数) 中有大量0的情况
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from copy import deepcopy

# 注意：ZINB的模拟比NB更复杂，ALPHA仅用于NB部分
ALPHA = 1.0 

def build_zinb_example_df(n_samples=50, seed=42):
    """
    构造一个更大的、包含“结构性零”的模拟数据集
    """
    rng = np.random.default_rng(seed)
    
    # 1. 创建协变量
    df = pd.DataFrame({
        "id": range(n_samples),
        # 保护因子: 权威教养 (0=否, 1=是)
        "auth_parenting": rng.choice([0, 1], n_samples, p=[0.6, 0.4]),
        # 风险因子: 不良同伴 (0-3)
        "peer_risk": rng.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        # 风险因子: 冲动控制 (0-5, 越高越差)
        "impulse_control": rng.randint(0, 6, n_samples),
    })
    
    # 2. 创建 "零膨胀" 逻辑 (Logit)
    # 假设 "权威教养" 能极大增加 Y=0 的概率
    logit_z_eta = -1.5 + 3.0 * df["auth_parenting"] - 0.5 * df["peer_risk"]
    p_zero = 1 / (1 + np.exp(-logit_z_eta))
    is_structural_zero = rng.binomial(1, p_zero, n_samples)
    
    # 3. 创建 "计数" 逻辑 (NB)
    # 假设 "不良同伴" 和 "冲动控制" 增加计数
    log_mu_eta = -1.0 + 0.5 * df["peer_risk"] + 0.3 * df["impulse_control"] - 0.8 * df["auth_parenting"]
    mu = np.exp(log_mu_eta)
    
    # 从NB分布中采样
    n = 1.0 / ALPHA
    p_nb = n / (n + mu)
    counts = rng.negative_binomial(n, p_nb, n_samples)
    
    # 4. 组合
    # 如果是 "结构性零"，则Y=0；否则，Y=NB采样值
    df["Y"] = np.where(is_structural_zero == 1, 0, counts)
    
    print(f"Generated {n_samples} samples. Zero proportion: { (df['Y'] == 0).mean():.2% }")
    return df

def fit_zinb_model(df, alpha=ALPHA):
    """拟合ZINB模型"""
    y = df["Y"]
    
    # 1. 计数模型的设计矩阵 (X_count)
    X_count = df[["auth_parenting", "peer_risk", "impulse_control"]]
    X_count = sm.add_constant(X_count, prepend=True, has_constant="add")
    
    # 2. 零膨胀模型的设计矩阵 (X_inflation)
    # 我们用 'auth_parenting' 和 'peer_risk' 来预测 "是否为零"
    X_inflation = df[["auth_parenting", "peer_risk"]]
    X_inflation = sm.add_constant(X_inflation, prepend=True, has_constant="add")
    
    # 拟合模型
    # 注意: exog_infl 是零膨胀部分, exog 是计数部分
    # ZINB 在 statsmodels 中仍在完善，这里使用 ZeroInflatedNegativeBinomialP
    model = sm.ZeroInflatedNegativeBinomialP(y, exog=X_count, exog_infl=X_inflation, p=2)
    
    # 使用 'powell' 或 'bfgs' 等优化器，ZINB有时拟合较难
    res = model.fit(method='powell', maxiter=5000)
    return res

def simulate_zinb_scenario(res, df_base, scenario_map):
    """
    对整个数据集应用情景，并返回总Y的期望值
    scenario_map: e.g., {"auth_parenting": 1} (应用全员干预)
    """
    df_sim = df_base.copy()
    
    # 1. 应用干预
    for col, val in scenario_map.items():
        df_sim[col] = val
        
    # 2. 准备预测矩阵
    X_count = df_sim[["auth_parenting", "peer_risk", "impulse_control"]]
    X_count = sm.add_constant(X_count, prepend=True, has_constant="add")
    
    X_inflation = df_sim[["auth_parenting", "peer_risk"]]
    X_inflation = sm.add_constant(X_inflation, prepend=True, has_constant="add")
    
    # 3. 预测
    # predict() 返回 E[Y|X] = (1 - p_zero) * mu
    expected_Y = res.predict({'exog': X_count, 'exog_infl': X_inflation})
    
    return expected_Y.sum()
    

def main():
    print("\n" + "="*30 + "\nRunning: ZINB Simulation\n" + "="*30)
    
    df = build_zinb_example_df(n_samples=500, seed=42)
    res = fit_zinb_model(df, alpha=ALPHA)
    
    print("\n--- ZINB Model Fit Summary ---")
    print(res.summary())

    # --- 情景推演 ---
    
    # 基线: 不做任何改变
    baseline_total_Y = simulate_zinb_scenario(res, df, {})
    
    # 情景1: 推广 "权威教养" (auth_parenting = 1)
    scenario_1_map = {"auth_parenting": 1}
    scenario_1_total_Y = simulate_zinb_scenario(res, df, scenario_1_map)
    
    # 情景2: 降低 "不良同伴" 风险 (假设 peer_risk 降低 1, 最小为0)
    df_s2 = df.copy()
    df_s2["peer_risk"] = (df_s2["peer_risk"] - 1).clip(lower=0)
    scenario_2_total_Y = simulate_zinb_scenario(res, df_s2, {}) # 用修改后的df
    
    # 情景3: 组合干预 (教养 + 同伴)
    scenario_3_total_Y = simulate_zinb_scenario(res, df_s2, scenario_1_map)


    print("\n--- ZINB Scenario Simulation (Total Expected Y) ---")
    print(f"  Baseline (N=500): {baseline_total_Y:.2f}")
    print(f"  Scenario 1 (Auth Parenting): {scenario_1_total_Y:.2f} (Change: {scenario_1_total_Y - baseline_total_Y:.2f})")
    print(f"  Scenario 2 (Peer Risk -1): {scenario_2_total_Y:.2f} (Change: {scenario_2_total_Y - baseline_total_Y:.2f})")
    print(f"  Scenario 3 (Combined): {scenario_3_total_Y:.2f} (Change: {scenario_3_total_Y - baseline_total_Y:.2f})")

if __name__ == "__main__":
    main()
