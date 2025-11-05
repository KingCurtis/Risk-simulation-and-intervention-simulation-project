"""
scripts/risk_simulation.py
基线模型：负二项回归 (NB) 与情景模拟
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from copy import deepcopy

# 假设的过度离散参数 (方差 = mu + ALPHA * mu^2)
ALPHA = 1.2 

def build_example_df():
    """构造6个月的示例数据"""
    data = [
        {"month": 1, "parenting_risk": 1, "peer_risk": 2, "law_edu": 0, "night_net": 18, "Y": 3},
        {"month": 2, "parenting_risk": 1, "peer_risk": 2, "law_edu": 1, "night_net": 16, "Y": 3},
        {"month": 3, "parenting_risk": 1, "peer_risk": 3, "law_edu": 0, "night_net": 20, "Y": 5},
        # 月4开始干预
        {"month": 4, "parenting_risk": 0, "peer_risk": 3, "law_edu": 2, "night_net": 14, "Y": 2},
        {"month": 5, "parenting_risk": 0, "peer_risk": 2, "law_edu": 2, "night_net": 12, "Y": 1},
        {"month": 6, "parenting_risk": 0, "peer_risk": 1, "law_edu": 2, "night_net": 10, "Y": 1},
    ]
    df = pd.DataFrame(data)
    # 创建滞后项 (上个月的Y)
    df["y_lag"] = df["Y"].shift(1).fillna(0.0)
    return df

def design_matrix(df):
    """创建设计矩阵 X"""
    X = df[["parenting_risk", "peer_risk", "law_edu", "night_net", "y_lag"]]
    X = sm.add_constant(X, has_constant="add")
    return X

def fit_nb_model(df, alpha=ALPHA):
    """拟合负二项回归模型"""
    y = df["Y"].values
    X = design_matrix(df).values
    
    # 使用 GLM 和 NegativeBinomial 族
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))
    res = model.fit()
    return res

def predict_mu(res, row_dict):
    """对单行数据预测期望值 mu"""
    x_df = pd.DataFrame([row_dict])
    X = design_matrix(x_df).values
    mu = float(res.predict(X)[0])
    return mu

def nb_sample(mu, alpha=ALPHA, rng=None):
    """从NB分布中采样"""
    rng = rng or np.random.default_rng()
    if mu <= 0: return 0
    # NB(n, p) 参数转换
    n = 1.0 / alpha
    p = n / (n + mu)
    return int(rng.negative_binomial(n, p))

def simulate_forward(res, df_base, start_month, end_month, scenario_map, deterministic=True, n_runs=1000, alpha=ALPHA, seed=42):
    """
    前向滚动模拟
    deterministic=True: 使用期望值 mu 作为 Y_t+1 的 y_lag
    deterministic=False: 使用NB采样值 作为 Y_t+1 的 y_lag (蒙特卡洛)
    """
    months = list(df_base["month"])
    rng = np.random.default_rng(seed)

    def one_run(run_rng):
        df = deepcopy(df_base)
        df.loc[:, "y_lag"] = df["Y"].shift(1).fillna(0.0)
        
        for m in months:
            if m < start_month:
                continue
            
            row = df.loc[df["month"] == m].iloc[0].to_dict()
            
            # 1. 应用干预情景
            if m in scenario_map:
                for k, v in scenario_map[m].items():
                    row[k] = v
                    df.loc[df["month"] == m, k] = v
            
            # 2. 预测 mu
            mu = predict_mu(res, row)
            
            # 3. 确定 y_t (用于下一轮的 y_lag)
            if deterministic:
                y_t = mu
            else:
                y_t = nb_sample(mu, alpha=alpha, rng=run_rng)
            
            df.loc[df["month"] == m, "Y_sim"] = y_t
            
            # 4. 更新下一个月的 y_lag
            if m < max(months):
                df.loc[df["month"] == m + 1, "y_lag"] = y_t
        
        # 填充模拟开始前的月份
        df["Y_sim"] = df["Y_sim"].fillna(df["Y"])
        return df

    if deterministic:
        return one_run(rng)
    
    # 蒙特卡洛
    sims = [one_run(rng) for _ in range(n_runs)]
    big = pd.concat(sims, ignore_index=True)
    # 计算所有模拟的平均Y_sim
    agg = big.groupby("month", as_index=False)["Y_sim"].mean()
    df_avg = df_base.merge(agg, on="month", suffixes=("", "_mc_mean"))
    df_avg["Y_sim"] = df_avg["Y_sim_mc_mean"]
    return df_avg

def main():
    print("="*30 + "\nRunning: NB Baseline Simulation\n" + "="*30)
    df = build_example_df()
    res = fit_nb_model(df, alpha=ALPHA)
    
    print("Fitted coefficients (const, parenting, peer, law, night_net, y_lag):")
    print(res.params)
    
    print("\nIn-sample expected counts (mu):")
    mu_in = res.predict(design_matrix(df).values)
    df_tmp = df.copy()
    df_tmp["mu_hat"] = mu_in
    print(df_tmp[["month", "Y", "mu_hat"]].to_string(index=False))

    # 情景1: 干预 (即原始数据中的 4-6 月)
    scn_intervene = {
        4: {"parenting_risk": 0, "law_edu": 2, "night_net": 14, "peer_risk": 3},
        5: {"parenting_risk": 0, "law_edu": 2, "night_net": 12, "peer_risk": 2},
        6: {"parenting_risk": 0, "law_edu": 2, "night_net": 10, "peer_risk": 1},
    }
    # 情景2: 无干预 (假设 4-6 月风险持续)
    scn_no = {
        4: {"parenting_risk": 1, "law_edu": 0, "night_net": 18, "peer_risk": 3},
        5: {"parenting_risk": 1, "law_edu": 0, "night_net": 19, "peer_risk": 3},
        6: {"parenting_risk": 1, "law_edu": 0, "night_net": 20, "peer_risk": 3},
    }

    # 确定性模拟
    det_intervene = simulate_forward(res, df, 4, 6, scn_intervene, deterministic=True)
    det_no = simulate_forward(res, df, 4, 6, scn_no, deterministic=True)
    
    total_intervene = det_intervene.query("month >= 4")["Y_sim"].sum()
    total_no = det_no.query("month >= 4")["Y_sim"].sum()

    print("\n--- Deterministic Simulation (Months 4-6) ---")
    print(f"  Intervention total Y_sim: {total_intervene:.2f}")
    print(f"  No-intervention total Y_sim: {total_no:.2f}")
    print(f"  Estimated reduction: {total_no - total_intervene:.2f}")

    # 蒙特卡洛模拟
    mc_intervene = simulate_forward(res, df, 4, 6, scn_intervene, deterministic=False, n_runs=2000)
    mc_no = simulate_forward(res, df, 4, 6, scn_no, deterministic=False, n_runs=2000)
    
    mc_total_intervene = mc_intervene.query("month >= 4")["Y_sim"].sum()
    mc_total_no = mc_no.query("month >= 4")["Y_sim"].sum()
    
    print("\n--- Monte Carlo (Avg) Simulation (Months 4-6) ---")
    print(f"  Intervention total Y_sim: {mc_total_intervene:.2f}")
    print(f"  No-intervention total Y_sim: {mc_total_no:.2f}")
    print(f"  Estimated reduction: {mc_total_no - mc_total_intervene:.2f}")

if __name__ == "__main__":
    main()
