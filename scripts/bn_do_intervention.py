# 这是贝叶斯网络（BN）模型。它最适合回答 "What-if" 和 "Do-干预" 问题。 例如：“如果我们强制将教养方式变为‘民主型’，违纪的概率会如何变化？”

# 注意：BN通常需要将连续变量（如“上网时间”）离散化（如“低/中/高”）。

"""
scripts/bn_do_intervention.py
模型二：贝叶斯网络 (BN) 与 do-干预
演示 "what-if" 情景
"""
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_bn_model():
    """
    定义一个简化的贝叶斯网络结构
    结构:
    1. 家庭教养 (Parenting) -> 个体心理 (Psychology)
    2. 家庭教养 (Parenting) -> 同伴关系 (Peer)
    3. 同伴关系 (Peer) -> 在校表现 (Discipline)
    4. 个体心理 (Psychology) -> 在校表现 (Discipline)
    5. 法治教育 (LawEdu) -> 在校表现 (Discipline)
    """
    
    # 1. 定义网络结构
    model = BayesianNetwork([
        ('Parenting', 'Psychology'),
        ('Parenting', 'Peer'),
        ('Peer', 'Discipline'),
        ('Psychology', 'Discipline'),
        ('LawEdu', 'Discipline')
    ])
    
    # 2. 定义变量状态 (0=保护/低风险, 1=风险/高风险)
    
    # CPD: Parenting (0=民主/权威, 1=忽视/暴力)
    cpd_p = TabularCPD(variable='Parenting', variable_card=2, values=[[0.6], [0.4]])
    
    # CPD: LawEdu (0=充足, 1=缺乏)
    cpd_l = TabularCPD(variable='LawEdu', variable_card=2, values=[[0.7], [0.3]])

    # CPD: Psychology (受 Parenting 影响)
    # P(Psych | Parenting)
    cpd_psy = TabularCPD(variable='Psychology', variable_card=2,
                         values=[[0.8, 0.3],  # Psy=0 (健康)
                                 [0.2, 0.7]], # Psy=1 (冲动)
                         evidence=['Parenting'], evidence_card=[2])
                         
    # CPD: Peer (受 Parenting 影响)
    # P(Peer | Parenting)
    cpd_peer = TabularCPD(variable='Peer', variable_card=2,
                          values=[[0.7, 0.4],  # Peer=0 (积极)
                                  [0.3, 0.6]], # Peer=1 (不良)
                          evidence=['Parenting'], evidence_card=[2])

    # CPD: Discipline (在校表现, 0=良好, 1=违纪)
    # 受 Peer, Psychology, LawEdu 影响
    cpd_d = TabularCPD(variable='Discipline', variable_card=2,
                       values=[
                           # P(Discipline=0 | Peer, Psy, LawEdu) - 良好
                           [0.99, 0.8, 0.7, 0.4, 0.9, 0.6, 0.5, 0.2],
                           # P(Discipline=1 | Peer, Psy, LawEdu) - 违纪
                           [0.01, 0.2, 0.3, 0.6, 0.1, 0.4, 0.5, 0.8]
                       ],
                       evidence=['Peer', 'Psychology', 'LawEdu'],
                       evidence_card=[2, 2, 2])
                       
    # 3. 将CPD添加到模型
    model.add_cpds(cpd_p, cpd_l, cpd_psy, cpd_peer, cpd_d)
    
    # 4. 检查模型
    if not model.check_model():
        raise ValueError("模型定义错误")
        
    print("贝叶斯网络构建成功!")
    return model

def main():
    print("\n" + "="*30 + "\nRunning: Bayesian Network (BN) Simulation\n" + "="*30)
    
    model = build_bn_model()
    inference = VariableElimination(model)

    # --- 1. 基线查询 (Baseline) ---
    # 在没有任何干预的情况下，"在校违纪" (Discipline=1) 的概率是多少?
    baseline_prob = inference.query(
        variables=['Discipline'], 
        evidence={}
    )
    print("\n--- 1. 基线 (Baseline) ---")
    print(baseline_prob)
    baseline_risk = baseline_prob.values[1] # P(Discipline=1)
    print(f"基线违纪风险 (Discipline=1): {baseline_risk:.2%}")


    # --- 2. "Do-干预" 查询 (Intervention) ---
    # 情景A: 如果我们 "do" (强制干预) "Parenting" = 0 (民主教养)
    # 这会切断所有指向 Parenting 的边 (虽然这里没有)
    # 并将其值固定为0
    
    # pgmpy 使用 do_query 来实现 Pearl's do-calculus
    # 注意: pgmpy 的 do() 实现有时不稳定, 我们用 'evidence' 模拟固定值干预
    # 对于这个无环图, 'evidence' (条件概率) 和 'do' (干预) 在此例中结果一致
    
    print("\n--- 2. 干预情景推演 ---")
    
    # 情景A: 干预教养方式 (Parenting=0)
    prob_A = inference.query(
        variables=['Discipline'],
        evidence={'Parenting': 0} # 0=民主/权威
    )
    risk_A = prob_A.values[1]
    print(f"情景 A (干预教养): 违纪风险 {risk_A:.2%}")

    # 情景B: 干预法治教育 (LawEdu=0)
    prob_B = inference.query(
        variables=['Discipline'],
        evidence={'LawEdu': 0} # 0=充足
    )
    risk_B = prob_B.values[1]
    print(f"情景 B (干预法教): 违纪风险 {risk_B:.2%}")

    # 情景C: 组合拳 (Parenting=0 且 LawEdu=0)
    prob_C = inference.query(
        variables=['Discipline'],
        evidence={'Parenting': 0, 'LawEdu': 0}
    )
    risk_C = prob_C.values[1]
    print(f"情景 C (组合干预): 违纪风险 {risk_C:.2%}")


    print("\n--- 总结: 风险降低百分比 (vs 基线) ---")
    print(f"干预教养 (A): { (baseline_risk - risk_A) / baseline_risk:.2% }")
    print(f"干预法教 (B): { (baseline_risk - risk_B) / baseline_risk:.2% }")
    print(f"组合干预 (C): { (baseline_risk - risk_C) / baseline_risk:.2% }")

if __name__ == "__main__":
    main()
