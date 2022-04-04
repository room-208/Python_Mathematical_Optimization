from cmath import cos, pi
from turtle import pu
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

DIRNAME = "./4.coupon"


class CouponSolver:
    def __init__(self) -> None:
        self.customer_df = None
        self.visit_probability_df = None
        self.problem = None
        self.xsm = None
        self.y = None

    def read_input_data(self):
        self.customer_df = pd.read_csv(os.path.join(DIRNAME, "customers.csv"))
        self.visit_probability_df = pd.read_csv(os.path.join(DIRNAME, "visit_probability.csv"))

    def check_input_data(self):
        print(self.customer_df)
        self.customer_df["age_cat"].hist()
        plt.show()
        self.customer_df["freq_cat"].hist()
        plt.show()
        customer_pivot_df = pd.pivot_table(data=self.customer_df, values='customer_id', columns='freq_cat', index='age_cat', aggfunc='count')
        customer_pivot_df = customer_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
        sns.heatmap(customer_pivot_df, annot=True, fmt='d', cmap='Blues')
        plt.show()
        print(self.visit_probability_df)
        ax = {}
        fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
        for i, ptn in enumerate(["prob_dm1", "prob_dm2", "prob_dm3"]):
            prob_pivot_df = pd.pivot_table(data=self.visit_probability_df, values=ptn, columns='freq_cat', index='age_cat')
            prob_pivot_df = prob_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
            sns.heatmap(prob_pivot_df, annot=True, fmt='.0%', cmap='Blues', ax=ax[i])
            ax[i].set_title(f'Visit Probability of {ptn}')
        plt.show()

    def build1(self):
        # 最適化問題
        self.problem = pulp.LpProblem(name="DiscountCouponProblem", sense=pulp.LpMaximize)

        I = self.customer_df["customer_id"].to_list()
        M = [1, 2, 3]

        # 変数
        xim = {}
        for i in I:
            for m in M:
                xim[i, m] = pulp.LpVariable(name=f"xim({i},{m})", cat="Binary")

        # 目的関数
        keys = ["age_cat", "freq_cat"]
        merge_df = pd.merge(self.customer_df, self.visit_probability_df, on=keys)
        Pim = {}
        for _, row in merge_df.iterrows():
            Pim[(row["customer_id"], 1)] = row["prob_dm1"]
            Pim[(row["customer_id"], 2)] = row["prob_dm2"]
            Pim[(row["customer_id"], 3)] = row["prob_dm3"]
        self.problem += pulp.lpSum((Pim[i, m]-Pim[i, 1])*xim[i, m] for i in I for m in [2, 3])

        # 制約条件
        for i in I:
            self.problem += pulp.lpSum(xim[i, m] for m in M) == 1

        C = {1: 0, 2: 1000, 3: 2000}
        self.problem += pulp.lpSum(Pim[i, m]*C[m]*xim[i, m] for i in I for m in [2, 3]) <= 1000000

        Ns = merge_df.groupby("segment_id")["customer_id"].count().to_dict()
        Si = dict(zip(merge_df["customer_id"], merge_df["segment_id"]))
        for ns in Ns.keys():
            for m in M:
                self.problem += pulp.lpSum(xim[i, m] for i in I if Si[i] == ns) >= 0.1 * Ns[ns]

    def build2(self):
        # 最適化問題
        self.problem = pulp.LpProblem(name="DiscountCouponProblem", sense=pulp.LpMaximize)

        I = self.customer_df["customer_id"].to_list()
        S = self.visit_probability_df["segment_id"].to_list()
        C = {1: 0, 2: 1000, 3: 2000}
        M = [1, 2, 3]

        keys = ["age_cat", "freq_cat"]
        merge_df = pd.merge(self.customer_df, self.visit_probability_df, on=keys)
        Ns = merge_df.groupby("segment_id")["customer_id"].count().to_dict()
        Si = dict(zip(merge_df["customer_id"], merge_df["segment_id"]))

        # 変数
        self.xsm = {}
        for s in S:
            for m in M:
                self.xsm[s, m] = pulp.LpVariable(name=f"xim({s},{m})", lowBound=0, upBound=1, cat="Continuous")

        for s in S:
            self.problem += pulp.lpSum(self.xsm[s, m] for m in M) == 1

        Psm = {}
        for _, row in self.visit_probability_df.iterrows():
            Psm[row["segment_id"], 1] = row["prob_dm1"]
            Psm[row["segment_id"], 2] = row["prob_dm2"]
            Psm[row["segment_id"], 3] = row["prob_dm3"]

        self.problem += pulp.lpSum(Ns[s] * self.xsm[s, m]*(Psm[s, m]-Psm[s, 1]) for s in S for m in [2, 3])

        self.problem += pulp.lpSum(Ns[s] * self.xsm[s, m]*Psm[s, m]*C[m] for s in S for m in [2, 3]) <= 1000000

        for s in S:
            for m in M:
                self.problem += self.xsm[s, m] >= 0.1

    def build3(self):
        # 最適化問題
        self.problem = pulp.LpProblem(name="DiscountCouponProblem", sense=pulp.LpMaximize)

        I = self.customer_df["customer_id"].to_list()
        S = self.visit_probability_df["segment_id"].to_list()
        C = {1: 0, 2: 1000, 3: 2000}
        M = [1, 2, 3]

        keys = ["age_cat", "freq_cat"]
        merge_df = pd.merge(self.customer_df, self.visit_probability_df, on=keys)
        Ns = merge_df.groupby("segment_id")["customer_id"].count().to_dict()
        Si = dict(zip(merge_df["customer_id"], merge_df["segment_id"]))

        # 変数
        self.xsm = {}
        for s in S:
            for m in M:
                self.xsm[s, m] = pulp.LpVariable(name=f"xim({s},{m})", lowBound=0, upBound=1, cat="Continuous")

        self.y = pulp.LpVariable(name=f"y", lowBound=0, upBound=1, cat="Continuous")

        for s in S:
            self.problem += pulp.lpSum(self.xsm[s, m] for m in M) == 1

        Psm = {}
        for _, row in self.visit_probability_df.iterrows():
            Psm[row["segment_id"], 1] = row["prob_dm1"]
            Psm[row["segment_id"], 2] = row["prob_dm2"]
            Psm[row["segment_id"], 3] = row["prob_dm3"]

        self.problem += self.y

        self.problem += pulp.lpSum(Ns[s] * self.xsm[s, m]*Psm[s, m]*C[m] for s in S for m in [2, 3]) <= 1000000

        for s in S:
            for m in M:
                self.problem += self.xsm[s, m] >= self.y

    def build4(self, cost):
        # 最適化問題
        self.problem = pulp.LpProblem(name="DiscountCouponProblem", sense=pulp.LpMaximize)

        I = self.customer_df["customer_id"].to_list()
        S = self.visit_probability_df["segment_id"].to_list()
        C = {1: 0, 2: 1000, 3: 2000}
        M = [1, 2, 3]

        keys = ["age_cat", "freq_cat"]
        merge_df = pd.merge(self.customer_df, self.visit_probability_df, on=keys)
        Ns = merge_df.groupby("segment_id")["customer_id"].count().to_dict()
        Si = dict(zip(merge_df["customer_id"], merge_df["segment_id"]))

        # 変数
        self.xsm = {}
        for s in S:
            for m in M:
                self.xsm[s, m] = pulp.LpVariable(name=f"xim({s},{m})", lowBound=0, upBound=1, cat="Continuous")

        for s in S:
            self.problem += pulp.lpSum(self.xsm[s, m] for m in M) == 1

        Psm = {}
        for _, row in self.visit_probability_df.iterrows():
            Psm[row["segment_id"], 1] = row["prob_dm1"]
            Psm[row["segment_id"], 2] = row["prob_dm2"]
            Psm[row["segment_id"], 3] = row["prob_dm3"]

        self.problem += pulp.lpSum(Ns[s] * self.xsm[s, m]*(Psm[s, m]-Psm[s, 1]) for s in S for m in [2, 3])

        self.problem += pulp.lpSum(Ns[s] * self.xsm[s, m]*Psm[s, m]*C[m] for s in S for m in [2, 3]) <= cost

        for s in S:
            for m in M:
                self.problem += self.xsm[s, m] >= 0.1

    def solve(self):
        time_start = time.time()
        status = self.problem.solve()
        time_end = time.time()
        cpa = cost/pulp.value(self.problem.objective)
        inc_action = pulp.value(self.problem.objective)
        return status, cpa, inc_action

    def check_output_data(self):
        S = self.visit_probability_df["segment_id"].to_list()
        M = [1, 2, 3]

        pivot_df = pd.DataFrame([[self.xsm[s, m].value() for m in M] for s in S], columns=["dm1", "dm2", "dm3"])
        pivot_df = pd.concat([pivot_df, self.visit_probability_df], axis=1)

        ax = {}
        fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
        for i, ptn in enumerate(["dm1", "dm2", "dm3"]):
            prob_pivot_df = pd.pivot_table(data=pivot_df, values=ptn, columns='freq_cat', index='age_cat')
            prob_pivot_df = prob_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
            sns.heatmap(prob_pivot_df, annot=True, fmt='.0%', cmap='Blues', ax=ax[i])
            ax[i].set_title(f'Visit Probability of {ptn}')
        plt.show()

        keys = ["age_cat", "freq_cat"]
        merge_df = pd.merge(self.customer_df, self.visit_probability_df, on=keys)
        Ns = merge_df.groupby("segment_id")["customer_id"].count().to_dict()

        pivot_df["num"] = pivot_df["segment_id"].map(Ns)
        pivot_df["Nsdm1"] = pivot_df["num"] * pivot_df["dm1"]
        pivot_df["Nsdm2"] = pivot_df["num"] * pivot_df["dm2"]
        pivot_df["Nsdm3"] = pivot_df["num"] * pivot_df["dm3"]
        print(pivot_df)

        ax = {}
        fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
        for i, ptn in enumerate(["Nsdm1", "Nsdm2", "Nsdm3"]):
            prob_pivot_df = pd.pivot_table(data=pivot_df, values=ptn, columns='freq_cat', index='age_cat')
            prob_pivot_df = prob_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
            sns.heatmap(prob_pivot_df, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=800, ax=ax[i])
            ax[i].set_title(f'Visit Probability of {ptn}')
        plt.show()


if __name__ == "__main__":
    solver = CouponSolver()
    solver.read_input_data()

    status_list = []
    cost_list = []
    inc_action_list = []
    cpa_list = []
    for cost in range(761850, 3000000, 100000):
        solver.build4(cost)
        status, cpa, inc_action = solver.solve()
        cost_list.append(cost)
        status_list.append(status)
        cpa_list.append(cpa)
        inc_action_list.append(inc_action)

    for cost, status, cpa, inc_action in zip(cost_list, status_list, cpa_list, inc_action_list):
        print(f"Status:{pulp.LpStatus[status]} Cost:{cost} Inc_action{inc_action} CPA{cpa}")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(cost_list, inc_action_list)
    ax2.scatter(cost_list, cpa_list)
    plt.show()
