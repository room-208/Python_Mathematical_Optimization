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

    def build(self):
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

    def solve(self):
        time_start = time.time()
        status = self.problem.solve()
        time_end = time.time()
        print(f"Status:{pulp.LpStatus[status]}")
        print(f"Objective:{pulp.value(self.problem.objective)}")
        print(f"Time:{time_end-time_start}")


if __name__ == "__main__":
    solver = CouponSolver()
    solver.read_input_data()
    solver.build()
    solver.solve()
