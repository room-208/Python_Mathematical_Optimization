from random import sample
import cvxopt
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

DIRNAME = "./7.recommendation"


class RecommendProblem():
    def __init__(self):
        self.__log_df = None
        self.__tar_df = None
        self.__RF2Prob = None
        self.__R = None
        self.__F = None
        self.__RF2Idx = None
        self.__G = None
        self.__h = None
        self.__P = None
        self.__q = None
        self.__sol = None
        self.__rf_df = None
        self.__rcen_df = None
        self.__freq_df = None

    def read_data(self):
        self.__log_df = pd.read_csv(os.path.join(DIRNAME, "access_log.csv"), parse_dates=["date"])
        # print(self.__log_df)
        # print(self.__log_df["user_id"].value_counts().describe())
        # print(self.__log_df["item_id"].value_counts().describe())
        # print(self.__log_df["date"].value_counts())
        # self.__log_df.hist(bins=4000)
        # plt.show()

    def checK_assumption(self):
        start_date = dt.datetime(2015, 7, 1)
        end_date = dt.datetime(2015, 7, 7)
        target_date = dt.datetime(2015, 7, 8)

        x_df = self.__log_df[(start_date <= self.__log_df["date"]) & (self.__log_df["date"] <= end_date)]
        y_df = self.__log_df[target_date <= self.__log_df["date"]]

        U2I2Rcens = {}
        for row in x_df.itertuples():
            rcen = (target_date-row.date).days
            U2I2Rcens.setdefault(row.user_id, {})
            U2I2Rcens[row.user_id].setdefault(row.item_id, [])
            U2I2Rcens[row.user_id][row.item_id].append(rcen)

        Rowsl = []
        for user_id, I2Rcens in U2I2Rcens.items():
            for item_id, Rcens in I2Rcens.items():
                freq = len(Rcens)
                rcen = min(Rcens)
                Rowsl.append((user_id, item_id, rcen, freq))

        UI2RF_df = pd.DataFrame(Rowsl, columns=["user_id", "item_id", "rcen", "freq"])
        print(UI2RF_df)

        y_df = y_df.drop_duplicates()
        y_df["pv_flag"] = 1
        print(y_df)

        UI2RFP_df = pd.merge(UI2RF_df, y_df[["user_id", "item_id", "pv_flag"]], how="left", on=["user_id", "item_id"])
        UI2RFP_df = UI2RFP_df.fillna(0)
        print(UI2RFP_df)
        print(sorted(UI2RFP_df["rcen"].unique()))
        print(sorted(UI2RFP_df["freq"].unique()))

        tar_df = UI2RFP_df[UI2RFP_df["freq"] <= 7]
        print(tar_df)
        print(tar_df["pv_flag"].sum())

        rcen_df = pd.crosstab(index=tar_df["rcen"], columns=tar_df["pv_flag"])
        rcen_df = rcen_df.rename(columns={0: "neg", 1: "pos"})
        print(rcen_df)

        rcen_df["N"] = rcen_df["neg"]+rcen_df["pos"]
        rcen_df["prob"] = rcen_df["pos"]/rcen_df["N"]
        rcen_df[["prob"]].plot.bar()
        print(rcen_df)
        plt.show()

        freq_df = pd.crosstab(index=tar_df["freq"], columns=tar_df["pv_flag"])
        freq_df = freq_df.rename(columns={0: "neg", 1: "pos"})
        print(freq_df)

        freq_df["N"] = freq_df["neg"]+freq_df["pos"]
        freq_df["prob"] = freq_df["pos"]/freq_df["N"]
        freq_df[["prob"]].plot.bar()
        print(freq_df)
        plt.show()

        self.__tar_df = tar_df
        self.__rcen_df = rcen_df
        self.__freq_df = freq_df

    def cal_rf_df(self):
        RF2N = {}
        RF2PV = {}
        for row in self.__tar_df.itertuples():
            key = (row.rcen, row.freq)
            RF2N.setdefault(key, 0)
            RF2PV.setdefault(key, 0)
            RF2N[key] += 1
            if row.pv_flag == 1:
                RF2PV[key] += 1

        RF2Prob = {}
        for rf, N in RF2N.items():
            RF2Prob[rf] = RF2PV[rf]/N

        Rows3 = []
        for rf, N in sorted(RF2N.items()):
            pv = RF2PV[rf]
            prob = RF2Prob[rf]
            row = (rf[0], rf[1], N, pv, prob)
            Rows3.append(row)
        rf_df = pd.DataFrame(Rows3, columns=["rcen", "freq", "N", "pv", "prob"])
        print(rf_df)
        print(rf_df.pivot_table(index="rcen", columns="freq", values="prob"))

        Freq = rf_df["freq"].unique().tolist()
        Rcen = rf_df["rcen"].unique().tolist()
        Z = [rf_df[(rf_df["freq"] == freq) & (rf_df["rcen"] == rcen)]["prob"].iloc[0] for freq in Freq for rcen in Rcen]
        Z = np.array(Z).reshape((len(Rcen), len(Freq)))
        X, Y = np.meshgrid(Rcen, Freq)
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            projection="3d",
            xlabel="rcen",
            ylabel="freq",
            zlabel="prob"
        )
        ax.plot_wireframe(X, Y, Z)
        plt.show()

        self.__RF2Prob = RF2Prob
        self.__rf_df = rf_df

    def build(self):
        R = sorted(self.__tar_df["rcen"].unique().tolist())
        F = sorted(self.__tar_df["freq"].unique().tolist())
        print(R)
        print(F)

        Idx = []
        RF2Idx = {}
        idx = 0
        for r in R:
            for f in F:
                Idx.append(idx)
                RF2Idx[r, f] = idx
                idx += 1
        print(Idx)
        print(RF2Idx)

        G_list = []
        h_list = []
        var_vec = [0.0]*len(Idx)
        print(var_vec)

        for r in R:
            for f in F:
                idx = RF2Idx[r, f]
                G_row = var_vec[:]
                G_row[idx] = -1
                G_list.append(G_row)
                h_list.append(0)

        for r in R:
            for f in F:
                idx = RF2Idx[r, f]
                G_row = var_vec[:]
                G_row[idx] = 1
                G_list.append(G_row)
                h_list.append(1)

        for r in R[:-1]:
            for f in F:
                idx1 = RF2Idx[r, f]
                idx2 = RF2Idx[r+1, f]
                G_row = var_vec[:]
                G_row[idx1] = -1
                G_row[idx2] = 1
                G_list.append(G_row)
                h_list.append(0)

        for r in R:
            for f in F[:-1]:
                idx1 = RF2Idx[r, f]
                idx2 = RF2Idx[r, f+1]
                G_row = var_vec[:]
                G_row[idx1] = 1
                G_row[idx2] = -1
                G_list.append(G_row)
                h_list.append(0)

        P_list = []
        q_list = []

        for r in R:
            for f in F:
                idx = RF2Idx[r, f]
                N = RF2Idx[r, f]
                prob = self.__RF2Prob[r, f]
                P_row = var_vec[:]
                P_row[idx] = 2*N
                P_list.append(P_row)
                q_list.append(-2*N*prob)

        for r in R[:-2]:
            for f in F:
                idx1 = RF2Idx[r, f]
                idx2 = RF2Idx[r+1, f]
                idx3 = RF2Idx[r+2, f]
                G_row = var_vec[:]
                G_row[idx1] = -1
                G_row[idx2] = 2
                G_row[idx3] = -1
                G_list.append(G_row)
                h_list.append(0)

        for r in R:
            for f in F[:-2]:
                idx1 = RF2Idx[r, f]
                idx2 = RF2Idx[r, f+1]
                idx3 = RF2Idx[r, f+2]
                G_row = var_vec[:]
                G_row[idx1] = 1
                G_row[idx2] = -2
                G_row[idx3] = 1
                G_list.append(G_row)
                h_list.append(0)

        self.__R = R
        self.__F = F
        self.__RF2Idx = RF2Idx
        self.__G = cvxopt.matrix(np.array(G_list), tc="d")
        self.__h = cvxopt.matrix(np.array(h_list), tc="d")
        self.__P = cvxopt.matrix(np.array(P_list), tc="d")
        self.__q = cvxopt.matrix(np.array(q_list), tc="d")

    def solve(self):
        self.__sol = cvxopt.solvers.qp(self.__P, self.__q, self.__G, self.__h)
        status = self.__sol["status"]
        print(status)

    def check_solution(self):
        RF2PredProb = {}
        X = self.__sol["x"]
        for r in self.__R:
            for f in self.__F:
                idx = self.__RF2Idx[r, f]
                pred_prob = X[idx]
                RF2PredProb[r, f] = pred_prob

        self.__rf_df["pred_prob"] = self.__rf_df.apply(lambda x: RF2PredProb[x["rcen"], x["freq"]], axis=1)
        print(self.__rf_df)

        Freq = self.__rf_df["freq"].unique().tolist()
        Rcen = self.__rf_df["rcen"].unique().tolist()
        Z = [self.__rf_df[(self.__rf_df["freq"] == freq) & (self.__rf_df["rcen"] == rcen)]["pred_prob"].iloc[0] for freq in Freq for rcen in Rcen]
        Z = np.array(Z).reshape((len(Rcen), len(Freq)))
        X, Y = np.meshgrid(Rcen, Freq)
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            projection="3d",
            xlabel="rcen",
            ylabel="freq",
            zlabel="pred_prob"
        )
        ax.plot_wireframe(X, Y, Z)
        plt.show()

        Rows4 = [
            ('item1', 1, 6),
            ('item2', 2, 2),
            ('item3', 1, 2),
            ('item4', 1, 1)
        ]
        sample_df = pd.DataFrame(Rows4, columns=["item_name", "rcen", "freq"])
        print(pd.merge(sample_df, self.__rf_df, left_on=["rcen", "freq"], right_on=["rcen", "freq"]))

    def check_brushup_assumption(self):
        self.__rcen_df["prob"].diff().plot.bar()
        plt.show()
        self.__freq_df["prob"].diff().plot.bar()
        plt.show()


if __name__ == '__main__':
    solver = RecommendProblem()
    solver.read_data()
    solver.checK_assumption()
    solver.cal_rf_df()
    solver.build()
    solver.solve()
    solver.check_solution()
