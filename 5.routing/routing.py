from turtle import pu
from typing import overload
from IPython.core.display import display
from flask import current_app
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pulp
from itertools import product, combinations_with_replacement
from joblib import Parallel, delayed

DIRNAME = "./5.routing"


class RouteSolver:
    def __init__(self) -> None:
        np.random.seed(10)
        self.num_places = 10
        self.num_days = 30
        self.num_requests = 120
        self.mean_travel_time_to_destinations = 100
        self.H_regular = 8*60
        self.H_max_overtime = 3*60
        self.c = 3000/60
        self.W = 4000
        self.delivery_outsourcing_unit_cost = 4600
        self.delivery_time_window = 3
        self.avg_weight = 1000

        self.K = range(self.num_places)
        self.o = 0
        self.K_minus_0 = self.K[1:]
        self._K = np.random.normal(
            0, self.mean_travel_time_to_destinations, size=(len(self.K), 2))
        self._K[self.o, :] = 0
        self.t = np.array(
            [[np.floor(np.linalg.norm(self._K[k]-self._K[l]))for k in self.K]for l in self.K])
        self.D = range(self.num_days)

        self.R = range(self.num_requests)
        self.k = np.random.choice(self.K_minus_0, size=len(self.R))
        self.d_0 = np.random.choice(self.D, size=len(self.R))
        self.d_1 = self.d_0 + self.delivery_time_window-1
        self.w = np.floor(np.random.gamma(
            10, self.avg_weight/10, size=len(self.R)))
        self.f = np.ceil(self.w/100)*self.delivery_outsourcing_unit_cost

        self.routes_df = None

    def plot_everything(self):
        a = plt.subplot()
        a.scatter(self._K[1:, 0], self._K[1:, 1], marker='x')
        a.scatter(self._K[0, 0], self._K[0, 1], marker='o')
        a.set_aspect('equal')
        plt.show()
        plt.hist(self.w, bins=20, range=(0, 2000))
        plt.show()

    def simulate_route(self, z):
        if z[0] == 0:
            return None

        daily_route_prob = pulp.LpProblem(sense=pulp.LpMinimize)

        x = {(k, l): pulp.LpVariable(f'x_{k}_{l}', cat='Binary')
             if k != l else pulp.LpAffineExpression()
             for k, l in product(self.K, self.K)}

        u = {k: pulp.LpVariable(f'u_{k}', lowBound=1, upBound=len(self.K)-1, cat='Continuous')
             for k in self.K_minus_0}

        h = pulp.LpVariable('h', lowBound=0, cat='Continuous')

        for l in self.K:
            daily_route_prob += pulp.lpSum([x[k, l] for k in self.K]) <= 1

        for l in self.K:
            if z[l] == 1:
                daily_route_prob += pulp.lpSum([x[k, l] for k in self.K]) == 1
                daily_route_prob += pulp.lpSum([x[l, k] for k in self.K]) == 1
            else:
                daily_route_prob += pulp.lpSum([x[k, l] for k in self.K]) == 0
                daily_route_prob += pulp.lpSum([x[l, k] for k in self.K]) == 0

        for k, l in product(self.K_minus_0, self.K_minus_0):
            daily_route_prob += u[k] + 1 <= u[l] + (len(self.K)-1)*(1-x[k, l])

        travel = pulp.lpSum([self.t[k, l]*x[k, l]
                            for k, l in product(self.K, self.K)])
        daily_route_prob += (travel-self.H_regular) <= h
        daily_route_prob += h <= self.H_max_overtime

        daily_route_prob += travel
        daily_route_prob.solve()

        return {
            'z': z,
            'route': {(k, l): x[k, l].value() for k, l in product(self.K, self.K)},
            'optimal': daily_route_prob.status == 1,
            '移動時間': travel.value(),
            '残業時間': h.value()}

    def enumerate_routes(self):
        print("enumerate_routes begin...")
        routes = Parallel(n_jobs=16)(
            [delayed(self.simulate_route)(z)
             for z in product([0, 1], repeat=len(self.K))]
        )
        routes = pd.DataFrame(filter(lambda x: x is not None, routes))
        self.routes_df = routes[routes.optimal].copy()
        print("enumerate_routes end...")

    def is_OK(self, requests):
        weight = sum([self.w[r] for r in requests])
        if weight > self.W:
            return False

        best_route_idx = None
        best_hours = sys.float_info.max
        for route_idx, row in self.routes_df.iterrows():
            all_requests_on_route = all(
                [row.z[self.k[r]] == 1 for r in requests])
            if all_requests_on_route and row.移動時間 < best_hours:
                best_route_idx = route_idx
                best_hours = row.移動時間
        if best_route_idx is None:
            return False
        else:
            return best_route_idx, best_hours

    def _enumerate_feasible_schedules(self, requests_cands, current_idx_set, idx_to_add, res):
        idx_set_to_check = current_idx_set + [idx_to_add]
        next_idx = idx_to_add + 1
        is_next_idx_valid = next_idx < len(requests_cands)
        requests = [requests_cands[i] for i in idx_set_to_check]
        is_ok = self.is_OK(requests)

        print(len(requests_cands))

        # 最悪2^len(requests_cands)試すことになる。
        # requests_candsで積めるパターンを全列挙する。
        if is_ok:
            best_route_idx, best_hours = is_ok
            res.append({'requests': [requests_cands[i]
                                     for i in idx_set_to_check],
                        'route_idx': best_route_idx,
                        'hours': best_hours})

            if is_next_idx_valid:
                self._enumerate_feasible_schedules(
                    requests_cands, idx_set_to_check, next_idx, res)

        if is_next_idx_valid:
            self._enumerate_feasible_schedules(
                requests_cands, current_idx_set, next_idx, res)

    def enumerate_feasible_schedules(self, d):
        print(f"Day = {d} begin...")
        requests_cands = [r for r in self.R if self.d_0[r] <= d <= self.d_1[r]]

        res = [{'requests': [],
                'route_idx': 0,
                'hours': 0}]

        self._enumerate_feasible_schedules(requests_cands, [], 0, res)

        feasible_schedules_df = pd.DataFrame(res)
        feasible_schedules_df["overwork"] = (
            feasible_schedules_df.hours-self.H_regular).clip(0)
        feasible_schedules_df["requests_set"] = feasible_schedules_df.requests.apply(
            set)

        idx_cands = set(feasible_schedules_df.index)
        dominated_idx_set = set()
        for dominant_idx in feasible_schedules_df.index:        # チェックする側
            for checked_idx in feasible_schedules_df.index:     # チェックされる側
                requests_strict_dominance = (
                    feasible_schedules_df.requests_set.loc[checked_idx] <
                    feasible_schedules_df.requests_set.loc[dominant_idx]
                )
                overwork_weak_dominance = (
                    feasible_schedules_df.overwork.loc[checked_idx] >=
                    feasible_schedules_df.overwork.loc[dominant_idx]
                )
                # 劣る計画と判断
                if requests_strict_dominance and overwork_weak_dominance:
                    dominated_idx_set.add(checked_idx)

        nondominated_idx = idx_cands - dominated_idx_set

        # d日における劣らない計画
        nondominated_feasible_schedules_df = feasible_schedules_df.loc[nondominated_idx, :]
        print(f"Day = {d} end...")
        return nondominated_feasible_schedules_df


if __name__ == "__main__":
    solver = RouteSolver()
    # solver.plot_everything()
    solver.enumerate_routes()

    _schedules = Parallel(n_jobs=16)(
        [delayed(solver.enumerate_feasible_schedules)(d) for d in solver.D])
    feasible_schedules = dict(zip(solver.D, _schedules))
    print(feasible_schedules)
