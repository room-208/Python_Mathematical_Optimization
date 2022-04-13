from turtle import pu
from IPython.core.display import display
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pulp
from itertools import product, combinations_with_replacement
from joblib import Parallel, delayed
from soupsieve import select

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
        self._K = np.random.normal(0, self.mean_travel_time_to_destinations, size=(len(self.K), 2))
        self._K[self.o, :] = 0
        self.t = np.array([[np.floor(np.linalg.norm(self._K[k]-self._K[l]))for k in self.K]for l in self.K])
        self.D = range(self.num_days)

        self.R = range(self.num_requests)
        self.k = np.random.choice(self.K_minus_0, size=len(self.R))
        print(self.k)
        self.d_0 = np.random.choice(self.D, size=len(self.R))
        self.d_1 = self.d_0 + self.delivery_time_window-1
        self.w = np.floor(np.random.gamma(10, self.avg_weight/10, size=len(self.R)))
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

        travel = pulp.lpSum([self.t[k, l]*x[k, l] for k, l in product(self.K, self.K)])
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
        routes = Parallel(n_jobs=16)(
            [delayed(self.simulate_route)(z) for z in product([0, 1], repeat=len(self.K))]
        )
        routes = pd.DataFrame(filter(lambda x: x is not None, routes))
        self.routes_df = routes[routes.optimal].copy()

    def is_OK(self, requests):
        weight = sum([self.w[r] for r in requests])
        if weight > self.W:
            return False

        best_route_idx = None
        best_hours = sys.float_info.max
        for route_idx, row in self.routes_df.iterrows():
            all_requests_on_route = all([row.z[self.k[r]] == 1 for r in requests])
            if all_requests_on_route and row.移動時間 < best_hours:
                best_route_idx = route_idx
                best_hours = row.移動時間
        if best_route_idx is None:
            return False
        else:
            return best_route_idx, best_hours


if __name__ == "__main__":
    solver = RouteSolver()
    # solver.plot_everything()
    print(solver.enumerate_routes())
