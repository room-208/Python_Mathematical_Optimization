import numpy as np
import matplotlib.pyplot as plt
import pulp
from itertools import product
import time


class TSPSolver:
    def __init__(self) -> None:
        np.random.seed(10)
        self.num_places = 30

        self.dist_range = 1000

        self.K = range(self.num_places)
        self.o = 0
        self.K_minus_0 = self.K[1:]
        self._K = np.random.randint(-self.dist_range, self.dist_range, size=(len(self.K), 2))
        self._K[self.o, :] = 0
        self.t = np.array(
            [[np.linalg.norm(self._K[k]-self._K[l], ord=2)for k in self.K]for l in self.K])

        self.prob = None
        self.x = None
        self.u = None

    def build(self):

        self.prob = pulp.LpProblem(sense=pulp.LpMinimize)

        self.x = {(k, l): pulp.LpVariable(f'x_{k}_{l}', cat='Binary')
                  if k != l else pulp.LpAffineExpression()
                  for k, l in product(self.K, self.K)}

        self.u = {k: pulp.LpVariable(f'u_{k}', lowBound=1, upBound=len(self.K)-1, cat='Continuous')
                  for k in self.K_minus_0}

        for l in self.K:
            self.prob += pulp.lpSum([self.x[k, l] for k in self.K]) == 1
            self.prob += pulp.lpSum([self.x[l, k] for k in self.K]) == 1

        for k, l in product(self.K_minus_0, self.K_minus_0):
            self.prob += self.u[k] + 1 <= self.u[l] + (len(self.K)-1)*(1-self.x[k, l])

        self.prob += pulp.lpSum([self.t[k, l]*self.x[k, l]
                                for k, l in product(self.K, self.K)])

    def solve(self):
        s = time.time()
        status = self.prob.solve(pulp.PULP_CBC_CMD(msg=0))
        print("N", self.num_places)
        print("Status", pulp.LpStatus[status])
        print("Objective", self.prob.objective.value())
        print(time.time() - s)

    def plot_before_solve(self):
        ax = plt.subplot()
        ax.scatter(self._K[1:, 0], self._K[1:, 1], marker='x')
        ax.scatter(self._K[0, 0], self._K[0, 1], marker='o')
        ax.set_aspect('equal')
        plt.show()

    def plot_after_solve(self):
        ax = plt.subplot()
        ax.scatter(self._K[1:, 0], self._K[1:, 1], marker='x')
        ax.scatter(self._K[0, 0], self._K[0, 1], marker='o')

        for k, l in product(self.K, self.K):
            if self.x[k, l].value() == 1:
                p_from = self._K[k]
                p_to = self._K[l]
                ax.arrow(
                    *p_from, *(p_to-p_from),
                    head_width=3,
                    length_includes_head=True,
                    overhang=0.5,
                    color='gray',
                    alpha=0.5
                )

        ax.set_aspect('equal')
        plt.show()


if __name__ == "__main__":
    solver = TSPSolver()
    solver.plot_before_solve()
    solver.build()
    solver.solve()
    solver.plot_after_solve()
