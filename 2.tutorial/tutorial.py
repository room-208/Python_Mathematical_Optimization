import pulp
import pandas as pd
import os

DIRNAME = "./2.tutorial"


def solve_1():

    problem = pulp.LpProblem("SLE", pulp.LpMaximize)

    x = pulp.LpVariable("x", cat='Continuous')
    y = pulp.LpVariable("y", cat='Continuous')

    # problem += 120*x + 150 * y == 1440
    # problem += x+y == 10
    problem.addConstraint(120*x + 150 * y == 1440)
    problem.addConstraint(x+y == 10)

    status = problem.solve()

    print(f"Status:{pulp.LpStatus[status]}")
    print(f"x={x.value()}, y={y.value()},")


def solve_2():

    problem = pulp.LpProblem("LP", pulp.LpMaximize)

    x = pulp.LpVariable("x", cat='Continuous')
    y = pulp.LpVariable("y", cat='Continuous')

    problem += x + 3 * y <= 30
    problem += 2*x + y <= 40
    problem += x >= 0
    problem += y >= 0
    problem += x + 2*y

    status = problem.solve()

    print(f"Status:{pulp.LpStatus[status]}")
    print(f"x={x.value()}, y={y.value()}, obj={problem.objective.value()}")


def solve_3():

    stock_df = pd.read_csv(os.path.join(DIRNAME, "stocks.csv"))
    require_df = pd.read_csv(os.path.join(DIRNAME, "requires.csv"))
    gain_df = pd.read_csv(os.path.join(DIRNAME, "gains.csv"))

    P = gain_df["p"].to_list()
    M = stock_df["m"].to_list()

    stock = {row["m"]: row["stock"] for _, row in stock_df.iterrows()}
    require = {(row["p"], row["m"]): row["require"] for _, row in require_df.iterrows()}
    gain = {row["p"]: row["gain"] for _, row in gain_df.iterrows()}

    problem = pulp.LpProblem("LP", pulp.LpMaximize)

    x = {}
    for p in P:
        #x[p] = pulp.LpVariable(f"x_{p}", cat='Continuous')
        x[p] = pulp.LpVariable(f"x_{p}", cat='Integer')
    for p in P:
        problem += x[p] >= 0
    for m in M:
        problem += pulp.lpSum([require[p, m]*x[p] for p in P]) <= stock[m]
    for m in M:
        problem += pulp.lpSum([gain[p]*x[p] for p in P])

    status = problem.solve()

    print(f"Status:{pulp.LpStatus[status]}")
    for p in P:
        print(f"x[{p}] = {x[p].value()}")
    print(f"obj = {problem.objective.value()}")


if __name__ == "__main__":
    solve_3()
