import pulp


def solve():

    problem = pulp.LpProblem("SLE", pulp.LpMaximize)

    x = pulp.LpVariable("x", cat='Continuous')
    y = pulp.LpVariable("y", cat='Continuous')

    #problem += 120*x + 150 * y == 1440
    #problem += x+y == 10
    problem.addConstraint(120*x + 150 * y == 1440)
    problem.addConstraint(x+y == 10)

    status = problem.solve()

    print(f"Status:{pulp.LpStatus[status]}")
    print(f"x={x.value()}, y={y.value()},")


if __name__ == "__main__":
    solve()
