import pulp
import pandas as pd
import matplotlib.pyplot as plt
import os

DIRNAME = "./3.school"


def solve():

    student_df = pd.read_csv(os.path.join(DIRNAME, "students.csv"))
    student_pair_df = pd.read_csv(os.path.join(DIRNAME, "student_pairs.csv"))

    problem = pulp.LpProblem("ClassAssignmentProblem", pulp.LpMaximize)

    S = student_df["student_id"].to_list()
    C = ["A", "B", "C", "D", "E", "F", "G", "H"]

    x = {}
    for s in S:
        for c in C:
            x[s, c] = pulp.LpVariable(f"x_{s,c}", cat='Binary')

    for s in S:
        problem += pulp.lpSum([x[s, c] for c in C]) == 1

    for c in C:
        problem += pulp.lpSum([x[s, c] for s in S]) <= 40
        problem += pulp.lpSum([x[s, c] for s in S]) >= 39

    S_male = set([row["student_id"] for _, row in student_df.iterrows() if row["gender"] == 1])
    S_female = set([row["student_id"] for _, row in student_df.iterrows() if row["gender"] == 0])
    for c in C:
        problem += pulp.lpSum([x[s, c] for s in S if s in S_male]) <= 20
        problem += pulp.lpSum([x[s, c] for s in S if s in S_female]) <= 20

    score_mean = student_df["score"].mean()
    score = {row["student_id"]: row["score"] for _, row in student_df.iterrows()}
    for c in C:
        problem += pulp.lpSum([x[s, c]*score[s] for s in S]) <= pulp.lpSum([x[s, c] for s in S])*(score_mean + 10)
        problem += pulp.lpSum([x[s, c]*score[s] for s in S]) >= pulp.lpSum([x[s, c] for s in S])*(score_mean - 10)

    S_leader = set([row["student_id"] for _, row in student_df.iterrows() if row["leader_flag"] == 1])
    for c in C:
        problem += pulp.lpSum([x[s, c] for s in S if s in S_leader]) >= 2

    S_support = set([row["student_id"] for _, row in student_df.iterrows() if row["support_flag"] == 1])
    for c in C:
        problem += pulp.lpSum([x[s, c] for s in S if s in S_support]) <= 1

    SS = [(row["student_id1"], row["student_id2"]) for _, row in student_pair_df.iterrows()]
    for s1, s2 in SS:
        for c in C:
            problem += x[s1, c]+x[s2, c] <= 1

    status = problem.solve()

    print(f"Status:{pulp.LpStatus[status]}")

    ans = {}
    for c in C:
        ans[c] = [s for s in S if x[s, c].value() == 1]

    for c, Ss in ans.items():
        print(f"Class: {c}")
        print(f"Num: {len(Ss)}")
        print(f"Student: {Ss}")
        print()


if __name__ == "__main__":
    solve()
