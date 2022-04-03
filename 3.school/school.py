from unittest import result
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import os

DIRNAME = "./3.school"


def get_init_flag(student_df, S, C):
    df = student_df.copy()
    df = df.sort_values(by='score', ascending=False)
    class_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
    df["init_assigned_class"] = -1
    for i in range(len(df)):
        df.loc[i, "init_assigned_class"] = class_dict[i % 8]
    init_flag = {(s, c): 0 for s in S for c in C}
    for _, row in df.iterrows():
        init_flag[row["student_id"], row["init_assigned_class"]] = 1
    return init_flag


def solve():

    student_df = pd.read_csv(os.path.join(DIRNAME, "students.csv"))
    student_pair_df = pd.read_csv(os.path.join(DIRNAME, "student_pairs.csv"))

    S = student_df["student_id"].to_list()
    C = ["A", "B", "C", "D", "E", "F", "G", "H"]

    init_flag = get_init_flag(student_df, S, C)

    problem = pulp.LpProblem("ClassAssignmentProblem", pulp.LpMaximize)

    x = {}
    for s in S:
        for c in C:
            x[s, c] = pulp.LpVariable(f"x_{s,c}", cat='Binary')

    problem += pulp.lpSum([x[s, c] for s in S for c in C if init_flag[s, c] == 1])

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

    for s in S:
        assigned_class = [x[s, c].value() for c in C if x[s, c].value() == 1]
        assert len(assigned_class) == 1, f"error: {s, assigned_class}"

    S2C = {s: c for s in S for c in C if x[s, c].value() == 1}

    result_df = student_df.copy()
    result_df["assigned_class"] = result_df["student_id"].map(S2C)
    print(result_df.groupby(["assigned_class"])["student_id"].count())
    print(result_df.groupby(["assigned_class", "gender"])["student_id"].count())
    print(result_df.groupby(["assigned_class"])["score"].mean())
    print(result_df.groupby(["assigned_class"])["leader_flag"].sum())
    print(result_df.groupby(["assigned_class"])["support_flag"].sum())

    SS = [(row["student_id1"], row["student_id2"]) for _, row in student_pair_df.iterrows()]
    for s1, s2 in SS:
        c1 = S2C[s1]
        c2 = S2C[s2]

        assert c1 != c2, "c1 == c2"
        print(f"s1 = {s1}, c1 = {c1}")
        print(f"s2 = {s2}, c2 = {c2}")

    fig = plt.figure(figsize=(12, 20))
    for i, c in enumerate(C):
        cls_df = result_df[result_df["assigned_class"] == c]
        ax = fig.add_subplot(4, 2, i+1, xlabel="score", ylabel="num", xlim=(0, 500), ylim=(0, 20), title=f"Class:{c}")
        ax.hist(cls_df["score"], bins=range(0, 500, 40))
    plt.show()


if __name__ == "__main__":
    solve()
