import pandas as pd
import os
import pulp

DIRNAME = "./6.api/resource"


class CarGroupProblem():
    def __init__(self, students_df, cars_df, name='ClubCarProblem'):
        self.__students_df = students_df
        self.__cars_df = cars_df
        self.__name = name
        self.__prob = self.__formulate()

    def __formulate(self):
        prob = pulp.LpProblem(sense=pulp.LpMaximize)

        S = self.__students_df["student_id"].to_list()
        C = self.__cars_df["car_id"].to_list()
        G = [1, 2, 3, 4]
        SC = [(s, c) for s in S for c in C]
        S_licence = self.__students_df[self.__students_df["license"] == 1]["student_id"].to_list()
        S_g = {g: self.__students_df[self.__students_df["grade"] == g]["student_id"].to_list() for g in G}
        male = self.__students_df[self.__students_df["gender"] == 0]["student_id"].to_list()
        female = self.__students_df[self.__students_df["gender"] == 1]["student_id"].to_list()
        U = self.__cars_df["capacity"].to_list()

        x = pulp.LpVariable.dicts('x', SC, cat='Binary')

        for s in S:
            prob += pulp.lpSum([x[s, c] for c in C]) == 1

        for c in C:
            prob += pulp.lpSum([x[s, c] for s in S]) <= U[c]

        for c in C:
            prob += pulp.lpSum([x[s, c] for s in S_licence]) >= 1

        for c in C:
            for g in G:
                prob += pulp.lpSum([x[s, c] for s in S_g[g]]) >= 1

        for c in C:
            prob += pulp.lpSum([x[s, c] for s in male]) >= 1

        for c in C:
            prob += pulp.lpSum([x[s, c] for s in female]) >= 1

        return {'prob': prob, 'variable': {'x': x}, 'list': {'S': S, 'C': C}}

    def solve(self):
        self.__prob['prob'].solve()
        x = self.__prob['variable']['x']
        S = self.__prob['list']['S']
        C = self.__prob['list']['C']

        car2students = {c: [s for s in S if x[s, c].value() == 1] for c in C}
        student2car = {s: c for c, ss in car2students.items() for s in ss}
        solution_df = pd.DataFrame(list(student2car.items()), columns=["student_id", "car_id"])
        solution_df = solution_df.sort_values("student_id")
        solution_df = solution_df.reset_index(drop=True)

        return solution_df


if __name__ == '__main__':
    # データの読み込み
    students_df = pd.read_csv(os.path.join(DIRNAME, "students.csv"))
    cars_df = pd.read_csv(os.path.join(DIRNAME, "cars.csv"))

    # 数理モデル インスタンスの作成
    prob = CarGroupProblem(students_df, cars_df)

    # 問題を解く
    solution_df = prob.solve()

    # 結果の表示
    print('Solution: \n', solution_df)
