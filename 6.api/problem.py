import pandas as pd
import os
import pulp

DIRNAME = "./6.api/resource"


class CarAssignSolver:
    def __init__(self) -> None:
        self.__students_df = None
        self.__cars_df = None
        self.__prob = None
        self.__S = None
        self.__C = None
        self.__G = None
        self.__SC = None
        self.__S_licence = None
        self.__S_g = None
        self.__S_male = None
        self.__S_female = None
        self.__U = None
        self.__x = None

    def read_data(self):
        self.__students_df = pd.read_csv(os.path.join(DIRNAME, "students.csv"))
        self.__cars_df = pd.read_csv(os.path.join(DIRNAME, "cars.csv"))

    def build(self):
        self.__prob = pulp.LpProblem(sense=pulp.LpMaximize)

        self.__S = self.__students_df["student_id"].to_list()
        self.__C = self.__cars_df["car_id"].to_list()
        self.__G = [1, 2, 3, 4]
        self.__SC = [(s, c) for s in self.__S for c in self.__C]
        self.__S_licence = self.__students_df[self.__students_df["license"] == 1]["student_id"].to_list()
        self.__S_g = {g: self.__students_df[self.__students_df["grade"] == g]["student_id"].to_list() for g in self.__G}
        self.__male = self.__students_df[self.__students_df["gender"] == 0]["student_id"].to_list()
        self.__female = self.__students_df[self.__students_df["gender"] == 1]["student_id"].to_list()
        self.__U = self.__cars_df["capacity"].to_list()

        self.__x = pulp.LpVariable.dicts('x', self.__SC, cat='Binary')

        for s in self.__S:
            self.__prob += pulp.lpSum([self.__x[s, c] for c in self.__C]) == 1

        for c in self.__C:
            self.__prob += pulp.lpSum([self.__x[s, c] for s in self.__S]) <= self.__U[c]

        for c in self.__C:
            self.__prob += pulp.lpSum([self.__x[s, c] for s in self.__S_licence]) >= 1

        for c in self.__C:
            for g in self.__G:
                self.__prob += pulp.lpSum([self.__x[s, c] for s in self.__S_g[g]]) >= 1

        for c in self.__C:
            self.__prob += pulp.lpSum([self.__x[s, c] for s in self.__male]) >= 1

        for c in self.__C:
            self.__prob += pulp.lpSum([self.__x[s, c] for s in self.__female]) >= 1

    def solve(self):
        self.__prob.solve()

    def check_solution(self):
        car2students = {
            c: [s for s in self.__S if self.__x[s, c].value() == 1] for c in self.__C
        }

        for c, ss in car2students.items():
            print(f"car = {c}")
            print(f"capacity  = {self.__U[c]}")
            print(f"students  = {ss}")


if __name__ == "__main__":
    solver = CarAssignSolver()
    solver.read_data()
    solver.build()
    solver.solve()
    solver.check_solution()
