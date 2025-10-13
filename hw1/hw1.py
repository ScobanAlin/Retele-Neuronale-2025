import math
import pathlib

def reader(path) ->tuple[list[list[float]],list[float]]:
    A = []
    B = []

    with open(path, 'r') as f:
        for line in f:
            line = line.replace(' ','');
            left , right = line.split('=')
            B.append(float(right.strip()))
            x=y=z=0.0
            terms = left.replace('-', '+-').split('+')
            for term in terms:
                if 'x' in term:

                    coefficient = term.replace('x', '').strip()
                    if coefficient not in ('+' , '-' ,''):
                        x = float(coefficient)
                    else:
                        x =float(f"{coefficient}1")

                elif 'y' in term:
                    coefficient = term.replace('y', '').strip()
                    if coefficient not in ('+', '-', ''):
                        y = float(coefficient)
                    else:
                        y=float(f"{coefficient}1")

                elif 'z' in term:
                    coefficient = term.replace('z', '').strip()
                    if coefficient not in ('+', '-', ''):
                        z = float(coefficient)
                    else:
                        z = float(f"{coefficient}1")
            A.append([x,y,z])
    print('A=', A)
    print('B=', B)
    return A,B

A,B = reader(pathlib.Path("system.txt"))

def determinant(matrix: list[list[float]]) -> float:
    a,b,c = matrix[0]
    d,e,f = matrix[1]
    g,h,i = matrix[2]
    return a*i*e + b*f*g + d*h*c -c*e*g -a*f*h - d*b*i

print(f"{determinant(A)=}")

def trace(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a+e+i

print(f"{trace(A)=}")

def norm(vector):
    squareSum = 0.0
    for elem in vector:
        squareSum += elem**2
    return math.sqrt(squareSum)

print(f"{norm(B)=}")


def transpose(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return [[a,d,g],[b,e,h],[c,f,i]]

print(f"{transpose(A)=}")


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    a,b,c= matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    x,y,z = vector

    r1 = a*x +b*y +c*z
    r2 = d*x +e*y +f*z
    r3 = g*x +h*y +i*z
    return [r1,r2,r3]

print(f"{multiply(A,B)=}")

def replace_column(matrix,vector,column):
    return [[vector[r] if  c == column else matrix[r][c] for c in range(3)] for r in range(3)]

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    detA = determinant(matrix)
    if detA == 0:
        raise ValueError("No unique solution")

    Ax = replace_column(matrix, vector, 0)
    Ay = replace_column(matrix, vector, 1)
    Az = replace_column(matrix, vector, 2)
    detAx = determinant(Ax)
    detAy = determinant(Ay)
    detAz = determinant(Az)

    x = detAx / detA
    y = detAy / detA
    z = detAz / detA
    return [x,y,z]


print(f"{solve_cramer(A,B)=}")

def solve_using_inverse(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix)
    if det == 0:
        raise ValueError("No unique solution")

    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    cof = [
        [ (e*i - f*h), -(d*i - f*g),  (d*h - e*g)],
        [-(b*i - c*h),  (a*i - c*g), -(a*h - b*g)],
        [ (b*f - c*e), -(a*f - c*d),  (a*e - b*d)]
    ]

    adj = [[cof[j][i] for j in range(3)] for i in range(3)]

    inv = [[adj[i][j] / det for j in range(3)] for i in range(3)]

    result = [
        inv[0][0]*vector[0] + inv[0][1]*vector[1] + inv[0][2]*vector[2],
        inv[1][0]*vector[0] + inv[1][1]*vector[1] + inv[1][2]*vector[2],
        inv[2][0]*vector[0] + inv[2][1]*vector[1] + inv[2][2]*vector[2]
    ]

    return result

print(f"{solve_using_inverse(A,B)=}")

def minor(matrix, i, j):
    new_matrix = []
    for row_idx, row in enumerate(matrix):
        if row_idx == i:
            continue
        new_row = []
        for col_idx, val in enumerate(row):
            if col_idx == j:
                continue
            new_row.append(val)
        new_matrix.append(new_row)
    return new_matrix

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    return [[(-1)**(i+j) * determinant(minor(matrix, i, j)) for j in range(3)] for i in range(3)]

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    cof = cofactor(matrix)
    return [[cof[j][i] for j in range(3)] for i in range(3)]


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)
    if det_A == 0:
        raise ValueError("Matrix is singular, cannot solve")

    solution = []
    for i in range(3):
        temp = [row[:] for row in matrix]
        for j in range(3):
            temp[j][i] = vector[j]
        solution.append(determinant(temp) / det_A)
    return solution

print(f"{solve(A,B)=}")