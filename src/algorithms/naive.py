from typing import List

def multiply_matrices(a: List[List[int]], 
                     b: List[List[int]], 
                     c: List[List[int]], 
                     n: int) -> None:
    for i in range(n):
        for j in range(n):
            c[i][j] = 0
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
