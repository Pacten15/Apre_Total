import numpy as np

c = int(input("n colunas: "))

l = int(input("n linhas: "))

print("Enter the entries in a single line (separated by space): ")

entries = list(map(float, input().split()))

matrix = np.array(entries).reshape(l, c)

inv_matrix = np.linalg.pinv(matrix)

print(inv_matrix)



