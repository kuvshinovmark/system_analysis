import numpy as np
import json

def task(json_input):
    input_list = np.array(json.loads(json_input)).T
    com_matrixs = []

    def get_matrix(input_col):
      matrix = np.zeros((len(input_col), len(input_col)))
      for row_ind, row in enumerate(matrix):
        for col_ind, col in enumerate(row):
          if input_col[row_ind] < input_col[col_ind]:
            matrix[row_ind, col_ind] = 1
          elif input_col[row_ind] == input_col[col_ind]:
            matrix[row_ind, col_ind] = 0.5
          else:
            matrix[row_ind, col_ind] = 0
      return matrix

    for input_col in input_list.T:
        com_matrixs.append(get_matrix(input_col))

    gen_matrix = np.zeros((len(input_list), len(input_list)))
    for i in range(len(input_list)):
        gen_matrix += com_matrixs[i]
    general_matrix = gen_matrix / len(input_list)

    k0 = np.array([1 / len(input_list)] * len(input_list))
    y = general_matrix.dot(k0)
    lambda1 = (np.ones(len(input_list))).dot(y)
    k1 = 1 / lambda1 * y
    while abs(max(k1 - k0)) >= 0.001:
        k0 = k1
        y = general_matrix.dot(k0)
        lambda1 = (np.ones(len(input_list))).dot(y)
        k1 = 1 / lambda1 * y
    k1 = np.around(k1, 3)
    return json.dumps(k1.tolist())