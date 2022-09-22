


# A = [[1,0,0],
#      [0,5,2],
#      [0,2,5]]


# a,b = np.linalg.eig(A)
# j = np.linalg.det(A)
# # print(j)
# # for i in range (len(a)):
# #     print('特征值，',a[i],'对应的特征向量',b[:,i])
# mu = np.matrix([[1],
#       [2],
#       [2]])

# Lambd = [[1,0,0],
#          [0,3,0],
#          [0,0,7]]
# sqrt_2_2 = np.sqrt(2)/2
# Phi =np.matrix( [[1,0,0],
#        [0,sqrt_2_2,sqrt_2_2],
#        [0,-sqrt_2_2,sqrt_2_2]])

# Lambd_12 = [[1**(-0.5),0,0],
#          [0,3**(-0.5),0],
#          [0,0,7**(-0.5)]]
# Aw = Phi@(Lambd_12)
# x0 = np.matrix([[0.5],[0],[1]])



# inv_A = np.linalg.inv(A)
# r2 = (x0-mu).T @inv_A@(x0-mu)
# r2_2 = (Aw.T@x0-Aw.T @ mu).T @(Aw.T@x0-Aw.T @ mu)
# print(r2)
# print(r2_2)
# print(Aw.T @ mu)
# print(Aw.T @ A @ Aw)
# print((Aw.T@x0).T @(Aw.T@x0))