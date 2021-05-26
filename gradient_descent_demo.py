'''
*************************
Gradient Descent Demo
Implementation of the gradient descent algorithm with linear regression
Creator: Dhruv N. Patel (2021)
**************************
'''

import numpy as np

# NOTE: Program assumes no data validation required

# necessary functions
def J(t0, t1): # cost function
    cost = 0

    for i in range(m):
        cost += ((t0 + t1*data_set[i,0]) - data_set[i, 1])**2

    cost /= 2*m

    return cost

def d_dt0_of_J(t0, t1): # partial derivative of J with respect to t0
    value = 0

    for i in range(m):
        value += (t0 + t1*data_set[i,0]) - data_set[i, 1]

    value /= m

    return value

def d_dt1_of_J(t0, t1): # partial derivative of J with respect to t1
    value = 0

    for i in range(m):
        value += ((t0 + t1*data_set[i,0]) - data_set[i, 1])*data_set[i,0]

    value /= m

    return value

# INTRO
print("\nDemonstration of the Gradient Descent Algorithm using Linear Regression")
print("-------------------------------------------------------------------------\n")


# simple way to get data from file without needing with-as block
data_set = np.genfromtxt("data_set.txt", delimiter=' ', skip_header=3)
print(f"Inputted data set:\n{data_set}\n")

m = data_set.shape[0]


# entering initial values of the parameters
t0 = float(input("Enter initial value for parameter t0: "))
t1 = float(input("Enter initial value for parameter t1: "))
a = float(input("Enter value for learning step a: "))
cost = J(t0, t1)
previous_cost = cost


# algorithm execution
print("\nBeginning gradient descent...\n")

print("INITIALLY...")

while(cost > 1e-12):
    print(f"t0 = {t0}")
    print(f"t1 = {t1}")
    print(f"Hypothesis function: {t0} + {t1}x")
    print(f"Cost = {cost}")
    print("NEXT ITERATION\n")
    temp0 = t0 - a*d_dt0_of_J(t0, t1)
    temp1 = t1 - a*d_dt1_of_J(t0, t1)
    t0 =  temp0
    t1 = temp1
    cost = J(t0, t1)

print("Final Results")
print("-------------")
print(f"t0 = {t0}")
print(f"t1 = {t1}")
print(f"Hypothesis function: {t0} + {t1}x")
print(f"Cost = {cost}")
