import numpy as np
import math
import itertools


def GraspPointFiltering(numPts, P, N, C):
    # create superset of all possibilities:
    counter = list(range(0, numPts))
    points = list(itertools.combinations(counter, 2))
    curvatureVals = []

    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        curvatureVals.append(Term1(C[x], C[y]))

    curvatureVals = np.asarray(curvatureVals)

    # now sort the curvature values high to low for grasping point filter.
    sortIndices = (-curvatureVals).argsort()
    for i in range(0, len(points)):

        idx = sortIndices[i]
        x = points[idx][0]
        y = points[idx][1]

        # perform force closure test
        fcTest = Term2(P[x], P[y], N[x], N[y])

        if curvatureVals[idx] <= 0:
            break

        if fcTest < 0.34:
            return x, y

    # if we have finished the concave, test the convex...
    # sort low to high
    sortIndices = (curvatureVals).argsort()
    for i in range(0, len(points)):

        idx = sortIndices[i]
        x = points[idx][0]
        y = points[idx][1]

        # perform force closure test
        fcTest = Term2(P[x], P[y], N[x], N[y])

        if curvatureVals[idx] <= 0:
            break

        if fcTest < 0.34:
            return x, y

    # if we haven't returned anything till now, there are no good points...
    return 1, 1


def Term1(C1, C2):
    # print(C1+C2)
    return C1 + C2


def Term2(Pm1, Pm2, Nm1, Nm2):
    s1 = np.subtract(Pm1, Pm2)
    s2 = np.subtract(Pm1, Pm2)
    sub1 = s1 / np.linalg.norm(s1)
    sub2 = s2 / np.linalg.norm(s2)

    norm1 = Nm1 / np.linalg.norm(Nm1)
    norm2 = Nm2 / np.linalg.norm(Nm2)

    if np.dot(norm1, sub1) < 0:
        my_sub = sub2
    else:
        my_sub = sub1

    A = math.acos(np.dot(norm1, my_sub))
    B = math.acos(np.dot(norm2, my_sub))

    # print(math.degrees(A))
    # print(math.degrees(np.pi- B))

    return A ** 2 + (np.pi - B) ** 2


def FindBestGrasps(numPts, P, N, C):
    # THIS WAS AN ATTEMPT TO USE A GRADIENT ASCENT ALGORITHM TO OPTIMIZE THE GRASPING POINTS. IT FAILED HOWEVER,
    # BECAUSE THE OPTIMIZATION FUNCTION I USED WAS NOT BOUNDED, AND THUS NEVER CONVERGED. THERE IS PROBABLY SOME SIMILAR
    # FUNCTION THAT COULD WORK... ANYHOW, HERE IS THE CODE FOR GRADIENT DESCENT IF ANYONE GETS A WORKING FUNCTION.
    # gradient ascent to optimize the objective wr*Term1 - wf*Term2

    # find the entire set of pairs that we want to optimize.
    counter = list(range(0, numPts))
    points = list(itertools.combinations(counter, 2))

    learning_rate = 1
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              compute_total_score(
                                                                                  initial_b, initial_m, points, N, P,
                                                                                  C)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, N, P, C)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                      compute_total_score(b, m,
                                                                                          points, N, P, C)))


def compute_total_score(a, b, points, N, P, C):
    totalScore = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        print((a * Term1(C[x], C[y])) - (b * Term2(P[x], P[y], N[x], N[y])))
        totalScore += (a * Term1(C[x], C[y])) - (b * Term2(P[x], P[y], N[x], N[y]))
    return totalScore


def gradient_descent_runner(points, starting_a, starting_b, learning_rate, num_iterations, N, P, C):
    a = starting_a
    b = starting_b
    for i in range(num_iterations):
        a, b = step_gradient(a, b, points, learning_rate, N, P, C)
        print(compute_total_score(a, b, points, N, P, C), a, b)
    return [a, b]


def step_gradient(a_current, b_current, points, learningRate, N, P, C):
    a_g = 0
    b_g = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]

        # a_gradient += 1.0 / len(points) * Term1(C[x], C[y])
        # b_gradient += 1.0 / len(points) * -Term2(P[x], P[y], N[x], N[y])

        t1 = Term1(C[x], C[y])
        t2 = Term2(P[x], P[y], N[x], N[y])

        a_g += (1.0 / len(points)) * ((a_current * t1 - b_current * t2) - t1 - t2) * t1
        b_g += (1.0 / len(points)) * ((a_current * t1 - b_current * t2) - t1 - t2) * t2

    new_b = a_current + (learningRate * a_g)
    new_m = b_current + (learningRate * b_g)
    return [new_b, new_m]
