# This file predicts the position of a Celestial Body at any given time using ODE based simulation
from a_imports import *
from b_constants import GM_SUN

g = GM_SUN #idw write allat everytime so made an alias

'''
Phase space: y = [x,y,z,v_x,v_y,v_z] 
Thus,
Positions: y[:3] 
Velocities: y[3:]
'''
def gravity(t,y):
    r = y[:3] 
    v = y[3:] 

    # r is a position vector, let R be it's magnitude (Needed for vector form of Newton's equation of gravity)
    R = np.linalg.norm(r)
    drdt = v
    dvdt = -g*r/(R**3) # This is just -(GM/|r|^2).r (r is the position vector = (x y z))

    return np.concatenate([drdt,dvdt])

'''
t0 = Current time
r0, v0 = Current position and velocity (known)
t is the array of all times we need the positions at
'''
def simulate(r0, v0, t0, t):
    state0 = np.concatenate([r0,v0])
    
    t_before = []
    t_after =[]
    for i in range(0,len(t)):
        if (t[i] >= t0): t_after.append(t[i])
        if (t[i] < t0): t_before.append(t[i])

    beforeExists = len(t_before) > 0
    afterExists = len(t_after) > 0

    # Going till least value in t_before covers all numbers between t0 and t1, similarly for t2
    if (beforeExists): # Else it crashes
        t1 = min(t_before)
        solutions_before = solve_ivp(gravity, (t0, t1), state0, dense_output = True, rtol = 1e-12, atol = 1e-14)
        state_t_before = solutions_before.sol(t_before)

    if (afterExists):
        t2 = max(t_after)
        solutions_after = solve_ivp(gravity, (t0,t2), state0, dense_output = True, rtol = 1e-12, atol = 1e-14)
        state_t_after = solutions_after.sol(t_after)

    # Concatenated State Matrices of the Phase space at time t    
    if (beforeExists == True and afterExists == True):
        state_t = np.concatenate((state_t_before, state_t_after), axis = 1)
    if (beforeExists == True and afterExists == False):
        state_t = state_t_before
    if (beforeExists == False and afterExists == True):
        state_t = state_t_after
    '''
    Let, t1,t2 < t and t3,t4 > t so we get this (axis = 1 means horizontal concatenation)
               t=t1     t=t2     t=t3     t=t4
        x    [ ...      ...      ...      ... ]
        y    [ ...      ...      ...      ... ] 
        z    [ ...      ...      ...      ... ] 
        vx   [ ...      ...      ...      ... ] 
        vy   [ ...      ...      ...      ... ] 
        vz   [ ...      ...      ...      ... ] 

    NOTE:
        t (input) MUST be sorted to maintain mapping between the final state array and initial t array
    '''

    return state_t[:3].T # First 3 elements being the positions (Transposed to convert to a standard vector)


    