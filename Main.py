# MTF073 Computational Fluid Dynamics
# Task 1: 2D diffusion equation
# Template prepared by:
# Gonzalo Montero Villar
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# November 2020
# Packages needed
import numpy as np
import matplotlib.pyplot as plt
#===================== Schematic ==================
#
#                  0----------------0
#                  |                |
#                  |                |
#                  |    [i,j+1]     |
#                  |       X        |
#                  |      (N)       |
#                  |                |
#                  |                |
# 0----------------0----------------0----------------0
# |                |                |                |
# |                |                |                |
# |    [i-1,j]     |     [i,j]      |    [i+1,j]     |
# |       X        |       X        |       X        |
# |      (W)       |      (P)       |      (E)       |
# |                |                |                |
# |                |                |                |
# 0----------------0----------------0----------------0
#                  |                |
#                  |                |
#                  |    [i,j-1]     |
#                  |       X        |
#                  |      (S)       |
#                  |                |
#                  |                |
#                  0----------------0
#
# X:  marks the position of the nodes, which are the centers
#     of the control volumes, where temperature is computed.
# 0:  marks the position of the mesh points or control volume
#     corners.
# []: in between square brakets the indexes used to find a 
#     node with respect to the node "P" are displayed.
# (): in between brakets the name given to refer to the nodes
#     in the lectures as well as in the book with respect to 
#     the node "P" are displayed.   
#===================== Inputs =====================
# Geometric inputs
mI = 40 # number of mesh points X direction.
mJ = 40 # number of mesh points Y direction.
grid_type = 'non-equidistant' # this sets equidistant mesh sizing or non-equidistant
xL = 1 # length of the domain in X direction
yL = 0.5 # length of the domain in Y direction
# Solver inputs
nIterations  = 2000 # maximum number of iterations
resTolerance = 0.0001 # convergence criteria for residuals each variable
#====================== Code ======================
# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate (check Schematix above)
# Allocate all needed variables
nI = mI + 1                    # number of nodes in the X direction. Nodes 
                               # added in the boundaries
nJ = mJ + 1                    # number of nodes in the Y direction. Nodes 
                               # added in the boundaries
coeffsT = np.zeros((nI,nJ,5))  # coefficients for temperature
                               # E, W, N, S and P
S_U     = np.zeros((nI,nJ))    # source term for temperature
S_P     = np.zeros((nI,nJ))    # source term for temperature
T       = np.zeros((nI,nJ))    # temperature matrix
k       = np.zeros((nI,nJ))    # coefficient of conductivity
q       = np.zeros((nI,nJ,2))  # heat flux, first x and then y component
residuals = [] # List containing the value of the residual for each iteration
# Generate mesh and compute geometric variables
# Allocate all variables matrices
xCoords_M = np.zeros((mI,mJ)) # X coords of the mesh points
yCoords_M = np.zeros((mI,mJ)) # Y coords of the mesh points
xCoords_N = np.zeros((nI,nJ)) # X coords of the nodes
yCoords_N = np.zeros((nI,nJ)) # Y coords of the nodes
dxe_N     = np.zeros((nI,nJ)) # X distance to east node
dxw_N     = np.zeros((nI,nJ)) # X distance to west node
dyn_N     = np.zeros((nI,nJ)) # Y distance to north node
dys_N     = np.zeros((nI,nJ)) # Y distance to south node
dxe_F     = np.zeros((nI,nJ)) # X distance to east face
dxw_F     = np.zeros((nI,nJ)) # X distance to west face
dyn_F     = np.zeros((nI,nJ)) # Y distance to north face
dys_F     = np.zeros((nI,nJ)) # Y distance to south face
dx_CV     = np.zeros((nI,nJ)) # X size of the control volume
dy_CV     = np.zeros((nI,nJ)) # Y size of the control volume
if grid_type == 'equidistant':
    # Control volume size
    dx = xL/(mI - 1)
    dy = yL/(mJ - 1)
    # Fill the coordinates
    for i in range(mI):
        for j in range(mJ):
            # For the mesh points
            xCoords_M[i,j] = i*dx
            yCoords_M[i,j] = j*dy
            # For the nodes
            if i > 0:
                xCoords_N[i,j] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])
            if i == (mI-1) and j>0:
                yCoords_N[i+1,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
            if j >0:
                yCoords_N[i,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
            if j == (mJ-1) and i>0:
                xCoords_N[i,j+1] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])
            # Fill dx_CV and dy_CV
            if i>0:
                dx_CV[i,j] = xCoords_M[i,j] - xCoords_M[i-1,j]
            if j>0:
                dy_CV[i,j] = yCoords_M[i,j] - yCoords_M[i,j-1]
    
    for i in range(1,nI - 1):
        for j in range(1,nJ - 1):
            dxe_F[i,j] = (xCoords_N[i+1,j] - xCoords_N[i,j])/2
            dxw_F[i,j] = (xCoords_N[i,j] - xCoords_N[i-1,j])/2
            dyn_F[i,j] = (yCoords_N[i,j+1] - yCoords_N[i,j])/2
            dys_F[i,j] = (yCoords_N[i,j] - yCoords_N[i,j-1])/2
                
elif grid_type == 'non-equidistant':
    #One inflation region in x-dir, 2 in y-dir
    rx = 1.15
    ry = 1.15

    inflX = 10
    inflY = 10
    
    sx = ((1 - (1/rx)**(inflX+1))/(1-1/rx)-1) #Geometric sum of inflation layers x
    sy = ((1 - (1/ry)**(inflY+1))/(1-1/ry)-1) #Geometric sum of inflation layers y
    fx = nI - 2 - inflX #Nodes outside of inflation region x
    fy = nJ - 2 - 2 * inflY #Nodes outside of inflation region y

    dx = xL / (fx + sx)
    dy = yL / (fy + 2 * sy)

    for i in range(mI):
        for j in range(mJ):
            # For the mesh points
            if i < mI - inflX: #Inflation layer x
                xCoords_M[i,j] = i*dx
            else:
                xCoords_M[i,j] = dx * (1/rx)**(i-mI+inflX+1) + xCoords_M[i-1,j]

            if j == 0: #Special case to fix yCoords_M[i,j-1], where j is 1
                yCoords_M[i,j] = j*dy
            elif j <= inflY: #Inflation layer y
                yCoords_M[i,j] = dy * (1/ry)**(inflY-j+1) + yCoords_M[i,j-1]
            elif j < mJ - inflY:
                yCoords_M[i,j] = (j - inflY)*dy + yCoords_M[i,inflY]
            else: #Inflation layer x
                yCoords_M[i,j] = dy * (1/ry)**(j-mJ+inflY+1) + yCoords_M[i,j-1]

            # For the nodes
            if i > 0:
                xCoords_N[i,j] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])
            if i == (mI-1) and j>0:
                yCoords_N[i+1,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
            if j >0:
                yCoords_N[i,j] = 0.5*(yCoords_M[i,j] + yCoords_M[i,j-1])
            if j == (mJ-1) and i>0:
                xCoords_N[i,j+1] = 0.5*(xCoords_M[i,j] + xCoords_M[i-1,j])
            # Fill dx_CV and dy_CV
            if i>0:
                dx_CV[i,j] = xCoords_M[i,j] - xCoords_M[i-1,j]
            if j>0:
                dy_CV[i,j] = yCoords_M[i,j] - yCoords_M[i,j-1]

for i in range(1,nI - 1):
    for j in range(1,nJ - 1):
        dxe_F[i,j] = xCoords_M[i,j] - xCoords_N[i,j]
        dxw_F[i,j] = xCoords_N[i,j] - xCoords_M[i-1,j]
        dyn_F[i,j] = yCoords_M[i,j] - yCoords_N[i,j]
        dys_F[i,j] = yCoords_N[i,j] - yCoords_M[i,j-1]
    
xCoords_N[-1,:] = xL
yCoords_N[:,-1] = yL
# Fill dxe, dxw, dyn and dys
for i in range(1,nI - 1):
    for j in range(1,nJ - 1):
        dxe_N[i,j] = xCoords_N[i+1,j] - xCoords_N[i,j]
        dxw_N[i,j] = xCoords_N[i,j] - xCoords_N[i-1,j]
        dyn_N[i,j] = yCoords_N[i,j+1] - yCoords_N[i,j]
        dys_N[i,j] = yCoords_N[i,j] - yCoords_N[i,j-1]

# Update conductivity coefficient matrix, k, according to your case
k = 5 * (1 + 100 * xCoords_N / xL)

# Update source term matrix according to your case
S_U = -1.5*dx_CV*dy_CV
S_U_bd = np.zeros((nI,nJ))
#S_P = 0

#Define Dirichlet boundary conditions
for i in range(0,nI):
    j = 0
    T[i,j] = 15
    j = nJ-1
    T[i,j] = 15

for j in range(0, nJ):
    i = nI-1
    T[i,j] = 15 * np.cos(2*np.pi*yCoords_N[i,j]/yL)

# Initialize variable matrices and boundary conditions
# Looping
for iter in range(nIterations):
    
    # Compute coeffsT for all the nodes which are not boundary nodes
    ## Compute coefficients for nodes one step inside the domain
    ### First, north and south boundaries
    for i in range(2,nI-2):
        j = 1
        coeffsT[i,j,1] = (k[i,j] + (k[i+1,j]-k[i,j] / dxe_N[i,j]) * dxe_F[i,j]) * dy_CV[i,j] / dxe_N[i,j]#ae
        coeffsT[i,j,2] = (k[i,j] + (k[i,j]-k[i-1,j] / dxw_N[i,j]) * dxw_F[i,j]) * dy_CV[i,j] / dxw_N[i,j]#aw
        coeffsT[i,j,3] = (k[i,j] + (k[i,j+1]-k[i,j] / dyn_N[i,j]) * dyn_F[i,j]) * dx_CV[i,j] / dyn_N[i,j]#an
        coeffsT[i,j,4] = 0#as
        
        #Dirichlet
        S_U_bd[i,j] = k[i,j-1] * dx_CV[i,j] / dys_N[i,j] * T[i,j-1]
        S_P[i,j] = -k[i,j-1] * dx_CV[i,j] / dys_N[i,j]

        coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

        j = nJ-2
        coeffsT[i,j,1] = (k[i,j] + (k[i+1,j]-k[i,j] / dxe_N[i,j]) * dxe_F[i,j]) * dy_CV[i,j] / dxe_N[i,j]#ae
        coeffsT[i,j,2] = (k[i,j] + (k[i,j]-k[i-1,j] / dxw_N[i,j]) * dxw_F[i,j]) * dy_CV[i,j] / dxw_N[i,j]#aw
        coeffsT[i,j,3] = 0#an
        coeffsT[i,j,4] = (k[i,j] + (k[i,j]-k[i,j-1] / dys_N[i,j]) * dys_F[i,j]) * dx_CV[i,j] / dys_N[i,j]#as

        #Dirichlet
        S_U_bd[i,j] = k[i,j+1] * dx_CV[i,j] / dyn_N[i,j] * T[i,j+1]
        S_P[i,j] = -k[i,j+1] * dx_CV[i,j] / dyn_N[i,j]

        coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    ### Second, east and west boundaries
    for j in range(2,nJ-2):

        i = 1
        coeffsT[i,j,1] = (k[i,j] + (k[i+1,j]-k[i,j] / dxe_N[i,j]) * dxe_F[i,j]) * dy_CV[i,j] / dxe_N[i,j]#ae
        coeffsT[i,j,2] = 0 #aw
        coeffsT[i,j,3] = (k[i,j] + (k[i,j+1]-k[i,j] / dyn_N[i,j]) * dyn_F[i,j]) * dx_CV[i,j] / dyn_N[i,j]#an
        coeffsT[i,j,4] = (k[i,j] + (k[i,j]-k[i,j-1] / dys_N[i,j]) * dys_F[i,j]) * dx_CV[i,j] / dys_N[i,j]#as

        coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

        i = nI-2
        coeffsT[i,j,1] = 0#ae
        coeffsT[i,j,2] = (k[i,j] + (k[i,j]-k[i-1,j] / dxw_N[i,j]) * dxw_F[i,j]) * dy_CV[i,j] / dxw_N[i,j]#aw
        coeffsT[i,j,3] = (k[i,j] + (k[i,j+1]-k[i,j] / dyn_N[i,j]) * dyn_F[i,j]) * dx_CV[i,j] / dyn_N[i,j]#an
        coeffsT[i,j,4] = (k[i,j] + (k[i,j]-k[i,j-1] / dys_N[i,j]) * dys_F[i,j]) * dx_CV[i,j] / dys_N[i,j]#as

        #Dirichlet
        S_U_bd[i,j] = k[i+1,j] * dy_CV[i,j] / dxe_N[i,j] * T[i+1,j]
        S_P[i,j] = -k[i+1,j] * dy_CV[i,j] / dxe_N[i,j]

        coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    ## Compute coefficients for inner nodes
    for i in range(2,nI-2):
        for j in range(2,nJ-2):
            coeffsT[i,j,1] = (k[i,j] + (k[i+1,j]-k[i,j] / dxe_N[i,j]) * dxe_F[i,j]) * dy_CV[i,j] / dxe_N[i,j]#ae
            coeffsT[i,j,2] = (k[i,j] + (k[i,j]-k[i-1,j] / dxw_N[i,j]) * dxw_F[i,j]) * dy_CV[i,j] / dxw_N[i,j]#aw
            coeffsT[i,j,3] = (k[i,j] + (k[i,j+1]-k[i,j] / dyn_N[i,j]) * dyn_F[i,j]) * dx_CV[i,j] / dyn_N[i,j]#an
            coeffsT[i,j,4] = (k[i,j] + (k[i,j]-k[i,j-1] / dys_N[i,j]) * dys_F[i,j]) * dx_CV[i,j] / dys_N[i,j]#as

            coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    ## Compute coefficients corner nodes (one step inside)
    i=1
    j=1
    coeffsT[i,j,1] = (k[i,j] + (k[i+1,j]-k[i,j] / dxe_N[i,j]) * dxe_F[i,j]) * dy_CV[i,j] / dxe_N[i,j]#ae
    coeffsT[i,j,2] = 0#aw
    coeffsT[i,j,3] = (k[i,j] + (k[i,j+1]-k[i,j] / dyn_N[i,j]) * dyn_F[i,j]) * dx_CV[i,j] / dyn_N[i,j]#an
    coeffsT[i,j,4] = 0#as
     
    #Dirichlet for south
    S_U_bd[i,j] = k[i,j-1] * dx_CV[i,j] / dys_N[i,j] * T[i,j-1]
    S_P[i,j] = -k[i,j-1] * dx_CV[i,j] / dys_N[i,j]    

    coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    i=1
    j=nJ-2
    coeffsT[i,j,1] = (k[i,j] + (k[i+1,j]-k[i,j] / dxe_N[i,j]) * dxe_F[i,j]) * dy_CV[i,j] / dxe_N[i,j]#ae
    coeffsT[i,j,2] = 0#aw
    coeffsT[i,j,3] = 0#an
    coeffsT[i,j,4] = (k[i,j] + (k[i,j]-k[i,j-1] / dys_N[i,j]) * dys_F[i,j]) * dx_CV[i,j] / dys_N[i,j]#as
     
    #Dirichlet for north
    S_U_bd[i,j] = k[i,j+1] * dx_CV[i,j] / dyn_N[i,j] * T[i,j+1]
    S_P[i,j] = -k[i,j+1] * dx_CV[i,j] / dyn_N[i,j]

    coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    i=nI-2
    j=1
    coeffsT[i,j,1] = 0#ae
    coeffsT[i,j,2] = (k[i,j] + (k[i,j]-k[i-1,j] / dxw_N[i,j]) * dxw_F[i,j]) * dy_CV[i,j] / dxw_N[i,j]#aw
    coeffsT[i,j,3] = (k[i,j] + (k[i,j+1]-k[i,j] / dyn_N[i,j]) * dyn_F[i,j]) * dx_CV[i,j] / dyn_N[i,j]#an
    coeffsT[i,j,4] = 0#as
     
    #Dirichlet for south and east
    S_U_bd[i,j] = k[i,j-1] * dx_CV[i,j] / dys_N[i,j] * T[i,j-1] + k[i+1,j] * dy_CV[i,j] / dxe_N[i,j] * T[i+1,j]
    S_P[i,j] = -k[i,j-1] * dx_CV[i,j] / dys_N[i,j] - k[i+1,j] * dy_CV[i,j] / dxe_N[i,j]

    coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    i=nI-2
    j=nJ-2
    coeffsT[i,j,1] = 0#ae
    coeffsT[i,j,2] = (k[i,j] + (k[i,j]-k[i-1,j] / dxw_N[i,j]) * dxw_F[i,j]) * dy_CV[i,j] / dxw_N[i,j]#aw
    coeffsT[i,j,3] = 0#an
    coeffsT[i,j,4] = (k[i,j] + (k[i,j]-k[i,j-1] / dys_N[i,j]) * dys_F[i,j]) * dx_CV[i,j] / dys_N[i,j]#as
     
    #Dirichlet for north and east
    S_U_bd[i,j] = k[i,j+1] * dx_CV[i,j] / dyn_N[i,j] * T[i,j+1] + k[i+1,j] * dy_CV[i,j] / dxe_N[i,j] * T[i+1,j]
    S_P[i,j] = -k[i,j+1] * dx_CV[i,j] / dyn_N[i,j] - k[i+1,j] * dy_CV[i,j] / dxe_N[i,j]

    coeffsT[i,j,0] = coeffsT[i,j,1] + coeffsT[i,j,2] + coeffsT[i,j,3] + coeffsT[i,j,4] - S_P[i,j]#ap

    # Solve for T using Gauss-Seidel
    for i in range(1,nI-1):
        for j in range(1, nJ-1):
            RHS = coeffsT[i,j,1] * T[i+1,j] + \
                coeffsT[i,j,2] * T[i-1,j] + \
                coeffsT[i,j,3] * T[i,j+1] + \
                coeffsT[i,j,4] * T[i,j-1] + \
                S_U[i,j] + S_U_bd[i,j]
            T[i,j] = RHS / coeffsT[i,j,0]
    # Copy T to boundaries where homegeneous Neumann needs to be applied
    # bc 4 is Neumann
    T[0,:] = T[1,:]
    
    #Compute fluxes on boundaries
    flux1 = 0 #Heat flux on face 1
    flux2 = 0 #Heat flux on face 2
    flux3 = 0 #Heat flux on face 3
    flux4 = 0 #Heat flux on face 4
    for i in range(1,nI-1): #Integrate over b1, 3
        j = 1
        flux1 = flux1 + np.abs(-k[i,j-1]*(T[i,j]-T[i,j-1])/(dys_N[i,j])) * dx_CV[i,j]
        j = nJ-2
        flux3 = flux3 + np.abs(-k[i,j+1]*(T[i,j+1]-T[i,j])/(dyn_N[i,j])) * dx_CV[i,j]

    for j in range(1,nJ-1): #Integrate over b1, 3
        i = nI-2
        flux2 = flux2 + np.abs(-k[i+1,j]*(T[i+1,j]-T[i,j])/(dxe_N[i,j])) * dy_CV[i,j]

    Fluxtot = flux1 + flux2 + flux3 + flux4

    # Compute residuals (taking into account normalization)
    r = 0
    R = 0
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            R = R + np.abs(coeffsT[i,j,1] * T[i+1,j] + \
                coeffsT[i,j,2] * T[i-1,j] +\
                coeffsT[i,j,3] * T[i,j+1] +\
                coeffsT[i,j,4] * T[i,j-1] +\
                S_U[i,j] + S_U_bd[i,j] -\
                coeffsT[i,j,0] * T[i,j])
    r = R / Fluxtot
    
    residuals.append(r)
    
    print('iteration: %d\nresT = %.5e\n\n'  % (iter, residuals[-1]))
    
    #  Check convergence
    if resTolerance>residuals[-1]:
        break

# Compute heat fluxes
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        q[i,j,0] = -k[i,j]*(T[i+1,j]-T[i-1,j])/(dxe_N[i,j]+dxw_N[i,j])
        q[i,j,1] = -k[i,j]*(T[i,j+1]-T[i,j-1])/(dyn_N[i,j]+dys_N[i,j])

#Compute heat fluxes on boundaries
#TODO fix north, east boundary
for i in range(1,nI-1):
    j = 0
    q[i,j,1] = -k[i,j]*(T[i,j+1]-T[i,j])/(dys_N[i,j+1])
    j = nJ-1
    q[i,j,1] = -k[i,j]*(T[i,j]-T[i,j-1])/(dyn_N[i,j-1])

for j in range(1,nJ-1):
    i = nI-1
    q[i,j,0] = -k[i,j]*(T[i,j]-T[i-1,j])/(dxe_N[i-1,j])
    q[i,j,1] = -k[i,j]*(T[i,j+1]-T[i,j-1])/(dyn_N[i-1,j]+dys_N[i-1,j])

    
# Plotting section (these are some examples, more plots might be needed)
# Plot results
plt.figure()
# Plot mesh
plt.subplot(2,2,1)
#TODO check requirements of presentation
#plt.scatter(xCoords_M, yCoords_M, marker=".", color="b", label='meshgrid')
for i in range(0,mI):
    plt.plot(xCoords_M[i,[0,-1]],yCoords_M[i,[0,-1]], 'k-')
for j in range(0,mJ):
    plt.plot(xCoords_M[[0,-1],j],yCoords_M[[0,-1],j], 'k-')
plt.scatter(xCoords_N, yCoords_N, marker=".", color="r", label='nodes', s=1)
#plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Computational mesh')
plt.axis('equal')
# Plot temperature contour
plt.subplot(2,2,2)
plot_type = 'smooth'
if plot_type == 'rough':
    plt.pcolormesh(xCoords_N, yCoords_N, T)
elif plot_type == 'smooth':
    plt.contourf(xCoords_N, yCoords_N, T, 50)
plt.title('Temperature [??C]')
plt.colorbar()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
# Plot residual convergence
plt.subplot(2,2,3)
plt.plot(residuals)
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
plt.title('Residual')
# Plot heat fluxes
plt.subplot(2,2,4)
plt.quiver(xCoords_N,yCoords_N,q[:,:,0],q[:,:,1])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Heat flux')
plt.axis('equal')
plt.show(block=True)