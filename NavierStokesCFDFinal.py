# libraries 
# define environment
from math import ceil
import sys
import os
# numerical arrays
import numpy as np
# plotting functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from NSSimulation import *
from make_animation import create_animation

"""# Begin user inputs"""

############################DEFINE SPATIAL AND TEMPORAL PARAMETERS#############
length=4     # cavity width and length in 2D
breadth=4
colpts=65  # mesh dimension yields 66,049 values at each time step (257x257)
rowpts=65
countdown=150     # max time steps

###############################MISC############################################
CFL_number=0.4    # Do not touch this unless solution diverges (.8)
file_flag=1       # Keep 1 to print results to file
interval=100      # Record values in file per interval number of iterations
plot_flag=1       # Keep 1 to plot results at the end

###########################DEFINE PHYSICAL PARAMETERS##########################
# assume water as base case
rho= 1    # fluid density
mu=0.01   # fluid viscosity

##########################DEFINE INITIAL MOMENTUM PARAMETERS###################
u_in=1      # driving flow at lid
v_wall=0    # wall condition stagnant
p_out=0     # pressure uniform initially


########################CREATE SPACE OBJECT####################################

cavity=Space()
cavity.CreateMesh(rowpts,colpts)
cavity.SetDeltas(breadth,length)
water=Fluid(rho,mu)

"""# Boundary conditions
* pressure, velocity, boundary, noslip condition
"""

###########################VELOCITY############################################
flow=Boundary("D",u_in)
noslip=Boundary("D",v_wall)
zeroflux=Boundary("N",0)
############################PRESSURE###########################################
pressureatm=Boundary("D",p_out)
Re = rho*length/mu,rho*breadth/mu
Re = Re[0]        #since Re is a tuple
Re = round(Re)

# end simulation user inputs

"""# Run Simulation"""
import time
start = time.time()
run_my_simulation = False
if run_my_simulation:
    #############################INITIALIZATION####################################
    t=0
    i=0
    ############################THE RUN############################################
    print("######## Beginning FlowPy Simulation ########")
    print("#############################################")
    print("# Simulation time: {0:.2f}".format(countdown))
    print("# Mesh: {0} x {1}".format(colpts,rowpts))
    print("# Re/u: {0:.2f}\tRe/v:{1:.2f}".format(rho*length/mu,rho*breadth/mu))
    print("# Save outputs to text file: {0}".format(bool(file_flag)))
    MakeResultDirectory(wipe=True)

    while(t<countdown):
        sys.stdout.write("\rSimulation time left: {0:.2f}".format(countdown-t))
        sys.stdout.flush()

        CFL=CFL_number
        SetTimeStep(CFL,cavity,water)
        timestep=cavity.dt
        
        
        SetUBoundary(cavity,noslip,noslip,flow,noslip)
        SetVBoundary(cavity,noslip,noslip,noslip,noslip)
        SetPBoundary(cavity,zeroflux,zeroflux,pressureatm,zeroflux)
        GetStarredVelocities(cavity,water)
        
        
        SolvePressurePoisson(cavity,water,zeroflux,zeroflux,pressureatm,zeroflux)
        SolveMomentumEquation(cavity,water)
        
        SetCentrePUV(cavity)
        if(file_flag==1):
            WriteToFile(cavity,i,interval)

        t+=timestep
        i+=1
    
    cavity.write("current_cavity.db")
else:
    if os.path.exists("current_cavity.db"):
        cavity.read("current_cavity.db")
    else:
        print("Unable to open cavity db file...quitting.")
        sys.exit(1)

end = time.time()
print(f"\nActual Simulation Time: {end-start:.2f} seconds")



###########################END OF RUN##########################################

"""# Begin visualization of results

* compare to other simulations
* animate results of laminar flow for pressure and velocity in cavity
"""

#######################SET ARRAYS FOR PLOTTING#################################
x=np.linspace(0,length,colpts)
y=np.linspace(0,breadth,rowpts)
[X,Y]=np.meshgrid(x,y)              # mesh


u=cavity.u  # horizontal velocity in x
v=cavity.v  # vertical velocity in y
p=cavity.p  # pressure
u_c=cavity.u_c
v_c=cavity.v_c
p_c=cavity.p_c

#Ghia et al. Cavity test benchmark
# https://www.sciencedirect.com/science/article/pii/0021999182900584
y_g=[0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766]
u_g=[0,-0.08186,-0.09266,-0.10338,-0.14612,-0.24299,-0.32726,-0.17119,-0.11477,0.02135,0.16256,0.29093,0.55892,0.61756,0.68439,0.75837]

x_g=[0,0.0625,0.0703,0.0781,0.0983,0.1563,0.2266,0.2344,0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688]
v_g=[0,0.1836,0.19713,0.20920,0.22965,0.28124,0.30203,0.30174,0.05186,-0.38598,-0.44993,-0.23827,-0.22847,-0.19254,-0.15663,-0.12146]

y_g=[breadth*y_g[i] for i in range(len(y_g))]
x_g=[length*x_g[i] for i in range(len(x_g))]

######################EXTRA PLOTTING CODE BELOW################################

if(plot_flag==1):
    plt.figure(figsize=(10,10))
    plt.contourf(X,Y,p_c,cmap=cm.viridis)
    plt.colorbar()
    plt.quiver(X,Y,u_c,v_c)
    plt.title("Velocity and Pressure Plot")

    plt.figure()
    plt.plot(y,u_c[:,int(np.ceil(colpts/2))],"darkblue", label='Simulation data')
    plt.plot(y_g,u_g,"rx", label='Ghia Benchmark data')   # Ghia data
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xlabel("Vertical distance along center")
    plt.ylabel("Horizontal velocity")
    plt.title("Re 400 (Base Case) Plot 1")
    
    plt.figure()
    plt.plot(x,v_c[int(np.ceil(rowpts/2)),:],"darkblue", label='Simulation data')
    plt.plot(x_g,v_g,"rx",label='Ghia Benchmark data')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Horizontal distance along center")
    plt.ylabel("Vertical velocity")
    plt.title("Re 400 (Base Case) Plot 2")
    # plt.show()

# Mid Slice data for steady state at chosen Re 
resolution = 4
midSliceX = v_c[(rowpts-1)//2, resolution:colpts-resolution:resolution] 
midSliceY = u_c[resolution:rowpts-resolution:resolution, (colpts-1)//2]  

# Saving the data to a file 
dir_path=os.path.join(os.getcwd(),"Result")
filenameX = f"XDataRe{Re}.txt"
filenameY = f"YDataRe{Re}.txt"
filenameBoth = f"BothDataRe{int(Re)}.txt"
pathX =os.path.join(dir_path, filenameX)
pathY =os.path.join(dir_path, filenameY)
pathBoth = os.path.join(dir_path, filenameBoth)
savefileX = np.savetxt(pathX, midSliceX, delimiter=',')
savefileY = np.savetxt(pathY, midSliceY, delimiter=',')
# Adding both files together with the Reynolds Number in the first column  
ReArray = np.array([Re]) 
savefileBoth = np.savetxt(pathBoth, ReArray, delimiter=',')
savefileBoth = open(pathBoth,'ab') 
np.savetxt(savefileBoth, midSliceX, delimiter=',')
np.savetxt(savefileBoth, midSliceY, delimiter=',')
savefileBoth.close()



# Writing into complete data table 
import csv
cwdir=os.getcwd()
fdata = "Documents/NavierStokesFinalData.csv"
finaldata = os.path.join(cwdir, fdata)
if os.path.exists(finaldata):
    file = open(finaldata, 'a+')
    data = np.genfromtxt(pathBoth)
    datalist = data.tolist()
    with file: 
        write = csv.writer(file)
        write.writerow(datalist)
else:
    file = open(finaldata, 'w')
    data = np.genfromtxt(pathBoth)
    datalist = data.tolist()
    with file: 
        write = csv.writer(file)
        write.writerow(['Reynolds Number', 'v1', 'v2', 'v3', 'v4'])
        write.writerow(datalist)


#Go to the Result directory
cwdir=os.getcwd()
dir_path=os.path.join(cwdir,"Result")
os.chdir(dir_path)

#Go through files in the directory
filenames=[]
iterations=[]
for root,dirs,files in os.walk(dir_path):
    for datafile in files:
        if "PUV" in datafile:
            filenames.append(datafile)
            no_ext_file=datafile.replace(".txt","").strip()
            iter_no=int(no_ext_file.split("V")[-1])
            iterations.append(iter_no)

# this creates the .mp4 file that shows the lid driven cavity's development to steady state.
RunAnimation = False
if RunAnimation:
    create_animation(dir_path, iterations, rowpts, colpts, length, breadth)