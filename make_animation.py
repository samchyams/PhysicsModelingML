#Create blank figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys


#Define function to read datafile
def read_datafile(dir_path, rowpts, colpts, iteration):
    filename=f"PUV{iteration}.txt"
    filepath=os.path.join(dir_path,filename)
    arr=np.loadtxt(filepath,delimiter="\t")
    rows,cols=arr.shape
    p_p=np.zeros((rowpts,colpts))
    u_p=np.zeros((rowpts,colpts))
    v_p=np.zeros((rowpts,colpts))
    p_arr=arr[:,0]
    u_arr=arr[:,1]
    v_arr=arr[:,2]
    
    p_p=p_arr.reshape((rowpts,colpts))
    u_p=u_arr.reshape((rowpts,colpts))
    v_p=v_arr.reshape((rowpts,colpts))
    
    return p_p,u_p,v_p


def create_animation(dir_path, iterations, rowpts, colpts, length, breadth):
    #Discern the final iteration and interval
    initial_iter=np.amin(iterations)            
    final_iter=np.amax(iterations)
    inter=(final_iter - initial_iter)/(len(iterations)-1)
    number_of_frames=len(iterations)#int(final_iter/inter)+1
    sorted_iterations=np.sort(iterations)

    fig=plt.figure(num=1,figsize=(16,8))
    ax=plt.axes(xlim=(0,length),ylim=(0,breadth))

    x=np.linspace(0,length,colpts)
    y=np.linspace(0,breadth,rowpts)
    [X,Y]=np.meshgrid(x,y)
    #print(X.shape, Y.shape)

    #Determine indexing for streamplotf
    index_cut_x=int(colpts/10)
    index_cut_y=int(rowpts/10)

    def animate_init_func():
        return []

    def animate(i):
        nframes = len(sorted_iterations)
        iteration=sorted_iterations[i]
        sys.stdout.write(f"\rFrame {i} of {nframes-1}")
        sys.stdout.flush()
        p_p,u_p,v_p=read_datafile(dir_path, rowpts, colpts, iteration)
        ax.clear()
        ax.set_xlim([0,length])
        ax.set_ylim([0,breadth])
        ax.set_xlabel("$x$",fontsize=12)
        ax.set_ylabel("$y$",fontsize=12)
        ax.set_title(f"Frame No: {i}")
        cont=ax.contourf(X,Y,p_p)
        stream=ax.streamplot(X[::index_cut_y,::index_cut_x],Y[::index_cut_y,::index_cut_x],u_p[::index_cut_y,::index_cut_x],v_p[::index_cut_y,::index_cut_x],color="k")
        return cont,stream

    print("######## Making FlowPy Animation ########")
    print("#########################################")
    anim=animation.FuncAnimation(fig,animate,init_func=animate_init_func,frames=number_of_frames,interval=50,blit=False)
    movie_path=os.path.join(dir_path,"FluidFlowAnimation.mp4")
    anim.save(movie_path)
    print("\nAnimation saved as FluidFlowAnimation.mp4 in Result")
    #plt.clf() # to clear out animation from plt
    plt.close(1)

    # For plotting the sampled points
    filename="PUV27500.txt"
    filepath=os.path.join(dir_path,filename)
    arr=np.loadtxt(filepath,delimiter="\t")
    p_arr=arr[:,0]
    p_p=p_arr.reshape((rowpts,colpts))
    resolution = 4
    mid = np.full(63,(rowpts-1)//2)
    scatter = np.arange(resolution, (colpts-resolution), resolution)
    ax.contourf(X,Y,p_p)
    ax.scatter(mid, scatter,color= 'blueviolet')
    ax.scatter(scatter, mid,color= 'blueviolet')