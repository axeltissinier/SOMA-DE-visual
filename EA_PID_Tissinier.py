"""
This program aims to find the best PID controller parameters for a chosen
transfert function, that minimises overshoot, rising time and settling time,
with acceptance values and punishment factors.

This program requires Tkinter to work. Tkinter is the basic GUI tool for
Python and is installed by default on most distributions of Python. If your
computer runs on a distribution of Linux, your distribution may not have Tkinter,
you can install it with this command:
+-+-+ sudo apt-get install python3-tk

@author: Axel Tissinier, November 2021
axel.tissinier@ipsa.fr
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk
from tkinter.font import Font
import time
from scipy import signal
import warnings

#%% Subframe class & functions | This is the core of the program as each subframe
#   is an experience, and that the objective is to compare their results
#   depending on their parameters.

class Subframe:
    # Class initialization (GUI design)
    def __init__(self,name):
        global Data
        Data.append(self) # So that we can access data from outside the frame
        
        # General variables
        self.step_nb = 0
        self.scatter = []
        self.best_fit_value=[]
        self.time_elapsed=0
        self.is_SOMA=True
        
        # Main frame (of subframe)
        self.root = Tk()
        self.root.title(name)
        Button(self.root, text="Reset",command=self.reset_plot).grid(column=0, row=0, padx=3, pady=4)
        Button(self.root, text="Step",command=self.step).grid(column=1, row=0, padx=10, pady=4)
        
        # Run button
        self.run_button_frame = Frame(self.root)
        self.run_button_frame.grid(column=2,row=0)
        Button(self.run_button_frame, text="Run",command=self.run).grid(column=0, row=0, padx=3, pady=4)
        Label(self.run_button_frame, text="for").grid(column=1, row=0, pady=5)
        Label(self.run_button_frame, text="iterations.").grid(column=5, row=0, pady=5)
        self.itermax = Entry(self.run_button_frame, textvariable=StringVar(), width=10)
        self.itermax.grid(column=2, row=0, padx=3)
        self.itermax.insert(0, '10')
        
        # Evolution button (change of graph)
        self.evo_graph_on=False
        self.evo_button = Button(self.root, text="Show evolution graph",width=18, command=self.evo_graph)
        self.evo_button.grid(column=4, row=0, padx=8, pady=4, columnspan=2)
        
        
        # TF & PID parameters frame
        self.PID_parameters = LabelFrame(self.root, text='TF & PID Parameters')
        self.PID_parameters.grid(column=6, row=0, padx=10, pady=10, rowspan=2)
        
        # Transfert Function
        Label(self.PID_parameters, text="Transfert\nfunction:").grid(column=0, row=0, padx=2, pady=3, rowspan=2)
        Label(self.PID_parameters, text="Numerator").grid(column=1, row=0, padx=2, pady=3)
        self.entry_num = Entry(self.PID_parameters, textvariable=StringVar(), width=10)
        self.entry_num.grid(column=1, row=1, padx=3)
        self.entry_num.insert(0, '1')
        Label(self.PID_parameters, text="Denominator").grid(column=2, row=0, padx=2, pady=3)
        self.entry_den = Entry(self.PID_parameters, textvariable=StringVar(), width=10)
        self.entry_den.grid(column=2, row=1, padx=3)
        self.entry_den.insert(0, '1; 2; 1')
        Label(self.PID_parameters, text="Closed loop").grid(column=0, row=2, padx=2, pady=3)
        self.closed_loop = ttk.Combobox(self.PID_parameters, textvariable=StringVar(), width=6)
        self.closed_loop['values'] = ('Yes','No')
        self.closed_loop.state(["readonly"])
        self.closed_loop.grid(column=1, row=2,pady=10)
        self.closed_loop.set('Yes')
        
        # PID parameters
        Label(self.PID_parameters, text=" ").grid(column=0, row=4)
        Label(self.PID_parameters, text="Parameters").grid(column=0, row=5, padx=2, pady=3)
        Label(self.PID_parameters, text="Maximum\naccepted").grid(column=1, row=5, padx=2, pady=3)
        Label(self.PID_parameters, text="Punishment\nfactor").grid(column=2, row=5, padx=2, pady=3)
        
        Label(self.PID_parameters, text="Overshoot (%)").grid(column=0, row=6, padx=2, pady=3)
        self.entry_max_os = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_max_os.grid(column=1, row=6, padx=3)
        self.entry_max_os.insert(0, '10')
        self.entry_pun_os = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_pun_os.grid(column=2, row=6, padx=3)
        self.entry_pun_os.insert(0, '5')
        
        Label(self.PID_parameters, text="Rise time (s)").grid(column=0, row=7, padx=2, pady=3)
        self.entry_max_tr = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_max_tr.grid(column=1, row=7, padx=3)
        self.entry_max_tr.insert(0, '0.5')
        self.entry_pun_tr = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_pun_tr.grid(column=2, row=7, padx=3)
        self.entry_pun_tr.insert(0, '1')
        
        Label(self.PID_parameters, text="Settling time (s)").grid(column=0, row=8, padx=2, pady=3)
        self.entry_max_ts = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_max_ts.grid(column=1, row=8, padx=3)
        self.entry_max_ts.insert(0, '2')
        self.entry_pun_ts = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_pun_ts.grid(column=2, row=8, padx=3)
        self.entry_pun_ts.insert(0, '3')
        
        Label(self.PID_parameters, text="Steady-state\nerror").grid(column=0, row=9, padx=2, pady=3)
        self.entry_max_ess = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_max_ess.grid(column=1, row=9, padx=3)
        self.entry_max_ess.insert(0, '0')
        self.entry_pun_ess = Entry(self.PID_parameters, textvariable=StringVar(), width=8)
        self.entry_pun_ess.grid(column=2, row=9, padx=3)
        self.entry_pun_ess.insert(0, '10')
        
        # Algorithm parameter frame
        self.parameters = LabelFrame(self.root, text='EA Parameters')
        self.parameters.grid(column=7, row=0, padx=5, pady=5, rowspan=2)
        
        # Algorithm choice
        Label(self.parameters, text="Algorithm").grid(column=0, row=0, padx=4, pady=5)
        self.alg = ttk.Combobox(self.parameters, textvariable=StringVar(), width=8)
        self.alg['values'] = ('SOMA','DE')
        self.alg.state(["readonly"])
        self.alg.grid(column=1, row=0, padx=5, pady=5)
        self.alg.set('SOMA')
        self.alg.bind("<<ComboboxSelected>>", self.alg_change)
        
        # Population
        Label(self.parameters, text="Population").grid(column=0, row=1, padx=4,pady=5)
        self.population = Entry(self.parameters, textvariable=StringVar(), width=10)
        self.population.grid(column=1, row=1, padx=4)
        self.population.insert(0, 10)
        
        # Warning text if necessary
        self.warn_label = Label(self.parameters, text="")
        self.warn_label.grid(column=0, row=10, padx=4, pady=5, columnspan=2)
        
        
        # Algorithm specific parameters
        # By default, the SOMA algorithm is selected, thus we will hide the DE frame
        self.SOMA_frame = Frame(self.parameters)
        self.SOMA_frame.grid(column=0, row=2, columnspan=2)
        
        # Startegy selection
        Label(self.SOMA_frame, text="Strategy").grid(column=0, row=3, padx=4, pady=5)
        self.strategy = ttk.Combobox(self.SOMA_frame, textvariable=StringVar(), width=15)
        self.strategy['values'] = ('AllToOne','AllToAll','AllToOneRand','AllToAllAdaptative','T3A')
        self.strategy.state(["readonly"])
        self.strategy.grid(column=1, row=3, padx=5, pady=5)
        self.strategy.set('AllToOne')
            
        # Scale between the distance to the leader and the maximum
        Label(self.SOMA_frame, text="Path Length").grid(column=0, row=4, padx=4, pady=5)
        self.path_len = Entry(self.SOMA_frame, textvariable=StringVar(), width=10)
        self.path_len.grid(column=1, row=4, padx=2)
        self.path_len.insert(0, '2')
            
        # Step between jumps, a % can be added at the end to make it a percent of path length
        Label(self.SOMA_frame, text="Step").grid(column=0, row=5, padx=4, pady=5)
        self.step = Entry(self.SOMA_frame, textvariable=StringVar(), width=10)
        self.step.grid(column=1, row=5, padx=2)
        self.step.insert(0, '5%')
            
        # Use PRT or not
        Label(self.SOMA_frame, text="Use PRT").grid(column=0, row=6, padx=4, pady=5)
        self.prt = ttk.Combobox(self.SOMA_frame, textvariable=StringVar(), width=6)
        self.prt['values'] = ('Yes','No')
        self.prt.state(["readonly"])
        self.prt.grid(column=1, row=6, padx=5, pady=5)
        self.prt.set('No')
            
            
        # DE frame
        self.DE_frame = Frame(self.parameters)
        self.DE_frame.grid(column=0, row=3, columnspan=2)
        
        # Strategy selection
        Label(self.DE_frame, text="Strategy").grid(column=0, row=3, padx=4, pady=5)
        self.DE_strategy = ttk.Combobox(self.DE_frame, textvariable=StringVar(), width=12)
        self.DE_strategy['values'] = ('rand 1','rand 2','best 1','best 2','current-to-best 1','current-to-rand 1','rand-to-best 1')
        self.DE_strategy.state(["readonly"])
        self.DE_strategy.grid(column=1, row=3, padx=5, pady=5)
        self.DE_strategy.set('rand 1')
            
        # Mutation scalar F
        Label(self.DE_frame, text="Mutation Fxc").grid(column=0, row=4, padx=5,pady=5)
        self.mutation = Entry(self.DE_frame, textvariable=StringVar(), width=10)
        self.mutation.grid(column=1, row=4, padx=2)
        self.mutation.insert(0, '0.5')
        
        # Treshold crossing CR
        Label(self.DE_frame, text="Crossover CR").grid(column=0, row=5, padx=5,pady=5)
        self.crossover = Entry(self.DE_frame, textvariable=StringVar(), width=10)
        self.crossover.grid(column=1, row=5, padx=2)
        self.crossover.insert(0, '0.9')
        
        # Parameter lambda required in rand-to-best strategies
        Label(self.DE_frame, text="Lambda/K/Fcr").grid(column=0, row=6, padx=5,pady=5)
        self.lbda = Entry(self.DE_frame, textvariable=StringVar(), width=10)
        self.lbda.grid(column=1, row=6, padx=2)
        self.lbda.insert(0, '0.5')
        
        # Hiding DE frame
        self.DE_frame.grid_remove()
        
        
        # Graph frame (step response graph)
        self.graph= Frame(self.root)
        self.graph.grid(column=0, row=1, columnspan=6, rowspan=2)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph)  # A tk Drawing Area
        self.canvas.draw()
        
        self.ax = self.fig.add_subplot(111)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        
        # Result frame (Shows results)
        self.result_frame = LabelFrame(self.root, text='Results')
        self.result_frame.grid(column=6, row=2, padx=5, pady=5, columnspan=2)
        
        Label(self.result_frame, text="Current step:").grid(column=0, row=0, padx=3)
        self.result_step = Label(self.result_frame, text="0")
        self.result_step.grid(column=1, row=0, padx=3)
        Label(self.result_frame, text="Stable:").grid(column=0, row=1, padx=3)
        self.result_stable = Label(self.result_frame, text="Yes")
        self.result_stable.grid(column=1, row=1, padx=3)
        Label(self.result_frame, text="Cost:").grid(column=0, row=2, padx=3)
        self.result_cost = Label(self.result_frame, text="0")
        self.result_cost.grid(column=1, row=2, padx=3)
        Label(self.result_frame, text="Time elapsed:").grid(column=0, row=3, padx=3)
        self.result_time = Label(self.result_frame, text="0")
        self.result_time.grid(column=1, row=3, padx=3)
        
        Label(self.result_frame, text="    ").grid(column=2, row=0, padx=3)
        Label(self.result_frame, text="Overshoot:").grid(column=3, row=0, padx=3)
        self.result_overshoot = Label(self.result_frame, text="0")
        self.result_overshoot.grid(column=4, row=0, padx=3)
        Label(self.result_frame, text="Rise time:").grid(column=3, row=1, padx=3)
        self.result_rise_time = Label(self.result_frame, text="0")
        self.result_rise_time.grid(column=4, row=1, padx=3)
        Label(self.result_frame, text="Settling time:").grid(column=3, row=2, padx=3)
        self.result_settling_time = Label(self.result_frame, text="0")
        self.result_settling_time.grid(column=4, row=2, padx=3)
        Label(self.result_frame, text="Steady-state\nerror:").grid(column=3, row=3, padx=3)
        self.result_ess = Label(self.result_frame, text="0")
        self.result_ess.grid(column=4, row=3, padx=3)
        
        Label(self.result_frame, text="    ").grid(column=5, row=0, padx=3)
        Label(self.result_frame, text="Kp:").grid(column=6, row=0, padx=3)
        self.result_Kp = Label(self.result_frame, text="1")
        self.result_Kp.grid(column=7, row=0, padx=3)
        Label(self.result_frame, text="Ki:").grid(column=6, row=1, padx=3)
        self.result_Ki = Label(self.result_frame, text="0")
        self.result_Ki.grid(column=7, row=1, padx=3)
        Label(self.result_frame, text="Kd:").grid(column=6, row=2, padx=3)
        self.result_Kd = Label(self.result_frame, text="0")
        self.result_Kd.grid(column=7, row=2, padx=3)
        Label(self.result_frame, text="N:").grid(column=6, row=3, padx=3)
        self.result_N = Label(self.result_frame, text="10")
        self.result_N.grid(column=7, row=3, padx=3)
        
        # Bottom part
        # Button that opens the information frame
        Button(self.root, text='Help & information', width=18, command=information_frame, bg='yellow').grid(column=0, row=3, padx=3, pady=5, columnspan=2)
        # Button to plot 10 evolutions to compare
        Button(self.root, text="Plot 10 evolutions", width=16,command=self.plot_10).grid(column=2, row=3, padx=3, pady=5, columnspan=2)
        # Button that creates a 3D representation of space solutions
        Button(self.root, text="Plot 3d space of\nsolutions (lag warning)", width=20,command=self.plot_3d_space).grid(column=4, row=3, padx=3, pady=5, columnspan=2)
        # Button that creates a new frame
        Button(self.root, text='New frame', width=10, command=new_frame).grid(column=6, row=3, pady=5)
        # Button that closes all frames
        Button(self.root, text='Close all', width=10, command=close_all).grid(column=7, row=3, padx=3,pady=5)
        
        self.reset_plot() # Reset plot to have a visual of the untuned step response
        
        self.root.mainloop() # End of frame loop
        
    #%% Function that compute multiple generations and their results
    def run(self):
        try: # Recovering max iteration parameter
            iterations = int(self.itermax.get())
            if iterations!=float(self.itermax.get()) or iterations<2: # Testing if it's an integer greater than 1
                self.warn_label.configure(text="Iteration must be an integer\ngreater than 1.", fg='#e60000')
                return
            self.warn_label.configure(text="")
        except:
            self.warn_label.configure(text="Iteration must be an integer\ngreater than 1.", fg='#e60000')
            return
        
        # Calling computing function with a number of iteration (timed)
        tic = time.time()
        completed = self.compute(iterations)
        
        if not completed: # If completed is False, it means there has been an error
            # Most likely in retrieving parameters (help should be displayed)
            return
        toc = time.time()
        self.time_elapsed+=toc-tic # Add time
        
        self.plot_step() # Plotting values
        self.display_results() # Refresh numbers on the result frame
        
        
    #%% Function that compute one generation and its result
    def step(self):
        tic = time.time()  # Computing function once (timed)
        completed = self.compute(1)
        if not completed: # If completed is False, it means there has been an error
            # Most likely in retrieving parameters (help should be displayed)
            return
        toc = time.time()
        self.time_elapsed+=toc-tic # Add time
        
        self.plot_step() # Plotting values
        self.display_results() # Refresh numbers on the result frame
        
    #%% Function that plots 10 evolutions of the function with its current parameters
    # in order to compare them. Number of evolutions and number of steps can be changed
    def plot_10(self):
        plot_nb=10
        step_computed=10
        
        # Saving values if there are
        save=False
        if self.step_nb!=0:
            save = True
            save_stable=self.is_stable
            save_points=self.points
            save_best_fit=self.best_fit
            save_best_fit_value=self.best_fit_value
            save_best_fit_time=self.best_fit_time
            save_best_fit_values=self.best_fit_values
            save_best_fit_result=self.best_fit_result
            
        # Iteration loop
        tic = time.time()
        for i in range(plot_nb):
            self.step_nb=0
            self.best_fit_value=[]
            # Calling computing function with a number of iteration (timed)
            completed = self.compute(step_computed)
            if not completed: # If completed is False, it means there has been an error
                # Most likely in retrieving parameters (help should be displayed)
                return
            # Data recovering and grouping
            y=[]
            for j in range(len(self.best_fit_value)):
                y.append(min(self.best_fit_value[:j+1]))
            if i==0:
                evolutions=np.array(y)
            else:
                y=np.array(y)
                evolutions=np.block([[evolutions],[y]])
        
        toc = time.time()
        mean_time = round((toc-tic)/plot_nb, 2)
        
        if save:
            self.is_stable=save_stable
            self.points=save_points
            self.best_fit=save_best_fit
            self.best_fit_value=save_best_fit_value
            self.best_fit_time=save_best_fit_time
            self.best_fit_values=save_best_fit_values
            self.best_fit_result=save_best_fit_result
        else:
            self.reset_plot()
        
        # Taking transpose to plot easily
        evolutions=evolutions.T
        
        # Detroy previous window if it exists
        try:
            self.root_plot.destroy()
        except:
            True
        
        # New window
        self.root_plot = Tk()
        self.root_plot.title(self.root.title()+' - '+str(plot_nb)+' evolutions of '+str(step_computed)+' iterations - '+str(mean_time)+'s to compute.')
        graph= Frame(self.root_plot)
        graph.pack()
        
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=graph)  # A tk Drawing Area
        canvas.draw()
        
        ax = fig.add_subplot(111)
        toolbar = NavigationToolbar2Tk(canvas, graph)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        
        # plotting evolution
        ax.plot(evolutions)
        canvas.draw()
        
        self.root_plot.mainloop()
        
    #%% Function that displays the space of possible solutions (lag warning)
    def plot_3d_space(self):
        fig = plt.figure(1)
        # Computing
        # Recovering PID parameters
        completed=self.recover_PID_param()
        if not completed:
            return
        
        Kp = np.linspace(1, 100, num=20)
        Ki = np.linspace(0, 50, num=20)
        Kd = np.linspace(0, 100, num=20)
        N = np.linspace(2, 22, num=20)
        
        xv, yv = np.meshgrid(Kp, Ki)
        zv = np.zeros(np.shape(xv))
        
        for i in range(20):
            for j in range(20):
                stability, results, time, values = self.get_fit([xv[i][j],yv[i][j],1,10])
                cost=results.flatten()
                zv[i][j]= cost[4]
                
        ax = fig.add_subplot(3,2,1, projection='3d')    
        ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
        ax.set_xlabel('Kp')
        ax.set_ylabel('Ki')
        
        xv, yv = np.meshgrid(Kp, Kd)
        zv = np.zeros(np.shape(xv))
        for i in range(20):
            for j in range(20):
                stability, results, time, values = self.get_fit([xv[i][j],1,yv[i][j],10])
                cost=results.flatten()
                zv[i][j]= cost[4]
                
        ax = fig.add_subplot(3,2,2, projection='3d') 
        ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
        ax.set_xlabel('Kp')
        ax.set_ylabel('Kd')
        
        xv, yv = np.meshgrid(Kp, N)
        zv = np.zeros(np.shape(xv))
        for i in range(20):
            for j in range(20):
                stability, results, time, values = self.get_fit([xv[i][j],1,1,yv[i][j]])
                cost=results.flatten()
                zv[i][j]= cost[4]
                
        ax = fig.add_subplot(3,2,3, projection='3d') 
        ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
        ax.set_xlabel('Kp')
        ax.set_ylabel('N')
        
        xv, yv = np.meshgrid(Ki, Kd)
        zv = np.zeros(np.shape(xv))
        for i in range(20):
            for j in range(20):
                stability, results, time, values = self.get_fit([1,xv[i][j],yv[i][j],10])
                cost=results.flatten()
                zv[i][j]= cost[4]
                
        ax = fig.add_subplot(3,2,4, projection='3d') 
        ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
        ax.set_xlabel('Ki')
        ax.set_ylabel('Kd')
        
        xv, yv = np.meshgrid(Ki, N)
        zv = np.zeros(np.shape(xv))
        for i in range(20):
            for j in range(20):
                stability, results, time, values = self.get_fit([1,xv[i][j],1,yv[i][j]])
                cost=results.flatten()
                zv[i][j]= cost[4]
                
        ax = fig.add_subplot(3,2,5, projection='3d') 
        ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
        ax.set_xlabel('Ki')
        ax.set_ylabel('N')
         
        xv, yv = np.meshgrid(Kd, N)
        zv = np.zeros(np.shape(xv))
        for i in range(20):
            for j in range(20):
                stability, results, time, values = self.get_fit([1,1,xv[i][j],yv[i][j]])
                cost=results.flatten()
                zv[i][j]= cost[4]
                
        ax = fig.add_subplot(3,2,6, projection='3d') 
        ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
        ax.set_xlabel('Kd')
        ax.set_ylabel('N')
        
        # Completed
        return True
    
    #%% Function that recovers the PID parameters, because they need to be
    # recovered in both the reset_plot() and compute() function
    def recover_PID_param(self):
        # Recovering tf numerator
        try:
            string=self.entry_num.get()
            ind=string.find(';')
            if ind==-1:
                x=float(string)
                self.num = np.array([x])
            else:
                x=float(string[:ind])
                self.num=[x]
                ind1=ind
                ind2= string.find(';', ind+1)
                while ind2!=-1:
                    x=float(string[ind1+1:ind2])
                    self.num.append(x)
                    ind1=ind2
                    ind2=string.find(';', ind1+1)
                x=float(string[ind1+1:])
                self.num.append(x)
                self.num = np.array(self.num) # Transforming to numpy array for computing
        except:
            self.warn_label.configure(text="Numerator must be a list of\nreals separated with \";\" like\n1; 2; 3 which represents factors\nof powers of s, from n to 0.", fg='#e60000')
            return False
        
        # Recovering tf denominator
        try:
            string=self.entry_den.get()
            ind=string.find(';')
            if ind==-1:
                x=float(string)
                self.den = np.array([x])
            else:
                x=float(string[:ind])
                self.den=[x]
                ind1=ind
                ind2= string.find(';', ind+1)
                while ind2!=-1:
                    x=float(string[ind1+1:ind2])
                    self.den.append(x)
                    ind1=ind2
                    ind2=string.find(';', ind1+1)
                x=float(string[ind1+1:])
                self.den.append(x)
                self.den = np.array(self.den) # Transforming to numpy array for computing
        except:
            self.warn_label.configure(text="Denominator must be a list of\nreals separated with \";\" like\n1; 2; 3 which represents factors\nof powers of s, from n to 0.", fg='#e60000')
            return False
        
        # Recovering closed loop bolean
        cl_loop=self.closed_loop.current()
        if cl_loop==0:
            self.cl_loop=True
        else:
            self.cl_loop=False
        
        # Recovering max overshoot
        try:
            os=float(self.entry_max_os.get())
            if os<0:
                self.warn_label.configure(text="Overshoot is in percent,\nand must be greater\nthan or equal to 0.", fg='#e60000')
                return False
            self.max_os=os
        except:
            self.warn_label.configure(text="Overshoot is in percent,\nand must be greater\nthan or equal to 0.", fg='#e60000')
            return False
            
        # Recovering overshoot punishment factor
        try:
            os=float(self.entry_pun_os.get())
            if os<0:
                self.warn_label.configure(text="Overshoot punishment must be\na real greater than or equal to 0.", fg='#e60000')
                return False
            self.pun_os=os
        except:
            self.warn_label.configure(text="Overshoot punishment must be\na real greater than or equal to 0.", fg='#e60000')
            return False
            
        # Recovering max rise time
        try:
            tr=float(self.entry_max_tr.get())
            if tr<0:
                self.warn_label.configure(text="Rising time must be\na real greater than or\nequal to 0 in seconds.", fg='#e60000')
                return False
            self.max_tr=tr
        except:
            self.warn_label.configure(text="Rising time must be\na real greater than or\nequal to 0 in seconds.", fg='#e60000')
            return False
            
        # Recovering rise time punishment factor
        try:
            tr=float(self.entry_pun_tr.get())
            if tr<0:
                self.warn_label.configure(text="Rising time punishment\nmust be a real greater than\nor equal to 0.", fg='#e60000')
                return False
            self.pun_tr=tr
        except:
            self.warn_label.configure(text="Rising time punishment\nmust be a real greater than\nor equal to 0.", fg='#e60000')
            return False
            
        # Recovering max settling time
        try:
            ts=float(self.entry_max_ts.get())
            if tr<0:
                self.warn_label.configure(text="Settling time must be\na real greater than or\nequal to 0 in seconds.", fg='#e60000')
                return False
            self.max_ts=ts
        except:
            self.warn_label.configure(text="Settling time must be\na real greater than or equal\nto 0 in seconds.", fg='#e60000')
            return False
            
        # Recovering settling time punishment 
        try:
            ts=float(self.entry_pun_ts.get())
            if ts<0:
                self.warn_label.configure(text="Settling time punishment\nmust be a real greater\nor equal to 0.", fg='#e60000')
                return False
            self.pun_ts=ts
        except:
            self.warn_label.configure(text="Settling time punishment\nmust be a real greater than\nor equal to 0.", fg='#e60000')
            return False
        
        # Recovering max steady-state error
        try:
            ess=float(self.entry_max_ess.get())
            if tr<0:
                self.warn_label.configure(text="Steady-state error must be\na real greater than or\nequal to 0 in seconds.", fg='#e60000')
                return False
            self.max_ess=ess
        except:
            self.warn_label.configure(text="Steady-state error must be\na real greater than or equal\nto 0 in seconds.", fg='#e60000')
            return False
        
        # Recovering steady-state error punishment factor
        try:
            ess=float(self.entry_pun_ess.get())
            if ess<0:
                self.warn_label.configure(text="Steady-state error punishment\nmust be a real greater\nor equal to 0.", fg='#e60000')
                return False
            self.pun_ess=ess
        except:
            self.warn_label.configure(text="Steady-state error punishment\nmust be a real greater than\nor equal to 0.", fg='#e60000')
            return False
        
        # Successful recovery
        return True
    
    #%% Core of the program, computes the new soltutions and select the best one to display
    def compute(self,iterations):
        # First step after reset: recovering parameters (only done once)
        if self.step_nb==0:
            # Recovering PID parameters
            completed=self.recover_PID_param()
            if not completed:
                return False
            
            # Recovering EA parameters
            # Locking algorithm
            if self.is_SOMA:
                self.lock_SOMA=True
            else:
                self.lock_SOMA=False
                
            # Recovering parameter (population)
            try: # test if it has the correct form
                pop=self.population.get()
                pop=float(pop)
                if int(pop)!=pop or pop<6:
                    self.warn_label.configure(text="Population must be an\integer greater than 5.", fg='#e60000')
                    return False
                self.pop = int(pop)
            except: # Warning if not
                self.warn_label.configure(text="Population must be an\integer greater than 5.", fg='#e60000')
                return False
            
            # Recovering algorithm specific parameters
            if self.is_SOMA:
                # Recovering strategy choice
                self.strat=self.strategy.current()
            
                # Recovering path length
                try:
                    path=float(self.path_len.get())
                    if path<1:
                        self.warn_label.configure(text="Path length must be\na real (sugested [2,5]).", fg='#e60000')
                        return False
                    self.path_l=path
                except:
                    self.warn_label.configure(text="Path length must be\na real (sugested [2,5]).", fg='#e60000')
                    return False
                
                # Recovering step
                try:
                    step=self.step.get()
                    if step[-1]=='%': # Testing for percent option
                        step=float(step[:-1])
                        if step<=0 or step>100:
                            self.warn_label.configure(text="Adding % make step a percent\nof path length, thus it\nshould be between 0 and 100.", fg='#e60000')
                            return False
                        self.step_percent=True
                    else:
                        step=float(step)
                        if step>self.maxi-self.mini:
                            self.warn_label.configure(text="Step must be contained\nwithin interval", fg='#e60000')
                            return False
                        self.step_percent=False
                    self.stp=step
                except:
                    self.warn_label.configure(text="Step must be a real,\nyou can add % to make it\a percent of path length.", fg='#e60000')
                    return False
                
                # Recovering PRT option
                prt_chosen=self.prt.current()
                if prt_chosen==0:
                    self.prt_chosen=True
                else:
                    self.prt_chosen=False
            
            else: # DE
                # Recovering strategy choice
                self.strat=self.DE_strategy.current()
            
                # Recovering mutation scalar F
                try:
                    mut=float(self.mutation.get())
                    if mut<0 or mut>2:
                        self.warn_label.configure(text="Mutation (Fxc) must be a real,\nbetween 0 and 2.", fg='#e60000')
                        return False
                    self.mut=mut
                except:
                    self.warn_label.configure(text="Mutation (Fxc) must be a real,\nrecommended between 0 and 2.", fg='#e60000')
                    return False
                
                # Recovering crossover threshold
                try:
                    crossvr=float(self.crossover.get())
                    if crossvr<0 or crossvr>1:
                        self.warn_label.configure(text="Crossover must be a real\nbetween 0 and 1.", fg='#e60000')
                        return False
                    self.crossvr=crossvr
                except:
                    self.warn_label.configure(text="Crossover must be a real\nbetween 0 and 1.", fg='#e60000')
                    return False
                
                # Recovering mutation parameter
                try:
                    lbd=float(self.lbda.get())
                    if lbd<0 or lbd>1:
                        self.warn_label.configure(text="Lambda/K/Fcr must be a real,\nbetween 0 and 1.", fg='#e60000')
                        return False
                    self.lbd=lbd
                except:
                    self.warn_label.configure(text="Lambda/K/Fcr must be a real,\nbetween 0 and 1.", fg='#e60000')
                    return False
            
            
            # First iteration of algorithms - Random points in usual range of parameters
            Kp = np.random.rand(self.pop, 1)*100
            Ki = np.random.rand(self.pop, 1)*50
            Kd = np.random.rand(self.pop, 1)*100
            N = np.random.rand(self.pop, 1)*20+2
            points = np.block([Kp, Ki, Kd, N])
            
            # Fitting parameters and step response
            stability, results, time, values = self.get_fit(points)
            
            best_fit_row = self.get_best_row(stability,results)
            self.is_stable = stability[best_fit_row]
            self.best_fit=points[best_fit_row,:] # Global best fit for ploting
            self.best_fit_value.append(results[best_fit_row,4]) # Evolution of best fit through iterations to plot evolution graph
            self.best_fit_time = time[best_fit_row,:]
            self.best_fit_values = values[best_fit_row,:]
            self.best_fit_result = results[best_fit_row,:]
            # We don't take time and value for points as we don't need to display them
            self.points=np.delete(points,best_fit_row,0)
            # First iteration completed
            self.step_nb=1
            if iterations==1:
                return True
            else:
                iterations-=1
        
        #%% Iterations loop, core of the algorithms
        for i in range(iterations):
            self.step_nb+=1
        
            if self.lock_SOMA: # SOMA
                # Depending on the strategy chosen, we will compute new points differently
                # All to One (leader) - leader is refered to as the best fit
                if self.strat==0:
                    # Computing points
                    result_points, result_stability, result_results, result_time, result_values = self.SOMA_jump(self.points, self.best_fit, self.best_fit_result[4],self.best_fit_result[5])
                    # Grouping
                    result_points = np.block([[result_points],[self.best_fit]])
                    result_stability.append(self.is_stable)
                    result_results = np.block([[result_results],[self.best_fit_result]])
                    result_time = np.block([[result_time],[self.best_fit_time]])
                    result_values = np.block([[result_values],[self.best_fit_values]])
                    
                
                # All to All
                if self.strat==1:
                    points = np.block([[self.points],[self.best_fit]])
                    result_points = np.array([]).reshape(0,4)
                    result_stability = []
                    result_results = np.array([]).reshape(0,6)
                    result_time = np.array([]).reshape(0,1000)
                    result_values = np.array([]).reshape(0,1000)
                    
                    for j in range(len(points[:,0])):
                        it_jumper = points[j]
                        it_points = np.delete(points, j, 0)
                        # Computing
                        it_points, it_stability, it_results, it_time, it_values = self.SOMA_jump_2(it_points, it_jumper)
                        # Grouping
                        result_points = np.block([[result_points],[it_points]])
                        result_stability.append(it_stability)
                        result_results = np.block([[result_results],[it_results]])
                        result_time = np.block([[result_time],[it_time]])
                        result_values = np.block([[result_values],[it_values]])
                        
                # All to One random
                if self.strat==2:
                    points=np.block([[self.points],[self.best_fit]])
                    index = int(len(points[:,0])*np.random.rand())
                    best_fit=points[index,:]
                    points=np.delete(points,index,0)
                    # Computing points
                    best_stability, best_results, best_time, best_values = self.get_fit(best_fit)
                    result_points, result_stability, result_results, result_time, result_values = self.SOMA_jump(points, best_fit, best_results[4], best_results[5])
                    # Grouping
                    result_points = np.block([[result_points],[self.best_fit]])
                    result_stability.append(self.is_stable)
                    result_results = np.block([[result_results],[self.best_fit_result]])
                    result_time = np.block([[result_time],[self.best_fit_time]])
                    result_values = np.block([[result_values],[self.best_fit_values]])

                # All to all adaptative
                if self.strat==3:
                    points = np.block([[self.points],[self.best_fit]])
                    
                    result_points = np.array([]).reshape(0,4)
                    result_stability = []
                    result_results = np.array([]).reshape(0,6)
                    result_time = np.array([]).reshape(0,1000)
                    result_values = np.array([]).reshape(0,1000)
                    
                    for j in range(len(points[:,0])):
                        it_jumper = points[j]
                        it_points = np.delete(points, j, 0)
                        # Computing
                        it_points, it_stability, it_results, it_time, it_values = self.SOMA_jump_2(it_points, it_jumper)
                        # Grouping
                        result_points = np.block([[result_points],[it_points]])
                        result_stability.append(it_stability)
                        result_results = np.block([[result_results],[it_results]])
                        result_time = np.block([[result_time],[it_time]])
                        result_values = np.block([[result_values],[it_values]])
                        # To make it adaptative
                        points[j]=it_points
                        
                # Team to Team adaptative
                if self.strat==4:
                    # Selection of teams
                    points = np.block([[self.points],[self.best_fit]])
                    n=len(points[:,0])
                    """ Here are the T3A specific variables
                    with k the number of points in the migrant team
                    and m the number of points in the leading team
                    they must be lower than n
                    """
                    k = int(n/2) # Migrant team population
                    m = int(n/2) # Leading team population
                    mig_ind = np.random.permutation(n)[:k]
                    lead_ind = np.random.permutation(n)[:m]
                    
                    # Fitting
                    points_stability, points_results, points_time, points_values = self.get_fit(points)
                    # Choice of best leader
                    best_fit_row = self.get_best_row(points_stability[lead_ind], points_results[lead_ind])
                    lead_ind=lead_ind[best_fit_row] # Index in points
                    # Computing
                    mig_points, mig_stability, mig_results, mig_time, mig_values = self.SOMA_jump(points[mig_ind], points[lead_ind], points_results[lead_ind,4], points_results[lead_ind,5])
                    # Replacing migrants with their new values
                    result_points = points
                    result_stability = points_stability
                    result_results = points_results
                    result_time = points_time
                    result_values = points_values
                    
                    result_points[mig_ind] = mig_points
                    result_stability[mig_ind] = mig_stability
                    result_results[mig_ind] = mig_results
                    result_time[mig_ind] = mig_time
                    result_values[mig_ind] = mig_values
            
                
                # Choice of best fit for plotting or other iterations
                best_fit_row = self.get_best_row(result_stability,result_results)
                self.is_stable=result_stability[best_fit_row]
                self.best_fit=result_points[best_fit_row,:] # Global best fit for ploting
                self.best_fit_value.append(result_results[best_fit_row,4]) # Evolution of best fit through iterations to plot evolution graph
                self.best_fit_time = result_time[best_fit_row,:]
                self.best_fit_values = result_values[best_fit_row,:]
                self.best_fit_result = result_results[best_fit_row,:]
                # We don't take time and value for points as we don't need to display them
                self.points=np.delete(result_points,best_fit_row,0)
                
                if i==iterations-1:
                    return True # Successful end of iterations
            
            else: # DE
                points = np.block([[self.points],[self.best_fit]]) # Grouping
                new_points = points
                result_stability = []
                result_results = np.array([]).reshape(0,6)
                result_time = np.array([]).reshape(0,1000)
                result_values = np.array([]).reshape(0,1000)
                # Calcul loop for DE
                for j in range(len(points[:,0])):
                    # First solution
                    solution_1=points[j]
                    it_points=np.delete(points,j,0)
                    index = np.random.permutation(self.pop-1) # Random indexing
                    
                    # Second solution computation (depending on strategy)
                    if self.strat==0: # rand 1
                        solution_2 = it_points[index[0]] + self.mut*(it_points[index[1]]-it_points[index[2]])
                    
                    if self.strat==1: # rand 2
                        solution_2 = it_points[index[0]] + self.mut*(it_points[index[1]]-it_points[index[2]]+it_points[index[3]]-it_points[index[4]])
    
                    if self.strat==2: # best 1
                        solution_2 = self.best_fit + self.mut*(it_points[index[0]]-it_points[index[1]])
    
                    if self.strat==3: # best 2
                        solution_2 = self.best_fit + self.mut*(it_points[index[0]]-it_points[index[1]]+it_points[index[2]]-it_points[index[3]])
                    
                    if self.strat==4: # current-to-best 1
                        solution_2 = solution_1 + self.lbd*(self.best_fit-solution_1) + self.mut*(it_points[index[0]]-it_points[index[1]])
                        
                    if self.strat==5: # current-to-rand 1
                        solution_2 = solution_1 + self.lbd*(it_points[index[0]]-solution_1) + self.mut*(it_points[index[1]]-it_points[index[2]])
                    
                    if self.strat==6: # rand-to-best 1
                        solution_2 = it_points[index[0]] + self.lbd*(self.best_fit-solution_1) + self.mut*(it_points[index[1]]-it_points[index[2]])
                    
                    # Testing if new solution is within borders
                    # And correcting if necessary
                    if solution_2[0]<=0:
                        solution_2[0]=0.01
                    if solution_2[1]<0:
                        solution_2[1]=0
                    if solution_2[2]<0:
                        solution_2[2]=0
                    if solution_2[3]<0:
                        solution_2[3]=0
                        
                    # Crossover
                    cr=np.random.rand(4)
                    cr=cr<self.crossvr
                    cr=cr.astype(int)
                    solution_2=cr*solution_2 + (-cr+1)*solution_1
                    
                    # Fitting parameters and step response
                    stability_1, results_1, time_1, values_1 = self.get_fit(solution_1)
                    stability_2, results_2, time_2, values_2 = self.get_fit(solution_2)
                    
                    best_fit_row=self.get_best_row([stability_1,stability_2], np.block([[results_1],[results_2]]))
                    
                    if best_fit_row==0:
                        result_stability.append(stability_1)
                        result_results = np.block([[result_results],[results_1]])
                        result_time = np.block([[result_time],[time_1]])
                        result_values = np.block([[result_values],[values_1]])
                    else:
                        new_points[j]=solution_2
                        result_stability.append(stability_2)
                        result_results = np.block([[result_results],[results_2]])
                        result_time = np.block([[result_time],[time_2]])
                        result_values = np.block([[result_values],[values_2]])
                
                # Choice of best fit for plotting or other iterations
                best_fit_row = self.get_best_row(result_stability,result_results)
                self.is_stable=result_stability[best_fit_row]
                self.best_fit=new_points[best_fit_row,:] # Global best fit for ploting
                self.best_fit_value.append(result_results[best_fit_row,4]) # Evolution of best fit through iterations to plot evolution graph
                self.best_fit_time = result_time[best_fit_row,:]
                self.best_fit_values = result_values[best_fit_row,:]
                self.best_fit_result = result_results[best_fit_row,:]
                # We don't take time and value for points as we don't need to display them
                self.points=np.delete(new_points,best_fit_row,0)
                
                if i==iterations-1:
                    return True # Successful end of iterations
    
    
    #%% This function performs a jump of all the points in 'points' to the best fit,
    # points and best fit may differ from all points and best fit depending on strategy
    # eg. for all to one rand, best fit will be a random point, and points will be
    # the others points plus the actual best fit
    def SOMA_jump(self, points, best_fit, best_fit_cost, best_fit_addcost, adaptative=False):
        if self.step_percent:
            jump_nb=int(100/self.stp) # Number of jumps
            step=self.stp/100 # Converting to percent
        else:
            step=self.stp
            
        # Iteration loop
        try:
            n=len(points[:,0])
        except:
            n=1
        
        results = points # New positions to be changed and returned by this function
        # If percent is chosen, we'll take a percent of path length * vector
        results_cost = np.array([]).reshape(0,6)
        results_time = np.array([]).reshape(0,1000)
        results_values = np.array([]).reshape(0,1000)
        results_stability = []
        
        for j in range(n):
            new_points = points[j]
            start_point = points[j]
            # Vector between migrant and leader
            vector=best_fit-start_point
            vector= self.path_l*vector
            # In case migrant and leader are the same
            if np.linalg.norm(vector)==0:
                continue
            
            # If percent is not chosen, we'll use the normed vector and step
            if not self.step_percent:
                norm=np.linalg.norm(vector) # Distance between the point and leader
                jump_nb=int(norm/self.stp)
                vector=vector/norm # Normalizing vector
                
                # In case there is no jump because distance is too short (int is a floor)
                if jump_nb==0:
                    jump_nb=1
    
            if self.prt_chosen:
                jump_nb=2*jump_nb
                # As there is a 50% chance to move in a dimension while using PRT
                # We double the number of jumps to counter this effect
                # And usually arrive around the same spot than without prt
                for k in range(jump_nb):
                    # PRT computing with at least 1 one and 1 zero
                    prt = np.block([0,1,np.random.rand(2)*2]).astype(int)
                    # Random shuffling of 0 and 1 positions
                    prt = np.random.permutation(prt)
                    # New point after jump
                    new_point=start_point+prt*step*vector
                    # Testing if new solution is within borders
                    # And correcting if necessary
                    if new_point[0]<=0:
                        new_point[0]=0.01
                    if new_point[1]<0:
                        new_point[1]=0
                    if new_point[2]<0:
                        new_point[2]=0
                    if new_point[3]<0:
                        new_point[3]=0
                    # Grouping
                    new_points=np.block([[new_points],[new_point]])
                    start_point=new_point
            else:
                for k in range(jump_nb):
                    # New point after jump
                    new_point=start_point+step*vector
                    # Testing if new solution is within borders
                    # And correcting if necessary
                    if new_point[0]<=0:
                        new_point[0]=0.01
                    if new_point[1]<0:
                        new_point[1]=0
                    if new_point[2]<0:
                        new_point[2]=0
                    if new_point[3]<0:
                        new_point[3]=0
                    # Grouping
                    new_points=np.block([[new_points],[new_point]])
                    start_point=new_point
            
            # Fitting
            stability, costs, time, values = self.get_fit(new_points)
            
            best_fit_row = self.get_best_row(stability,costs)
            
            # Choice of new best position
            results[j] = new_points[best_fit_row]
            results_stability.append(stability[best_fit_row])
            results_cost = np.block([[results_cost],[costs[best_fit_row]]])
            results_time = np.block([[results_time],[time[best_fit_row]]])
            results_values = np.block([[results_values],[values[best_fit_row]]])
            
            if adaptative: # If adaptative, change the best fit when necessary
            # NB: original best fit will not move in this function
                if costs[4]==0:
                    if best_fit_cost==0:
                        if costs[5]<best_fit_addcost:
                            best_fit = new_points[best_fit_row]
                            best_fit_cost = costs[best_fit_row,4]
                            best_fit_addcost = costs[best_fit_row,5]
                    else:
                        best_fit = new_points[best_fit_row]
                        best_fit_cost = costs[best_fit_row,4]
                        best_fit_addcost = costs[best_fit_row,5]
                elif costs[4]<best_fit_cost:
                    best_fit = new_points[best_fit_row]
                    best_fit_cost = costs[best_fit_row,4]
                    best_fit_addcost = costs[best_fit_row,5]
                    
        return results, results_stability, results_cost, results_time, results_values
    
    #%% This function does the opposite of the first one: instead of making all points
    # jump to the leader, it's the jumper that jumps to all points.
    # By design, it can't be adaptative.
    def SOMA_jump_2(self, points, jumper):
        # List with all computed points, this function will return the best of them
        new_points = jumper
        # Step definition
        if self.step_percent:
            jump_nb=int(100/self.stp) # Number of jumps
            step=self.stp/100 # Converting to percent
        else:
            step=self.stp
        
        try:
            n=len(points[:,0])
        except:
            n=1
        # Iteration loop
        for j in range(n):
            # Vector between point and jumper
            start_point = jumper
            vector= points[j]-start_point
            vector= self.path_l*vector
            
            # If percent is not chosen, we'll use the normed vector and step
            if not self.step_percent:
                norm=np.linalg.norm(vector) # Distance between the point and leader
                jump_nb=int(norm/self.stp)
                vector=vector/norm # Normalizing vector
                
                # In case there is no jump because distance is too short (int is a floor)
                if jump_nb==0:
                    jump_nb=1
    
            if self.prt_chosen:
                jump_nb=2*jump_nb
                # As there is a 50% chance to move in a dimension while using PRT
                # We double the number of jumps to counter this effect
                # And usually arrive around the same spot than without prt
                for k in range(jump_nb):
                    # PRT computing with at least 1 one and 1 zero
                    prt = np.block([0,1,np.random.rand(2)*2]).astype(int)
                    # Random shuffling of 0 and 1 positions
                    prt = np.random.permutation(prt)
                    # New point after jump
                    new_point=start_point+prt*step*vector
                    # Testing if new solution is within borders
                    # And correcting if necessary
                    if new_point[0]<=0:
                        new_point[0]=0.01
                    if new_point[1]<0:
                        new_point[1]=0
                    if new_point[2]<0:
                        new_point[2]=0
                    if new_point[3]<0:
                        new_point[3]=0
                    # Grouping
                    new_points=np.block([[new_points],[new_point]])
                    start_point=new_point
            else:
                for k in range(jump_nb):
                    # New point after jump
                    new_point=start_point+step*vector
                    # Testing if new solution is within borders
                    # And correcting if necessary
                    if new_point[0]<=0:
                        new_point[0]=0.01
                    if new_point[1]<0:
                        new_point[1]=0
                    if new_point[2]<0:
                        new_point[2]=0
                    if new_point[3]<0:
                        new_point[3]=0
                    # Grouping
                    new_points=np.block([[new_points],[new_point]])
                    start_point=new_point
                    
        # Choice of new best position
        # Fitting
        stability, costs, time, values = self.get_fit(new_points)
        best_fit_row = self.get_best_row(stability,costs)
        
        return new_points[best_fit_row], stability[best_fit_row], costs[best_fit_row], time[best_fit_row], values[best_fit_row]
    
    
    #%% Function that returns the stability, overshoot, rise time, 
    # settling time, steady-state error, cost and additional cost (uniform
    # punishent with maximums at zeros for function that have cost=0)
    def get_fit(self, points):
        stability, results = [],np.array([]).reshape(0,6)
        time, values = np.array([]).reshape(0,1000) , np.array([]).reshape(0,1000)
        # If only one value is inputed (so points has only one row)
        try:
            n=len(points[:,0])
        except:
            n=1
        # Testing all points
        for i in range(n):
            if n==1:
                it_tf=self.get_tf(points[0],points[1],points[2],points[3],self.num,self.den,self.cl_loop)
            else:
                it_tf=self.get_tf(points[i,0],points[i,1],points[i,2],points[i,3],self.num,self.den,self.cl_loop)
            # Step response
            t, y = signal.step(it_tf,N=1000)
            
            # Stability test (over the last 10 values)
            # For some functions, it diverges so quickly that it creates numbers
            # higher than thoses contained by python float, creating NaN and other warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    mean=np.sum(abs(y[-20:-1]-y[-19:]))/20
                except Warning:
                    mean = mean=1e300
            ind_nan = np.where(np.isnan(y))[0]
            if len(ind_nan)!=0:
                mean=1e300
                y[ind_nan] = 1e300
            
            # Changing time vector to better display and compute fitting parameters 
            if mean<0.00001: # Converges too fast, we will reduce max time
                t,y,mean=self.autofit(t,y,it_tf,True)
            elif mean>0.0001 and mean<1: # Some function may have not got enough
                # time to converge, we will increase max time and test
                t,y,mean=self.autofit(t,y,it_tf,False)
            
            # grouping values for plotting
            time = np.block([[time],[t]])
            values = np.block([[values],[y]])
            
            if mean<0.001: # This value may need to be tweaked
                stability.append(True)
                # Steady state-value
                if y[-1]>2:
                    ss = round(y[-1],1)
                else:
                    ss = round(y[-1],2)
                # Overshoot
                if ss!=0:
                    result=[max((np.amax(y)/ss-1)*100,0)]
                else:
                    result=[0]
                # Rise time
                ind_10=-1
                ind_90=-1
                j=1
                while ind_90==-1 and j<1000:
                    if abs(y[j])>0.1*abs(ss) and ind_10==-1:
                        ind_10=j-1
                    if abs(y[j])>0.9*abs(ss):
                        ind_90=j
                    j+=1
                # In case of a warning
                if j==1000:
                    if ind_10!=-1:
                        result.append(t[-1]-t[ind_10])
                    else:
                        result.append(t[-1])
                else:
                    result.append(t[ind_90]-t[ind_10])
                # Settling time
                j=-1
                while abs(y[j]-ss)<0.02*abs(ss) and j>-1001:
                    j-=1
                result.append(t[j+1])
                # Steady-state error
                result.append(abs(ss-1))
                # Cost
                cost=np.array([max(result[0]-self.max_os,0)*self.pun_os, max(result[1]-self.max_tr,0)*self.pun_tr, max(result[2]-self.max_ts,0)*self.pun_ts, max(result[3]-self.max_ess,0)*self.pun_ess])
                result.append(np.sum(cost))
                # Additional cost
                add_cost=np.sum(result)
                result.append(add_cost)
                # Grouping
                results = np.block([[results],[np.array(result)]])
            else:
                stability.append(False)
                results=np.block([[results],[0,0,0,0,mean+10000,0]])
        
        # Selecting best fit
        if n==1:
            return stability[0], results, time, values
        else:
            return stability, results, time, values
    
    
    #%% Function that returns the row with the best fit
    def get_best_row(self, stability, results):
        # Selecting best fit to plot
        stable = np.where(stability)[0]
        try:
            if len(stable)==0: # No stable solution found, best solution will be
                # the one with the lowest cost
                means=results[:,4]
                best_fit_row = np.where(means == np.amin(means))[0][0]
                
            else: # Stable found, best one will be the one with the lowest cost
            # and if there are multiple 0 costs, the additional cost will
            # set the best_fit
                stable_points= results[stable,:] # Stable points
                if len(stable)==1:
                    best_fit_row=stable[0]
                else:
                    minimum = np.amin(stable_points[:,4])
                    if minimum==0:
                        ind_zeros = np.where(stable_points[:,4] == 0)[0]
                        if len(ind_zeros)==0:
                            best_fit_row=stable[ind_zeros[0]]
                        else:
                            zeros = stable_points[ind_zeros]
                            best_fit_row = np.where(zeros[:,5] == np.amin(zeros[:,5]))[0][0]
                            best_fit_row = stable[ind_zeros[best_fit_row]]
                    else:    
                        best_fit_row = np.where(stable_points[:,4] == np.amin(stable_points[:,4]))[0][0] # best fit row of stable points
                        best_fit_row = stable[best_fit_row] # actual best fit row (of all results)
        except:
            best_fit_row=-1
        return best_fit_row
        
    
    #%% Function that tries to find a better time vector than the default one
    # if basic stability requirements are met
    def autofit(self,t,y,tf,stable):
        # Already stable, reducing time vector
        if stable:
            mean=np.sum(abs(y[-20:-1]-y[-19:]))/20
            i=0
            while mean<0.00001:
                i-=1
                mean=np.sum(abs(y[-20+i:-1+i]-y[-19+i:i]))/20
            tmax=t[i]
            new_t = np.linspace(0, tmax, num=1000)
            t, y = signal.step(tf,T=new_t)
            return t,y,mean
        # Not stable, increasing time vector
        else:
            new_t=np.linspace(0,t[-1]*5,1000)
            t1,y1 = signal.step(tf,T=new_t)
            mean1=np.sum(abs(y1[-100:-1]-y1[-99:]))/100
            if mean1<0.0001:
                return self.autofit(t1,y1,tf,True)
            else:
                return t,y,np.sum(abs(y[-20:-1]-y[-19:]))/20
        
        
    #%% Function that returns a scipy transfert function from parameters
    def get_tf(self,Kp,Ki,Kd,N,num,den,closed_loop):
        PID_num=np.array([Kp+Kd*N, Kp*N+Ki, Ki*N])
        PID_den=np.array([1, N, 0])
        # Multiplying PID with tf
        num=np.convolve(PID_num,num)
        den=np.convolve(PID_den,den)
        # If closed-loop, compute a unit negative feedback
        if closed_loop:
            n=len(num)
            d=len(den)
            if n==d:
                den=den+num
            elif n>d:
                den = num + np.block([np.zeros((1,n-d)),den])
            else:
                den = np.block([np.zeros((1,d-n)),num]) + den
            den=den.flatten()
        # Transfert function creation
        tf = signal.lti(num, den)
        return tf
    
    #%% Function that plot the step response
    def plot_step(self):
        self.ax.clear()
        self.ax.plot(self.best_fit_time, self.best_fit_values, 'r-')
        self.canvas.draw() # Refresh plot so points appear
        return True
    
    
    #%% Function that displays values (and rounds them) on the results frame
    def display_results(self):
        # Step number
        self.result_step.configure(text=str(self.step_nb))
        # Stability
        if self.is_stable:
            self.result_stable.configure(text='Yes')
            if self.best_fit_result[4]==0:
                disp_text = '0 (' + str(round(self.best_fit_result[5],2)) +')'
                self.result_cost.configure(text=disp_text)
            else:
                self.result_cost.configure(text=str(round(self.best_fit_result[4],2)))
                
            self.result_overshoot.configure(text=str(round(self.best_fit_result[0],2)))
            self.result_rise_time.configure(text=str(round(self.best_fit_result[1],3)))
            self.result_settling_time.configure(text=str(round(self.best_fit_result[2],3)))
            self.result_ess.configure(text=str(round(self.best_fit_result[3],3)))
        else:
            self.result_stable.configure(text='No')
            self.result_overshoot.configure(text='NA')
            self.result_rise_time.configure(text='NA')
            self.result_settling_time.configure(text='NA')
            self.result_ess.configure(text='NA')
            self.result_cost.configure(text=str(round(self.best_fit_result[4])))
        # Best PID parameters
        self.result_Kp.configure(text=str(round(self.best_fit[0],3)))
        self.result_Ki.configure(text=str(round(self.best_fit[1],3)))
        self.result_Kd.configure(text=str(round(self.best_fit[2],3)))
        self.result_N.configure(text=str(round(self.best_fit[3],3)))
        # Time taken - to compare with other algorithms
        self.result_time.configure(text=str(np.around(self.time_elapsed,decimals=1)))
        
    #%% Function that reset the plot and clear all points and data
    #   and plot current untuned function
    def reset_plot(self):
        self.warn_label.configure(text="")
        self.step_nb = 0
        self.ax.clear()
        self.best_fit_value=[]
        self.evolution_scatter=[]
        self.time_elapsed=0
        
        # Plot current untuned function
        # Recovering PID parameters
        completed=self.recover_PID_param()
        if not completed:
            return False
        
        param = np.array([1,0,0,10]) # As if there was no PID (N must be greater than 1)
        
        stability, results, time, values = self.get_fit(param)
        if stability:
            self.is_stable=True
        else:
            self.is_stable=False
        # For plotting (need to flatten np.ndarray for matplolib to plot them)
        self.best_fit_result = results.flatten()
        self.best_fit=param
        self.best_fit_time = time.flatten()
        self.best_fit_values = values.flatten()
        
        self.plot_step() # Plotting values
        self.display_results() # Refresh numbers on the result frame
        
        return True
    
    #%% Function that shows/hides (destroys) evolution graph
    def evo_graph(self):
        if self.evo_graph_on:
            self.evo_graph_on = False
            self.evolution_graph.destroy()
            self.evo_button.configure(text='Show evolution graph')
            
        else:
            self.evo_graph_on = True
            self.evo_button.configure(text='Show plot graph')
            # Graph frame
            self.evolution_graph= Frame(self.root)
            self.evolution_graph.grid(column=0, row=1, columnspan=6, rowspan=2)
            # Graph figure
            self.evolution_fig = Figure(figsize=(5, 4), dpi=100)
            self.evolution_canvas = FigureCanvasTkAgg(self.evolution_fig, master=self.evolution_graph)  # A tk Drawing Area
            # 2D graph
            self.evolution_ax = self.evolution_fig.add_subplot(111)
            self.evolution_toolbar = NavigationToolbar2Tk(self.evolution_canvas, self.evolution_graph)
            self.evolution_toolbar.update()
            self.evolution_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            
            # Draw evolution
            if self.step_nb!=0:
                plot, = self.evolution_ax.plot(self.best_fit_value,'b-')
                self.evolution_scatter = [plot]
                y=[]
                for i in range(len(self.best_fit_value)):
                    y.append(min(self.best_fit_value[:i+1]))
                plot, = self.evolution_ax.plot(y,'r-')
                self.evolution_scatter.append(plot)
            # Refresh plot
            self.evolution_canvas.draw()
    
    #%% Function that hides/shows algorithms parameters depending on choice
    # Saves any change made to the other algorithms so that when selected again
    # it keeps its values.
    def alg_change(self,x):
        algo=self.alg.current()
        if self.is_SOMA and algo==1: # Change parameters to DE parameters
            self.SOMA_frame.grid_remove()
            self.DE_frame.grid()
            self.is_SOMA=False
        elif not self.is_SOMA and algo==0: # Change parameters to SOMA parameters
            self.DE_frame.grid_remove()
            self.SOMA_frame.grid()
            self.is_SOMA=True

#%% Function that opens a new frame
def new_frame():
    global Data
    n=len(Data)
    Subframe('PID Controller '+str(n))
    
    
#%% Function that closes everything and end the program
def close_all():
    global Data, info_frame
    try:
        info_frame.destroy()
    except:
        True
    for i in Data:
        try:
            i.root.destroy()
        except:
            True
        try:
            i.root_plot.destroy()
        except:
            True

#%% Window that appears when clicking on the help button
def information_frame():
    global info_frame
    try:
        info_frame.destroy()
    except:
        True
    info_frame = Tk()
    info_frame.title("Help & information")

    T=Text(info_frame, wrap='word')
    # Raw text (warning: very long)
    T.insert(INSERT, "PID Controller\n\nThis programs allow to view and compare using multiple frames, two evolutionary algorithms, SOMA and DE, in the PID tuning problem.\nThis program is meant to be a display program with nice visual and easy understanding, it isn't meant for efficiency and will be slow at computing, even with my best efforts to make it efficient.\n\nAbout the Frames\nWhen you launch the program, a frame appears. This frame has everything the program can offer on it. Let's review the different options and buttons.\n\nThe \"New Frame\" button will open another frame with default parameters. This is useful for comparing algorithms or parameters side to side.\n\nThe \"close all\" button closes all frames (except the 3d plot of space of possible solution) and ends the program.\n\nAbout the TF and PID parameter frame:\n\nThe closed loop option realize a unity feedback.\n\nThe parameters are what will define the cost of the function and their corresponding punishment factors.\n\nAs you can see from the result frame, a fourth parameter \"N\" named filter coefficient is also here. It is here because it is in theory impossible to make a ideal derivative; thus the program realise the following: Kp+Ki/s+Kd*N/(1+N*1/s) when computing PID parameters (see Simulink parallel PID controller), meaning when N tends to infinity, the formulae tends to Kp+Ki/s+Kd*s, which is our ideal formulae.\n\nIt is important to note that if an parameter is incorrect, the program will display in red a text explaining the corrections to make. Furthermore, the frame always starts with default parameters that work but may not be optimal for every transfert function.\n\nA very important information: To reduce errors, once the first iteration of an experiment is done, its parameters will not change. If you want to change the parameters, you will need to reset it. The program also performs a reset on the first iteration.\n\nHere is a list and explanation for most of what you can see:\n\nReset: Reset the experiment by deleting all variables and clearing plot. Reset also checks the interval parameter and the start point for Hill Climbing.\n\nStep: Step will compute and display one step of the experiment.\n\nRun for X iterations: Run will compute X step and displays the best fit of the last step computed of the experiment. X must be a integer greater than one.\n\nShow evolution/plot graph: For dimension lower or equal to 2, this button will change frame display to evolution graph or the opposite. It is safe to use at any point in the experiment as it will not affect variables.\n\nPlot 10 evolution button: This will compute by default 10 iterations of 10 evolutions and display their best fit evolution. This takes a lot of time to compute so be careful and patient when using it.\n\nParameter frame: In the frame named \"Parameters\" lie all required for the specified algorithm to work properly. Most of them are self explanatory, and if the program detects a mistake, it will display it. Here are those who need further explanation:\n\nStep of SOMA: the step can either be a real, or a percent of the vector the point will jump on, if the last character is a \"%\".\nUse PRT of SOMA: If this is set to \"Yes\", then the program, for each jump, will generate a PRT vector so that the point will move only in some dimensions. To compensate for the loss in overall jumping distance, the number of jump is doubled in this mode, so that the points theorically arrives around the same spot as if PRT wasn't used.\n\nNB: All strategies of DE use binary crossover. Depending on paper studied, the parameters of DE didn't always have the same name, explaining why there is multiple names for the noise vector parameters.\n\nResult frame: This frame will simply display the result that you can see on the plot to have precise numbers; they are rounded so that they do not take too much space.\n\n\nOther useful information\n\nIf you want to save obtained graphs, it can easily be done by clicking the save button on the toolbar of the graph.\n\nIf you find something that breaks the program, please let me know so I can try to fix it.")    
    
    T.pack(expand=1,fill=BOTH,side='left')
    T.configure(state='disabled')
    T.configure(font=("Century Schoolbook",12))
    
    # Scrollbar
    defilY = Scrollbar(info_frame, orient='vertical', command=T.yview)
    defilY.pack(side='right',fill=Y)
    T['yscrollcommand'] = defilY.set
    
    info_frame.mainloop()

#%% Start of program
Data=[]
Subframe('PID Controller 0')