"""
This program aims to find the minimum of well-known functions using 4 differents
evolutionnary algorithms : Blind, Hill Climbing, SOMA and DE.
These algoritms are meant to have multiple inputs (usually x and y) and only
one output (usually z).
A custom function with n input and 1 output may be manually added to this
program at line 1040, in 'Custom'.

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


#%% Subframe class & functions | This is the core of the program as each subframe
#   is an experience, and that the objective is to compare their results
#   depending on their parameters.
class Subframe:
    # Class functions
    def run(self): # Function that compute multiple generations and their results
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
        
        self.plot_values(iterations) # Plotting values
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
        
        self.plot_values(1) # Plotting values
        self.display_results() # Refresh numbers on the result frame

    
    #%% Main function of the program, compute results depending on all parameters
    # Returns points that will be displayed by step() and run()
    def compute(self,iterations):        
        # First step of all algorithms : recovering parameters (only once)
        if self.step_nb==0:
            # Reset plot because interval may have changed
            # Interval and starting point of HC are recovered in reset_plot
            completed = self.reset_plot()
            if not completed:
                return False
            
            # Recovering parameter (interval)
            try: # test if it has the correct form
                minmax=self.interval.get()
                mini=float(minmax[:minmax.find(';')])
                maxi=float(minmax[minmax.find(';')+1:])
            except: # Warning if not
                self.warn_label.configure(text="Interval must be in this form:\nmin;max  with min,max real numbers.", fg='#e60000')
                return False
                
            if mini>maxi: # If min and max are inversed
                a=mini
                mini=maxi
                maxi=a
            self.maxi=maxi
            self.mini=mini
            
            # Recovering parameter (population)
            try: # test if it has the correct form
                pop=self.population.get()
                pop=float(pop)
                if int(pop)!=pop or pop<2:
                    self.warn_label.configure(text="Population must be an\integer greater than 2.", fg='#e60000')
                    return False
                if pop<6 and self.alg_id==3:
                    self.warn_label.configure(text="Population must be an integer\ngreater than 5 for DE.", fg='#e60000')
                    return False
                self.pop = int(pop)
            except: # Warning if not
                self.warn_label.configure(text="Population must be an\integer greater than 2.", fg='#e60000')
                return False
            
            # Recovering algorithm specific parameters
            #%% Hill Climbing
            if self.alg_id==1:
                # Recovering starting point
                try:
                    string=self.start_point.get()
                    ind=string.find(';')
                    if ind==-1:
                        start=float(string)
                        self.start=np.ones(self.dimension)*start
                    else:
                        x=float(string[:ind])
                        self.start=[x]
                        ind1=ind
                        ind2= string.find(';', ind+1)
                        while ind2!=-1:
                            x=float(string[ind1+1:ind2])
                            self.start.append(x)
                            ind1=ind2
                            ind2=string.find(';', ind1+1)
                        x=float(string[ind1+1:])
                        self.start.append(x)
                        self.start = np.array(self.start) # Transforming to numpy array for computing
                        # Length check
                        if len(self.start)!=self.dimension:
                            self.warn_label.configure(text="Start point must have a number\nof values equal to dimension,\nor only one value.", fg='#e60000')
                            return False
                except:
                    self.warn_label.configure(text="Start point must be a real,\nor a list of reals with correct\ndimension such as x;y;...", fg='#e60000')
                    return False
                
                # Recovering radius
                try:
                    rad=float(self.radius.get())
                    self.rad=rad
                except:
                    self.warn_label.configure(text="Radius must be a real.", fg='#e60000')
                    return False
            
            #%% SOMA
            if self.alg_id==2:
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
            
            #%% DE
            if self.alg_id==3:
                # Recovering strategy choice
                self.strat=self.strategy.current()
            
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
            
            
            #%% First iteration of algorithms
            if self.alg_id!=1: # Blind, DE, SOMA
                # Random points in interval
                points = self.mini+np.random.rand(self.pop, self.dimension)*(self.maxi-self.mini)
                
            else: # HC
                # Random point in radius
                if self.dimension==1:
                    points = self.start+self.rad*np.random.rand(self.pop,1)
                    
                # N dimension spherical coordinates for dimension greater than 1
                else:
                    # Random phi (angles) of coordinates, with phi(n-1) [0,2pi] and [0,pi] otherwise (n-2, ... , 1)
                    phi=np.pi*np.random.rand(self.pop,self.dimension-1)
                    scal = np.eye(np.shape(phi)[1])
                    scal[-1,-1]=2
                    phi = phi @ scal
                    cphi=np.block([np.cos(phi),np.ones((self.pop,1))])
                    sphi=np.block([np.ones((self.pop,1)),np.sin(phi)])
                    for i in range(self.dimension-2):
                        sphi[:,i+2]=sphi[:,i+1]*sphi[:,i+2]
                    points=self.start+self.rad*np.random.rand(self.pop,1)*cphi*sphi
            
            # Fitting
            best_fit = self.fctn(points[0])
            best_fit_row = 0
            points_values= [best_fit]
            # Searching for best fit
            for i in range(self.pop-1):
                fit=self.fctn(points[i+1])
                points_values.append(fit)
                if fit<best_fit:
                    best_fit=fit
                    best_fit_row=i+1
        
            # Grouping coordinates and results to allow for more efficient computing
            if self.alg_id==1: # HC
                self.start=points[best_fit_row] # New starting point
            self.best_fit=np.block([points[best_fit_row],best_fit]) # Global best fit for ploting
            self.best_fit_value.append(best_fit) # Evolution of best fit through iterations to plot evolution graph
            points_values = np.reshape(points_values,(self.pop,1))
            self.points=np.block([points,points_values])
            
            # First iteration completed
            self.step_nb=1
            if iterations==1: # Removes best values for easier plotting when called by step()
                self.points=np.delete(self.points,best_fit_row,0)
                return True
            else:
                iterations-=1
                if self.alg_id==1:
                    self.points=np.delete(self.points,best_fit_row,0)
        
        #%% Iterations loop, core of the algorithms
        for i in range(iterations):
            self.step_nb+=1
            #%% Blind
            if self.alg_id==0:
                # Random points in interval
                points = self.mini+np.random.rand(self.pop, self.dimension)*(self.maxi-self.mini)
            
                # Fitting
                best_fit = self.fctn(points[0])
                best_fit_row = 0
                points_values= [best_fit]
                # Searching for best fit
                for j in range(self.pop-1):
                    fit=self.fctn(points[j+1])
                    points_values.append(fit)
                    if fit<best_fit:
                        best_fit=fit
                        best_fit_row=j+1
            
                # Testing if best fit of iterations 
                new_best_fit=False
                if self.best_fit[-1]>best_fit:
                    self.best_fit=np.block([points[best_fit_row],best_fit]) # Global best fit for ploting
                    new_best_fit=True
                self.best_fit_value.append(best_fit) # Evolution of best fit through iterations to plot evolution graph
                
                # Grouping for easier plotting of last iteration
                if i==iterations-1:
                    points_values = np.reshape(points_values,(self.pop,1))
                    self.points=np.block([points,points_values])
                    if new_best_fit:
                        self.points=np.delete(self.points,best_fit_row,0)
                    return True # Successful end of iterations
            
            #%% Hill Climbing
            if self.alg_id==1:
                # Random point in radius
                if self.dimension==1:
                    points = self.start+self.rad*np.random.rand(self.pop,1)
                    
                # N dimension spherical coordinates for dimension greater than 1
                else:
                    # Random phi (angles) of coordinates, with phi(n-1) [0,2pi] and [0,pi] otherwise (n-2, ... , 1)
                    phi=np.pi*np.random.rand(self.pop,self.dimension-1)
                    scal = np.eye(np.shape(phi)[1])
                    scal[-1,-1]=2
                    phi = phi @ scal
                    cphi=np.block([np.cos(phi),np.ones((self.pop,1))])
                    sphi=np.block([np.ones((self.pop,1)),np.sin(phi)])
                    for i in range(self.dimension-2):
                        sphi[:,i+2]=sphi[:,i+1]*sphi[:,i+2]
                    points=self.start+self.rad*np.random.rand(self.pop,1)*cphi*sphi
                
                # Fitting
                best_fit = self.fctn(points[0])
                best_fit_row = 0
                points_values= [best_fit]
                # Searching for best fit
                for j in range(self.pop-1):
                    fit=self.fctn(points[j+1])
                    points_values.append(fit)
                    if fit<best_fit:
                        best_fit=fit
                        best_fit_row=j+1
                
                # Grouping coordinates and results to allow for more efficient computing
                self.start=points[best_fit_row] # New starting point
                self.best_fit=np.block([[self.best_fit],[points[best_fit_row],best_fit]]) # Global best fit for ploting
                self.best_fit_value.append(best_fit) # Evolution of best fit through iterations to plot evolution graph
                points_values = np.block([points, np.reshape(points_values,(self.pop,1))])
                points_values = np.delete(points_values,best_fit_row,0)
                self.points=np.block([self.points,points_values])
                
                if i==iterations-1:
                    return True # Successful end of iterations
                
            #%% SOMA
            if self.alg_id==2:
                # Depending on the strategy chosen, we will compute new points differently
                # All to One (leader) - leader is refered to as the best fit
                if self.strat==0:
                    result = np.block([[self.SOMA_jump(self.points, self.best_fit)],[self.best_fit]])
                
                # All to All
                if self.strat==1:
                    points = np.block([[self.points],[self.best_fit]])
                    for j in range(len(points[:,0])):
                        it_best_fit = points[j]
                        it_points = np.delete(points, j, 0)
                        if j==0:
                            result = np.block([[it_best_fit],[self.SOMA_jump(it_points,it_best_fit)]])
                        else:
                            it_result = self.SOMA_jump(it_points,it_best_fit)
                            # np.insert(result, [j], [it_best_fit], axis=0)
                            np.insert(it_result, [j], it_best_fit, axis=0)
                            for k in range(len(it_result[:,0])):
                                if result[k,-1]>it_result[k,-1]:
                                    result[k]=it_result[k]
                            
                # All to One random
                if self.strat==2:
                    points=np.block([[self.points],[self.best_fit]])
                    index = int(len(points[:,0])*np.random.rand())
                    best_fit=points[index,:]
                    points=np.delete(points,index,0)
                    result = np.block([[self.SOMA_jump(points, best_fit)],[best_fit]])

                # All to all adaptative
                if self.strat==3:
                    result = np.block([[self.points],[self.best_fit]])
                    for j in range(len(result[:,0])):
                        it_jumper = result[j]
                        it_points = np.delete(result, j, 0)
                        # This calls the second version of SOMA jump, which realise a jump of
                        # it_best_fit to all other points (it_points)
                        it_best_jump = self.SOMA_jump_2(it_points,it_jumper)
                        result[j]=it_best_jump
                
                # Team to Team adaptative
                if self.strat==4:
                    # Selection of teams
                    result = np.block([[self.best_fit],[self.points]])
                    n=len(result[:,0])
                    """ Here are the T3A specific variables
                    with k the number of points in the migrant team
                    and m the number of points in the leading team
                    they must be lower than n
                    """
                    k = int(n/2) # Migrant team population
                    m = int(n/2) # Leading team population
                    mig_ind = np.random.permutation(n)[:k]
                    lead_ind = np.random.permutation(n)[:m]
                    migrants = result[mig_ind, :]
                    leaders = result[lead_ind, :]
                    
                    # Choice of best leader
                    values = leaders[:,-1]
                    index = np.where(values == np.amin(values))[0][0]
                    leader = leaders[index]
                    
                    result[mig_ind] = self.SOMA_jump(migrants, leader, True)

            
                # All to all adaptative - v2
                # This version isn't strictly the definition of the strategy,
                # so it isn't available, experience may be done to test its
                # efficiency compared to classic All To All adaptative
                if self.strat==5:
                    result = np.block([[self.best_fit],[self.points]]) # We put best fit first as it might improve efficiency
                    for j in range(len(result[:,0])):
                        it_best_fit = result[j]
                        it_points = np.delete(result, j, 0)
                        result = self.SOMA_jump(it_points,it_best_fit)
                        # np.insert(result, [j], [it_best_fit], axis=0)
                        np.insert(result, [j], it_best_fit, axis=0)
                
                # Choice of best fit for plotting or other iterations
                values = result[:,-1]
                best_fit_row = np.where(values == np.amin(values))[0][0]
                self.best_fit=result[best_fit_row] # Global best fit for ploting
                self.best_fit_value.append(self.best_fit[-1]) # Evolution of best fit through iterations to plot evolution graph
                self.points=np.delete(result,best_fit_row,0)
                
                if i==iterations-1:
                    return True # Successful end of iterations
                
            #%% DE
            if self.alg_id==3:
                points = np.block([[self.points],[self.best_fit]]) # Grouping
                new_points = points
                # Calcul loop for DE
                for j in range(len(points[:,0])):
                    # First solution
                    solution_1=points[j]
                    it_points=np.delete(points,j,0)
                    index = np.random.permutation(self.pop-1) # Random indexing
                    
                    # Second solution computation (depending on strategy)
                    # Note: this is not 100% efficient as it also makes operations
                    # on the last column which we don't care about
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
                    
                    # Crossover
                    cr=np.random.rand(self.dimension+1)
                    cr=cr<self.crossvr
                    cr=cr.astype(int)
                    solution_2=cr*solution_2 + (-cr+1)*solution_1
                    
                    # Fitting and change if necessary
                    solution_2[-1] = self.fctn(solution_2[:-1])
                    if solution_2[-1] < solution_1[-1]:
                        new_points[j]=solution_2
                    
                # Grouping and choice of best fit for plotting or other iterations
                values = new_points[:,-1]
                best_fit_row = np.where(values == np.amin(values))[0][0]
                self.best_fit=new_points[best_fit_row] # Global best fit for ploting
                self.best_fit_value.append(self.best_fit[-1]) # Evolution of best fit through iterations to plot evolution graph
                self.points=np.delete(new_points,best_fit_row,0)
                
                if i==iterations-1:
                    return True # Successful end of iterations
            
    
    #%% This function performs a jump of all the points in 'points' to the best fit,
    # points and best fit may differ from all points and best fit depending on strategy
    # eg. for all to one rand, best fit will be a random point, and points will be
    # the others points plus the actual best fit
    def SOMA_jump(self, points, best_fit, adaptative=False):
        results = points # New positions to be changed and returned by this function
        # If percent is chosen, we'll take a percent of path length * vector
        if self.step_percent:
            jump_nb=int(100/self.stp) # Number of jumps
            step=self.stp/100 # Converting to percent
        else:
            step=self.stp
            
        # Iteration loop
        for j in range(len(points[:,0])):
            new_points = points[j]
            start_point = points[j][:-1]
            # Vector between migrant and leader
            vector=best_fit[:-1]-start_point
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
    
            if self.prt_chosen and self.dimension>1:
                jump_nb=2*jump_nb
                # As there is a 50% chance to move in a dimension while using PRT
                # We double the number of jumps to counter this effect
                # And usually arrive around the same spot than without prt
                for k in range(jump_nb):
                    # PRT computing with at least 1 one and 1 zero
                    if self.dimension<3:
                        prt = np.array([0,1])
                    else: # Generation of 0 and 1
                        prt = np.block([0,1,np.random.rand(self.dimension-2)*2]).astype(int)
                    # Random shuffling of 0 and 1 positions
                    prt = np.random.permutation(prt)
                    # New point after jump
                    new_point=start_point+prt*step*vector
                    new_points=np.block([[new_points],[new_point,self.fctn(new_point)]])
                    start_point=new_point
            else:
                for k in range(jump_nb):
                    # New point after jump
                    new_point=start_point+step*vector
                    new_points=np.block([[new_points],[new_point,self.fctn(new_point)]])
                    start_point=new_point
                    
            # Choice of new best position
            values = new_points[:,-1]
            index = np.where(values == np.amin(values))[0][0]
            results[j] = new_points[index]
            
            if adaptative: # If adatative, change the best fit when necessary
            # NB: original best fit will not move in this function
                if values[index]<best_fit[-1]:
                    best_fit = new_points[index]
            
        return results
    
    #%% This function does the opposite of the first one: instead of making all points
    # jump to the leader, it's the jumper that jumps to all points.
    def SOMA_jump_2(self, points, jumper):
        # List with all computed points, this function will return the best of them
        new_points = jumper
        # Step definition
        if self.step_percent:
            jump_nb=int(100/self.stp) # Number of jumps
            step=self.stp/100 # Converting to percent
        else:
            step=self.stp
            
        # Iteration loop
        for j in range(len(points[:,0])):
            # Vector between point and jumper
            start_point = jumper[:-1]
            vector= points[j][:-1]-start_point
            vector= self.path_l*vector
            
            # If percent is not chosen, we'll use the normed vector and step
            if not self.step_percent:
                norm=np.linalg.norm(vector) # Distance between the point and leader
                jump_nb=int(norm/self.stp)
                vector=vector/norm # Normalizing vector
                
                # In case there is no jump because distance is too short (int is a floor)
                if jump_nb==0:
                    jump_nb=1
    
            if self.prt_chosen and self.dimension>1:
                jump_nb=2*jump_nb
                # As there is a 50% chance to move in a dimension while using PRT
                # We double the number of jumps to counter this effect
                # And usually arrive around the same spot than without prt
                for k in range(jump_nb):
                    # PRT computing with at least 1 one and 1 zero
                    if self.dimension<3:
                        prt = np.array([0,1])
                    else: # Generation of 0 and 1
                        prt = np.block([0,1,np.random.rand(self.dimension-2)*2]).astype(int)
                    # Random shuffling of 0 and 1 positions
                    prt = np.random.permutation(prt)
                    # New point after jump
                    new_point=start_point+prt*step*vector
                    new_points=np.block([[new_points],[new_point,self.fctn(new_point)]])
                    start_point=new_point
            else:
                for k in range(jump_nb):
                    # New point after jump
                    new_point=start_point+step*vector
                    new_points=np.block([[new_points],[new_point,self.fctn(new_point)]])
                    start_point=new_point
                    
        # Choice of new best position
        values = new_points[:,-1]
        index = np.where(values == np.amin(values))[0][0]
        result = new_points[index]
        return result
    
    #%% Function that plots values previously computed
    def plot_values(self,iterations):
        # Color palette for HC - 8 colors (yellow, orange, red, magenta, purple, blue, cyan, green)
        color=['#fff100','#ff8c00','#e81123','#ec008c','#68217a','#00188f','#00bcf2','#009e49']
        
        # Remove other points and plot for algorithms other than Hill Climbing
        if self.alg_id!=1 or self.dimension>2: # Blind, DE, SOMA or dimension greater than 2
            for i in self.scatter:
                i.remove()
            self.scatter = []
        
            # Plot new results
            if self.dimension==1:
                scatter = self.ax.scatter(self.best_fit[0],self.best_fit[1],c='#00ffff',marker='X',alpha=1)
                self.scatter.append(scatter)
                scatter = self.ax.scatter(self.points[:,0],self.points[:,1],c='#0060ff',marker='o',alpha=1)
                self.scatter.append(scatter)
            elif self.dimension==2:
                scatter = self.ax.scatter(self.best_fit[0],self.best_fit[1],self.best_fit[2],c='#00ffff',marker='X',alpha=1)
                self.scatter.append(scatter)
                scatter = self.ax.scatter(self.points[:,0],self.points[:,1],self.points[:,2],c='#0060ff',marker='o',alpha=1)
                self.scatter.append(scatter)
                
            # Plotting evolution of best fit for dimension greater than 2
            if self.dimension>2:
                plot, = self.ax.plot(self.best_fit_value,'b-')
                self.scatter.append(plot)
                y=[]
                for i in range(len(self.best_fit_value)):
                    y.append(min(self.best_fit_value[:i+1]))
                plot, = self.ax.plot(y,'r-')
                self.scatter.append(plot)
            
        else: # For hill climbing, we will keep all results with a color scheme
            if iterations==1 and self.step_nb==1:
                nb=self.step_nb-1
                if self.dimension==1:
                    scatter = self.ax.scatter(self.best_fit[0],self.best_fit[1],c=color[1],marker='X',alpha=1)
                    self.scatter.append(scatter)
                    scatter = self.ax.scatter(self.points[:,0],self.points[:,1],c=color[0],marker='o',alpha=0.4)
                    self.scatter.append(scatter)
                elif self.dimension==2:
                    scatter = self.ax.scatter(self.best_fit[0],self.best_fit[1],self.best_fit[2],c=color[1],marker='X',alpha=1)
                    self.scatter.append(scatter)
                    scatter = self.ax.scatter(self.points[:,0],self.points[:,1],self.points[:,2],c=color[0],marker='o',alpha=0.4)
                    self.scatter.append(scatter)
            else:
                for i in range(iterations):
                    nb=self.step_nb-iterations+i
                    if self.dimension==1:
                        scatter = self.ax.scatter(self.best_fit[nb,0],self.best_fit[nb,1],c=color[(nb+1)%8],marker='X',alpha=1)
                        self.scatter.append(scatter)
                        scatter = self.ax.scatter(self.points[:,nb*2],self.points[:,1+nb*2],c=color[nb%8],marker='o',alpha=0.4)
                        self.scatter.append(scatter)
                    elif self.dimension==2:
                        scatter = self.ax.scatter(self.best_fit[nb,0],self.best_fit[nb,1],self.best_fit[nb,2],c=color[(nb+1)%8],marker='X',alpha=1)
                        self.scatter.append(scatter)
                        scatter = self.ax.scatter(self.points[:,nb*3],self.points[:,1+nb*3],self.points[:,2+nb*3],c=color[nb%8],marker='o',alpha=0.4)
                        self.scatter.append(scatter)
        
        if self.evo_graph_on:
            # Remove old values on evolution graph
            for i in self.evolution_scatter:
                i.remove()
            # Plot values on evolution graph
            plot, = self.evolution_ax.plot(self.best_fit_value,'b-')
            self.evolution_scatter = [plot]
            y=[]
            for i in range(len(self.best_fit_value)):
                y.append(min(self.best_fit_value[:i+1]))
            plot, = self.evolution_ax.plot(y,'r-')
            self.evolution_scatter.append(plot)
            # Refresh plot
            self.evolution_canvas.draw()
        
        # Refresh the plot so points appear
        self.canvas.draw()
        return True
        
    #%% Function that displays values (and rounds them) on the results frame
    def display_results(self):
        # Step number
        self.result_step.configure(text=str(self.step_nb))
        
        # Best fit coordinates
        if self.alg_id==1 and self.step_nb!=1:
            values = self.best_fit[:,-1]
            index = np.where(values == np.amin(values))[0][0]
            best_fit = self.best_fit[index]
            self.result_best_fit.configure(text=str(np.around(best_fit[:-1],decimals=2)))
        else:
            self.result_best_fit.configure(text=str(np.around(self.best_fit[:-1],decimals=2)))
        
        # Best fit cost
        if self.alg_id==1 and self.step_nb!=1:
            self.result_cost.configure(text=str(np.around(best_fit[-1],decimals=3)))
        else:
            self.result_cost.configure(text=str(np.around(self.best_fit[-1],decimals=3)))
        
        # Time taken - to compare with other algorithms
        self.result_time.configure(text=str(np.around(self.time_elapsed,decimals=1)))
        
    #%% Function that resets the plot and clears all points and data
    def reset_plot(self):
        self.warn_label.configure(text="")
        self.step_nb = 0
        self.ax.clear()
        if self.dimension>2: # Don't plot the function if dimension is too high
            return True
        self.scatter = []
        self.best_fit_value=[]
        self.evolution_scatter=[]
        self.time_elapsed=0
        
        # Recovering parameter (interval)
        try: # test if it has the correct form
            minmax=self.interval.get()
            mini=float(minmax[:minmax.find(';')])
            maxi=float(minmax[minmax.find(';')+1:])
        except: # Warning if not
            self.warn_label.configure(text="Interval must be in this form:\nmin;max  with min,max real numbers.", fg='#e60000')
            return False
            
        if mini>maxi: # If min and max are inversed
            a=mini
            mini=maxi
            maxi=a
        
        if self.alg_id==1: # Hill Climbing
            # Recovering starting point
            try:
                string=self.start_point.get()
                ind=string.find(';')
                if ind==-1:
                    start=float(string)
                    self.start=np.ones(self.dimension)*start
                else:
                    x=float(string[:ind])
                    self.start=[x]
                    ind1=ind
                    ind2= string.find(';', ind+1)
                    while ind2!=-1:
                        x=float(string[ind1+1:ind2])
                        self.start.append(x)
                        ind1=ind2
                        ind2=string.find(';', ind1+1)
                    x=float(string[ind1+1:])
                    self.start.append(x)
                    self.start = np.array(self.start) # Transforming to numpy array for computing
                    # Length check
                    if len(self.start)!=self.dimension:
                        self.warn_label.configure(text="Start point must have a number\nof values equal to dimension,\nor only one value.", fg='#e60000')
                        return False
            except:
                self.warn_label.configure(text="Start point must be a real,\nor a list of reals with correct\ndimension such as x;y;...", fg='#e60000')
                return False
        
        # Plotting graph
        if self.dimension==1: # Drawing the graph in 2D
            x = np.linspace(mini,maxi,100)
            y=np.zeros(np.shape(x))
            for i in range(100):
                y[i]=self.fctn(x[i])
            self.ax.plot(x,y)
            # Plot starting point for HC
            if self.alg_id==1:
                self.best_fit_value.append(self.fctn(self.start))
                scatter = self.ax.scatter(self.start,self.fctn(self.start),c='#fff100',marker='X',alpha=1)
                self.scatter.append(scatter)
        
        elif self.dimension==2: # Drawing the graph in 3D
            x = np.linspace(mini,maxi,100)
            xv, yv = np.meshgrid(x,x)
            zv = np.zeros(np.shape(xv))
            for i in range(100):
                for j in range(100):
                    zv[i][j]= self.fctn(np.array([xv[i][j],yv[i][j]]))
            self.ax.plot_surface(xv, yv, zv, cmap=cm.autumn, alpha=0.4)
            # Plot starting point for HC
            if self.alg_id==1:
                self.best_fit_value.append(self.fctn(self.start))
                scatter = self.ax.scatter(self.start[0],self.start[1],self.fctn(self.start),c='#fff100',marker='X',alpha=1)
                self.scatter.append(scatter)
        
        # Reset Completed - Refreshing the plot
        self.canvas.draw()
        return True
    
    #%% Function that turns on or off the evolution graph
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
            # Plot if dimension is correct and add a button to plot evolution of best value,
            # otherwise plot evolution of best value
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
    
    #%% Function that plots 10 evolutions of the function with its current parameters
    # in order to compare them. Number of evolutions and number of steps can be changed
    def plot_10(self):
        plot_nb=10
        step_computed=10
        
        # Saving values if there are
        save=False
        if self.step_nb!=0:
            save = True
            if self.alg_id==1:
                save_start=self.start
            save_points=self.points
            save_best_fit=self.best_fit
            save_best_fit_value=self.best_fit_value
            
        # Iteration loop
        tic = time.time()
        for i in range(plot_nb):
            self.step_nb=0
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
            if self.alg_id==1:
                self.start=save_start
            self.points=save_points
            self.best_fit=save_best_fit
            self.best_fit_value=save_best_fit_value
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
        
    #%% Class initialization
    def __init__(self,alg_id,fcn_id,dim,nb):
        global Data
        Data.append(self)
        
        self.fctn, self.name, minmax = param(alg_id,fcn_id)
        self.dimension = dim
        self.step_nb = 0
        self.alg_id = alg_id
        self.scatter = []
        self.best_fit_value=[]
        self.time_elapsed=0
        
        # Main frame (of subframe)
        self.root = Tk()
        self.root.title(str(nb+1)+' - '+self.name)
        Button(self.root, text="Reset",command=self.reset_plot).grid(column=0, row=0, padx=3, pady=4)
        Button(self.root, text="Step",command=self.step).grid(column=1, row=0, padx=7, pady=4)
        
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
        if self.dimension<3:
            self.evo_graph_on=False
            self.evo_button = Button(self.root, text="Show evolution graph",command=self.evo_graph)
            self.evo_button.grid(column=3, row=0, padx=5, pady=4, columnspan=3)
        
        # Button to plot 10 evolutions to compare
        self.plot_10_evo = Button(self.root, text="Plot 10 evolutions",command=self.plot_10)
        self.plot_10_evo.grid(column=6, row=0, padx=3, pady=4)
        
        # Parameter frame
        self.parameters = LabelFrame(self.root, text='Parameters')
        self.parameters.grid(column=6, row=1, padx=5, pady=5)
        
        # Population
        Label(self.parameters, text="Population").grid(column=0, row=0, padx=4,pady=5)
        self.population = Entry(self.parameters, textvariable=StringVar(), width=10)
        self.population.grid(column=1, row=0, padx=4)
        self.population.insert(0, 10)
        
        # Interval
        Label(self.parameters, text="Interval").grid(column=0, row=1, padx=4,pady=5)
        self.interval = Entry(self.parameters, textvariable=StringVar(), width=10)
        self.interval.grid(column=1, row=1, padx=4)
        self.interval.insert(0, minmax)
        
        # Warning text if necessary
        self.warn_label = Label(self.parameters, text="")
        self.warn_label.grid(column=0, row=10, padx=4, pady=5, columnspan=2)
        
        # Algorithm specific parameters
        if alg_id==1: # Hill climbing
            # Starting point
            Label(self.parameters, text="Start point").grid(column=0, row=3, padx=4, pady=5)
            self.start_point = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.start_point.grid(column=1, row=3, padx=2)
            self.start_point.insert(0, str(int(abs(int(minmax[0:minmax.find(';')]))/2)))
            
            # maximum distance between best fit and new points
            Label(self.parameters, text="Max distance").grid(column=0, row=4, padx=4, pady=5)
            self.radius = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.radius.grid(column=1, row=4, padx=2)
            self.radius.insert(0, str(int(abs(int(minmax[0:minmax.find(';')]))/5)+1))
        
        elif alg_id==2: #SOMA
            # Startegy selection
            Label(self.parameters, text="Strategy").grid(column=0, row=3, padx=4, pady=5)
            self.strategy = ttk.Combobox(self.parameters, textvariable=StringVar(), width=15)
            self.strategy['values'] = ('AllToOne','AllToAll','AllToOneRand','AllToAllAdaptative','T3A')
            self.strategy.state(["readonly"])
            self.strategy.grid(column=1, row=3, padx=5, pady=5)
            self.strategy.set('AllToOne')
            
            # Scale between the distance to the leader and the maximum
            Label(self.parameters, text="Path Length").grid(column=0, row=4, padx=4, pady=5)
            self.path_len = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.path_len.grid(column=1, row=4, padx=2)
            self.path_len.insert(0, '2')
            
            # Step between jumps, a % can be added at the end to make it a percent of path length
            Label(self.parameters, text="Step").grid(column=0, row=5, padx=4, pady=5)
            self.step = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.step.grid(column=1, row=5, padx=2)
            self.step.insert(0, '5%')
            
            # Use PRT or not
            Label(self.parameters, text="Use PRT").grid(column=0, row=6, padx=4, pady=5)
            self.prt = ttk.Combobox(self.parameters, textvariable=StringVar(), width=6)
            self.prt['values'] = ('Yes','No')
            self.prt.state(["readonly"])
            self.prt.grid(column=1, row=6, padx=5, pady=5)
            self.prt.set('No')
            
            
        elif alg_id==3: #DE
            # Strategy selection
            Label(self.parameters, text="Strategy").grid(column=0, row=3, padx=4, pady=5)
            self.strategy = ttk.Combobox(self.parameters, textvariable=StringVar(), width=12)
            self.strategy['values'] = ('rand 1','rand 2','best 1','best 2','current-to-best 1','current-to-rand 1','rand-to-best 1')
            self.strategy.state(["readonly"])
            self.strategy.grid(column=1, row=3, padx=5, pady=5)
            self.strategy.set('rand 1')
            
            # Mutation scalar F
            Label(self.parameters, text="Mutation Fxc").grid(column=0, row=4, padx=5,pady=5)
            self.mutation = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.mutation.grid(column=1, row=4, padx=2)
            self.mutation.insert(0, '0.5')
            
            # Treshold crossing CR
            Label(self.parameters, text="Crossover CR").grid(column=0, row=5, padx=5,pady=5)
            self.crossover = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.crossover.grid(column=1, row=5, padx=2)
            self.crossover.insert(0, '0.9')
            
            # Parameter lambda required in rand-to-best strategies
            Label(self.parameters, text="Lambda/K/Fcr").grid(column=0, row=6, padx=5,pady=5)
            self.lbda = Entry(self.parameters, textvariable=StringVar(), width=10)
            self.lbda.grid(column=1, row=6, padx=2)
            self.lbda.insert(0, '0.5')
            
        
        # Graph frame
        self.graph= Frame(self.root)
        self.graph.grid(column=0, row=1, columnspan=6, rowspan=2)
        # Plot if dimension is correct and add a button to plot evolution of best value,
        # otherwise plot evolution of best value
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph)  # A tk Drawing Area
        self.canvas.draw()
        
        # 3D graph
        if self.dimension==2:
            self.ax = self.fig.add_subplot(111, projection="3d")            
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            self.reset_plot()
        # 2D graph
        else:
            self.ax = self.fig.add_subplot(111)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            if self.dimension==1:
                self.reset_plot()
        
        # Result frame (Shows results)
        self.result_frame = LabelFrame(self.root, text='Results')
        self.result_frame.grid(column=6, row=2, padx=5, pady=5)
        
        Label(self.result_frame, text="Current step:").grid(column=0, row=0, padx=3)
        self.result_step = Label(self.result_frame, text="0")
        self.result_step.grid(column=1, row=0, padx=3)
        Label(self.result_frame, text="Best fit:").grid(column=0, row=1, padx=3)
        self.result_best_fit = Label(self.result_frame, text="None")
        self.result_best_fit.grid(column=1, row=1, padx=3)
        Label(self.result_frame, text="Cost:").grid(column=0, row=2, padx=3)
        self.result_cost = Label(self.result_frame, text="None")
        self.result_cost.grid(column=1, row=2, padx=3)
        Label(self.result_frame, text="Time elapsed:").grid(column=0, row=3, padx=3)
        self.result_time = Label(self.result_frame, text="0")
        self.result_time.grid(column=1, row=3, padx=3)
        
        self.root.mainloop() # End of frame loop

#%% Tests functions
# Custom function - if you want to try the algorithms on a custom function
# Must have as an input an np.array x, usually size 2
# Must have as an output a real y
def Custom(x):
    y = np.sum(x*x)
    return y

#First De Jong
def Jong_1(x): #x can be a vector of dimension n>0 (we wil mainly use dimension 2)
    return np.sum(x*x)

#Rosenbrock's valley (Second De Jong)
def Jong_2(x): #x must be a vector of dimension n>1 (we will mainly use dimension 2)
    y=0.
    for i in range(len(x)-1):
        y+= 100*(x[i+1]-x[i]*x[i])**2+(1-x[i])**2
    return y

# Rastrigin's function
def Rastrigin(x): #x can be a vector of dimension n>0 (we wil mainly use dimension 2)
    try: # Tests if a single value has been inputed and transform it
         # into an array so that x[i] doesn't return an error
        n=len(x)
    except:
        n=1
        x=np.array([x])
    y= 10*n
    for i in range(n):
        y+=x[i]**2-10*np.cos(2*np.pi*x[i])
    return y

#Schwefel's function
def Schwefel(x): #x can be a vector of dimension n>0 (we wil mainly use dimension 2)
    return np.sum(-x*np.sin(np.sqrt(abs(x))))

#Ackley's function
def Ackley(x):
    try:
        n=len(x)
    except:
        n=1
    return -20*np.exp(-0.2*np.sqrt(np.sum(x*x)/n))-np.exp(np.sum(np.cos(2*np.pi*x)/n))+20+np.exp(1)


#%% Graphical Interface function
# Window that appears when clicking on the help button
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
    T.insert(INSERT, "Evolutionary Algorithms Visualiser\n\nThis programs allow to view and compare using multiple frames, different evolutionary algorithms including Blind, Hill Climbing, SOMA and DE with a range of well-known cost functions.\nIt's also possible to add your own custom cost function, that needs to have n inputs and 1 output.\nThis program is meant to be a display program with nice visual and easy understanding, it isn't meant for efficiency and will be slow at computing, even with my best efforts to make it efficient.\n\nAbout the Main Frame\nWhen you launch the program, the main frame appears. From it, you can create new subframes with chosen algorithm, function and dimension by selecting those and clicking on \"Start new experiment\" button. These parameters cannot be changed in the subframes.\nThe \"close all\" button closes all frames (main frame, subframes and information frame) and ends the program.\n\n\nAbout the Subframes\n\nA subframe is opened each time the button \"Start new experiment\" is pressed on the main frame with coresponding algorithm, function and dimension. Its name is composed of its index, algorithm name and function name.\n\nEach subframe has a set of buttons and parameters that are or aren't algorithm specific. It is important to note that if an parameter is incorrect, the program will display in red a text explaining the corrections to make. Furthermore, the subframe always starts with default parameters that work but may not be optimal.\n\nA very important information: To reduce errors, once the first iteration of an experiment is done, its parameters will not change. If you want to change the parameters, you will need to reset it. The program also performs a reset on the first iteration.\n\nHere is a list and explanation for what you can see in the subframe:\n\nReset: Reset the experiment by deleting all variables and clearing plot. Reset also checks the interval parameter and the start point for Hill Climbing.\n\nStep: Step will compute and display one step of the experiment.\n\nRun for X iterations: Run will compute X step and usually display the last step computed of the experiment. X must be a integer greater than one.\n\nShow evolution/plot graph: For dimension lower or equal to 2, this button will change frame display to evolution graph or the opposite. It is safe to use at any point in the experiment as it will not affect variables.\n\nPlot 10 evolution button: This will compute by default 10 iterations of 10 evolutions and display their best fit evolution. This takes a lot of time to compute so be careful and patient when using it.\n\nParameter frame: In the frame named \"Parameters\" lie all required for the specified algorithm to work properly. Most of them are self explanatory, and if the program detects a mistake, it will display it. Here are those who need further explanation:\nInterval: must be writen as follow: a;b with a being the minimum and b maximum for ALL dimensions.\nStart point of Hill Climbing: can either be a number alone, in which case the program will take this number in all dimensions, or a set of numbers separated with \";\", with the number of numbers corresponding to dimension.\nStep of SOMA: the step can either be a real, or a percent of the vector the point will jump on, if the last character is a \"%\".\nUse PRT of SOMA: If this is set to \"Yes\", then the program, for each jump, will generate a PRT vector so that the point will move only in some dimensions. To compensate for the loss in overall jumping distance, the number of jump is doubled in this mode, so that the points theorically arrives around the same spot as if PRT wasn't used.\n\nNB: All strategies of DE use binary crossover. Depending on paper studied, the parameters of DE didn't always have the same name, explaining why there is multiple names for the noise vector parameters.\n\nResult frame: This frame will simply display the result that you can see on the plot to have precise numbers; they are rounded so that they do not take too much space.\n\n\nOther useful information\n\nIf you want to save obtained graphs, it can easily be done by clicking the save button on the toolbar of the graph.\n\nIf you find something that breaks the program, please let me know so I can try to fix it.")
    
    T.pack(expand=1,fill=BOTH,side='left')
    T.configure(state='disabled')
    T.configure(font=("Century Schoolbook",12))
    
    # Scrollbar
    defilY = Scrollbar(info_frame, orient='vertical', command=T.yview)
    defilY.pack(side='right',fill=Y)
    T['yscrollcommand'] = defilY.set
    
    info_frame.mainloop()
    
#%% Function that creates a new frame with given parameters on main frame
def new_frame():
    global Data, dimension, labwarn
    # Data gathering
    algo=alg.current()
    f=fcn.current()
    
    # Resetting warning message if there is
    labwarn.configure(text="")
    # Verification to make sure we don't send bad info to our program
    try:
        dim= int(dimension.get())
        if dim!=float(dimension.get()) or dim<1:
            labwarn.configure(text="Dimension must be an integer greater or equal to 1.")
            return
    except:
        labwarn.configure(text="Dimension must be an integer greater than 0.")
        return
    if f==1 and dim==1:
        labwarn.configure(text="Dimension must be an integer greater than 1 for 2nd DE Jong.")
        return
    
    # Creating a subframe with parameters given
    Subframe(algo,f,dim, len(Data))
    
#%% Function that closes all subframes and mainframe
def close_all():
    global Data, root, info_frame
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
    root.destroy()
    
#%% Function that sets the default parameter of the subframe depending on arguments
def param(alg_id,fcn_id):
    #algorithm choice
    if alg_id==0:
        name='Blind on '
    elif alg_id==1:
        name='Hill Climbing on '
    elif alg_id==2:
        name='SOMA on '
    elif alg_id==3:
        name='DE on '
        
    #function choice
    if fcn_id==0:
        fctn = Jong_1
        name+= '1st De Jong function'
        minmax='-10;10'
    elif fcn_id==1:
        fctn = Jong_2
        name+= '2nd De Jong function'
        minmax='-2;2'
    elif fcn_id==2:
        fctn = Rastrigin
        name+= 'Rastrigin function'
        minmax='-5;5'
    elif fcn_id==3:
        fctn = Schwefel
        name+= 'Schwefel function'
        minmax='-500;500'
    elif fcn_id==4:
        fctn = Ackley
        name+= 'Ackley function'
        minmax='-20;20'
    elif fcn_id==5:
        fctn = Custom
        name+= 'Custom function'
        minmax='-10;10'
    
    return fctn, name, minmax

#%% Main frame and Data initialisation
Data = [] # List containing all subframes and their data (including plots)

root = Tk()
root.title("Evolutionary Algorithm")

#%% General options
# Algorithm select box
alg = ttk.Combobox(root, textvariable=StringVar(), width=16)
alg['values'] = ('Blind','Hill Climbing', 'SOMA', 'DE')
alg.state(["readonly"])
alg.grid(column=0, row=0, padx=5, pady=5)
alg.set('Blind')

# Function select box
fcn = ttk.Combobox(root, textvariable=StringVar(), width=16)
fcn['values'] = ('1st De Jong','2nd De Jong', 'Rastrigin', 'Schwefel', 'Ackley','Custom')
fcn.state(["readonly"])
fcn.grid(column=0, row=1, padx=5, pady=5)
fcn.set('1st De Jong')

# Default input dimension
Label(root, text="Dimension").grid(column=1, row=0, padx=5, pady=5)
dimension=StringVar()
Entry(root,textvariable=dimension,width=5).grid(column=2, row=0, padx=5, pady=5)
dimension.set('2')

# Start button that opens the calcul subframes
start_button = Button(root, text='Start new experiment', width=20, command=new_frame).grid(column=1, row=1, columnspan=2)
labwarn = Label(root, text="")
labwarn.grid(column=0, row=2, columnspan=3)


#%% End of main frame
# Bottom options
Button(root, text='Help & information', width =20, command=information_frame, bg='yellow').grid(column=0, row=3, padx=10, pady=10, columnspan=2)
Button(root,text='Close all',command=close_all).grid(column=2, row=3)

root.mainloop()