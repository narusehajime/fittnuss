
# coding: utf-8

# In[1]:


import numpy as np
from scipy import interpolate as ip
import forward_model as fmodel
import time as tm
import scipy.optimize as opt
from configparser import ConfigParser 


# In[3]:


Nfeval = 1 #number of epochs in optimization calculation
deposit_o = [] #Observed thickness of a tsunami deposit
spoints = [] #Location of sampling points
observation_x_file=[]
observation_deposit_file=[]
resultfile=[]
obj_func_file=[]
bound_values = []
start_params = []


# In[6]:


def read_setfile(configfile):
    """
    read setting file (config.ini) and set parameters to the inverse model
    """
    global observation_x_file, observation_deposit_file, resultfile, obj_func_file, start_params, bound_values

    parser = ConfigParser()
    parser.read(configfile)#read a setting file

    #set file names
    observation_x_file = parser.get("Import File Names", "observation_x_file")
    observation_deposit_file = parser.get("Import File Names",                                          "observation_deposit_file")
    resultfile = parser.get("Export File Names",                                        "resultfile")
    
    obj_func_file= parser.get("Export File Names",                                        "obj_func_file")
    #Read starting values
    Rw0_text = parser.get("Inversion Options", "Rw0")
    U0_text = parser.get("Inversion Options", "U0")
    H0_text = parser.get("Inversion Options", "h0")
    C0_text = parser.get("Inversion Options", "C0")
    Ds_text = parser.get("Sediments", "Ds")

    #Convert text(CSV) to ndarray
    Rw0 = [float(x) for x in Rw0_text.split(',') if len(x) !=0]
    U0 = [float(x) for x in U0_text.split(',') if len(x) !=0]
    H0 = [float(x) for x in H0_text.split(',') if len(x) !=0]
    C0 = [float(x) for x in C0_text.split(',') if len(x) !=0]
    Ds = [float(x) for x in Ds_text.split(',') if len(x) !=0]
    
    #Make a list of starting values
    for i in range(len(U0)):
        for j in range(len(H0)):
            for k in range(len(C0)):
                init = [Rw0[0],U0[i],H0[j]]
                for l in range(len(Ds)):
                    init.extend([C0[k]])
                start_params.append(init)

    #Import ranges of possible values
    Rwmax = parser.getfloat("Inversion Options", "Rwmax")
    Rwmin = parser.getfloat("Inversion Options", "Rwmin")
    Umax = parser.getfloat("Inversion Options", "Umax")
    Umin = parser.getfloat("Inversion Options", "Umin")
    hmax = parser.getfloat("Inversion Options", "hmax")
    hmin = parser.getfloat("Inversion Options", "hmin")
    Cmax_text = parser.get("Inversion Options", "Cmax")
    Cmax = [float(x) for x in Cmax_text.split(',') if len(x) !=0]
    Cmin_text = parser.get("Inversion Options", "Cmin")
    Cmin = [float(x) for x in Cmin_text.split(',') if len(x) !=0]
    bound_values_list = [(Rwmin, Rwmax), (Umin, Umax), (hmin, hmax)]
    for i in range(0, len(Cmax)):
        bound_values_list.append((Cmin[i],Cmax[i]))
    bound_values = tuple(bound_values_list)
    
    #Set the initial values to the forward model
    fmodel.read_setfile(configfile)


# In[7]:


def costfunction(optim_params):
    """
    Calculate objective function that quantifies the difference between
    field observations and results of the forward model calculation

    Fist, this function runs the forward model using a given set of parameters.
    Then, the mean square error of results was calculated.
    """
    (x, C, x_dep, deposit_c) = fmodel.forward(optim_params)
    f = ip.interp1d(x_dep, deposit_c, kind='cubic', bounds_error=False, fill_value=0.0)
    deposit_c_interp = f(spoints)
    dep_norm = np.matrix(np.max(deposit_o, axis = 1)).T
    residual = np.array((deposit_o - deposit_c_interp)/dep_norm)
    cost = np.sum((residual) ** 2)
    return cost


# In[9]:


def optimization(start_params, disp_init_cost=True, disp_result=True):
    """
    Calculate parameter set that minimize the objective function (cost function)
    Optimization is started at the starting values (initial_params). The 
    L-BFGS-B method was used for optimization with parametric boundaries defined
    by bound_values

    """
    if disp_init_cost:
        #show the value of objective function at the starting values
        cost = costfunction(start_params)
        print('Initial Cost Function = ', cost, '\n')
    
    #Start optimization by L-BFGS-B method
    t0 = tm.clock()
    res = opt.minimize(costfunction, start_params, method='L-BFGS-B',                   bounds=bound_values,callback=callbackF,                   options={'disp': True})
    print('Elapsed time for optimization: ', tm.clock() - t0, '\n')
    #Display result of optimization
    if disp_result:
        print('Optimized parameters: ')
        print(res.x)
    
    return res


# In[21]:


def readdata(spointfile, depositfile):
    """
    Read measurement dataset
    """
    global deposit_o, spoints
    
    #Set variables from data files
    spoints = np.loadtxt(spointfile, delimiter=',')
    deposit_o = np.loadtxt(depositfile, delimiter=',')
    
    return (spoints, deposit_o)


# In[22]:


def save_multiple_results(resultfile, obj_func_file, results):
    """
    Save the inversion results
    """
    #Calculate the forward model using the inversion result
    
    for i in range(len(results)):
        #Save the best result
#        np.savetxt(resultfile, results[i].x, 'ab')
#        np.savetxt(obj_func_file, results[i].fun, 'ab')
        resultfile= open('result.csv', 'a')
        np.savetxt(resultfile, np.array([results[i].x]))
        obj_func_file= open('objfunc.csv', 'a')
        np.savetxt(obj_func_file, np.array([results[i].fun]))
        resultfile.close()
        obj_func_file.close()


# In[23]:


def callbackF(x):
    """
    A function to display progress of optimization
    """
    global Nfeval
    print('{0: 3d}  {1: 3.0f}   {2: 3.2f}   {3: 3.2f}   {4: 3.3f}    {5: 3.6f}'.format(Nfeval, x[0], x[1], x[2], x[3], costfunction(x)))
    Nfeval +=1


# In[24]:


def inv_multistart():
    """
    Perform inversion using the multi-start method
    """

        
    #Read a configuration file
#    read_setfile('config_sendai.ini')
#
#    #Read the measurement data
#    (spoints, deposit_o) = readdata(observation_x_file,\
#                                    observation_deposit_file) 
#    
    result = list(map(optimization, start_params))
    return result, start_params


# In[25]:


if __name__=="__main__":
#    Set initial conditions of inverse model
    read_setfile('config_sendai.ini')

    #Read measurement data set
    (spoints, deposit_o) = readdata(observation_x_file, observation_deposit_file)

    #Conduct optimization
    res,start_params = inv_multistart()

    save_multiple_results(resultfile, obj_func_file, res)

