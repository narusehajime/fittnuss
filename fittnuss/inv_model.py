# -*- coding: cp932 -*-
import numpy as np
from scipy import interpolate as ip
import forward_model as fmodel
import time as tm
import scipy.optimize as opt
import matplotlib.pyplot as plt
from configparser import ConfigParser as scp

Nfeval = 1 #�œK���v�Z�X�e�b�v��
deposit_o = [] #�ώ@���ꂽ�͐ϕ��̌���
spoints = [] #�T���v�����O�|�C���g�̍��W
observation_x_file=[]
observation_deposit_file=[]
inversion_result_file=[]
inversion_x_file=[]
inversion_deposit_file=[]
initial_params = []
bound_values = []

def read_setfile(configfile):
    """
    �ݒ�t�@�C���i�t�@�C����configfile�j��ǂݍ����
    �����l��ݒ肷��
    """
    global observation_x_file, observation_deposit_file, inversion_result_file, inversion_x_file, inversion_deposit_file, initial_params, bound_values
    parser = scp()
    parser.read(configfile)#�ݒ�t�@�C���̓ǂݍ���

    observation_x_file = parser.get("Import File Names", "observation_x_file")
    observation_deposit_file = parser.get("Import File Names", "observation_deposit_file")
    inversion_result_file = parser.get("Export File Names", "inversion_result_file")
    inversion_x_file = parser.get("Export File Names", "inversion_x_file")
    inversion_deposit_file = parser.get("Export File Names", "inversion_deposit_file")

    #�����l�̓ǂݍ���
    Rw0 = parser.getfloat("Inversion Options", "Rw0")
    U0 = parser.getfloat("Inversion Options", "U0")
    h0 = parser.getfloat("Inversion Options", "h0")
    C0_text = parser.get("Inversion Options", "C0")
    #������𐔒l�z��ɕϊ��i�J���}��؂�j
    C0 = [float(x) for x in C0_text.split(',') if len(x) !=0]
    initial_params = [Rw0, U0, h0]
    initial_params.extend(C0)
    initial_params = np.array(initial_params)

    #���߂�l�͈̔͂�ǂݍ���
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
    
    #�t�H���[�h���f���ɂ��ݒ�𔽉f������
    fmodel.read_setfile(configfile)



def costfunction(optim_params):
    """
    �v���l�ƌv�Z�l�̃Y�����ʉ�
    """
    (x, C, x_dep, deposit_c) = fmodel.forward(optim_params)
    f = ip.interp1d(spoints, deposit_o, kind='cubic', bounds_error=False, fill_value=0.0)
    deposit_o_interp = f(x_dep)
    dep_norm = np.matrix(np.max(deposit_o_interp, axis = 1)).T
    residual = np.array((deposit_o_interp - deposit_c)/dep_norm)
    cost = np.sum((residual) ** 2)/len(x)
    return cost

def optimization(initial_params, bound_values, disp_init_cost=True, disp_result=True):
    """
    initial_params����o�����čœK���v�Z���s���D
    ����l�Ɖ����l��bound_values�Ŏw��
    �ϑ��l�ɍł��K�����錋�ʂ��o���p�����[�^�[���o�͂���
    """
    if disp_init_cost:
        #������costfunction���v�Z
        cost = costfunction(initial_params)
        print('Initial Cost Function = ', cost, '\n')
    
    #�œK���v�Z���s��
    t0 = tm.clock()
    res = opt.minimize(costfunction, initial_params, method='L-BFGS-B',\
    #res = opt.minimize(imodel.costfunction, optim_params, method='CG',\
                   bounds=bound_values,callback=callbackF,\
                   options={'disp': True})
    print('Elapsed time for optimization: ', tm.clock() - t0, '\n')

    #�v�Z���ʂ�\������
    if disp_result:
        print('Optimized parameters: ')
        print(res.x)
    
    return res

def readdata(spointfile, depositfile):
    """
    �f�[�^�̓ǂݍ��݂��s��
    """
    global deposit_o, spoints
    
    #�f�[�^���Z�b�g����
    spoints = np.loadtxt(spointfile, delimiter=',')
    deposit_o = np.loadtxt(depositfile, delimiter=',')
    
    return (spoints, deposit_o)

def save_result(resultfile, spointfile, depositfile, res):
    """
    �t��͌��ʂ�ۑ�����
    """
    #Forward Model���v�Z
    (x, C, x_dep, deposit_c) = fmodel.forward(res.x)

    #�œK����ۑ�
    fmodel.export_result(spointfile, depositfile)
    np.savetxt(resultfile, res.x)
    

def callbackF(x):
    """
    �r���o�߂̕\��
    """
    global Nfeval
    print('{0: 3d}  {1: 3.0f}   {2: 3.2f}   {3: 3.2f}   {4: 3.3f}    {5: 3.6f}'.format(Nfeval, x[0], x[1], x[2], x[3], costfunction(x)))
    Nfeval +=1

def plot_result(res):
    (x, C, x_dep, deposit_c) = fmodel.forward(res.x) #�œK�����v�Z
    cnum = fmodel.cnum #���x�K�̐����擾
    for i in range(cnum):
        plt.subplot(cnum,1,i+1) #���x�K�̐������O���t������
        plt.plot(spoints, deposit_o[i,:], marker = 'x', linestyle = 'None', label = "Observation")
        plt.plot(x_dep, deposit_c[i,:], marker = 'o', fillstyle='none', linestyle = 'None', label = "Calculation")
        plt.xlabel('Distance from the shoreline (m)')
        plt.ylabel('Deposit Thickness (m)')
        d = fmodel.Ds[i]*10**6
        gclassname='{0:.0f} $\mu$m'.format(d[0])
        plt.title(gclassname)
        plt.legend()

    plt.subplots_adjust(hspace=0.7)
    plt.show()

def inv_multistart():
    """
    �}���`�X�^�[�g�@�ɂ��t��͂��s��
    """
    
    #�����l�Ə���E������ݒ�
    read_setfile('config.ini') 

    #�ϑ��f�[�^��ǂݍ���
    (spoints, deposit_o) = readdata(observation_x_file, observation_deposit_file) #�ϑ��f�[�^��ǂݍ���
    
    #�����l�̃��X�g��ݒ�    
    initU = [2, 4, 6]
    initH = [2, 5, 8]
    initC = [[0.001, 0.001, 0.001,0.001], [0.004, 0.004, 0.004,0.004], [0.01, 0.01, 0.01, 0.01]]
    
    #�t��͂��s�����߂̏����l�̃��X�g�����
    res = []
    initparams = []
    for i in range(len(initU)):
        for j in range(len(initH)):
            for k in range(len(initC)):
                init = [initial_params[0],initU[i],initH[j]]
                init.extend(initC[k])
                initparams.append(init)

    #�����̏����l�ŋt��͂��s��
    for l in range(len(initparams)):
        res.append(optimization(initparams[l], bound_values))
    
    return res, initparams

if __name__=="__main__":
#    #�����l�Ə���E������ݒ�
#    read_setfile('config_sendai.ini') 
#
#    #�ϑ��f�[�^��ǂݍ���
#    (spoints, deposit_o) = readdata(observation_x_file, observation_deposit_file) #�ϑ��f�[�^��ǂݍ���
#
#    #�œK���v�Z���s��
#    res = optimization(initial_params, bound_values)
#
#    #���ʂ�ۑ�����
#    save_result(inversion_result_file, inversion_x_file, inversion_deposit_file, res)
#
#    #���ʂ�\��
#    plot_result(res)

    res, initparams = inv_multistart()
    print(initparams)
    print(res)