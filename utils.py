import numpy as np
import numpy.matlib
from num_string_eval import NumericStringParser

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    popt = [A, K, C]
    return popt

def model_exp_dec(t, A, K, C):
    return A * np.exp(- K * t) + C
def fun_exp_dec(par,t,y):
    A, K, C = par
    return model_exp_dec(t, A,K,C) - y
    
def get_mag_amplitude(fid,startpoint, endpoint, nblk, bs):
    phi=np.zeros(nblk)
    for blk in range(nblk):
        start=startpoint + blk * bs-1
        end=endpoint + blk * bs
        phi[blk]=fid['magnitude'].iloc[start:end].sum() / (endpoint-startpoint)
    return phi

def get_x_axis(parameters, nblk):
    nsp = NumericStringParser()
    T1MX = parameters['T1MX'] # T1MX is used in the 'eval' expressions below
    # TODO: not tested yet for all cases
    if parameters['BGRD'] == 'LIST':
        print('BGRD = LIST')
        temp = parameters['BLST']
        temp.replace(';', ':')
        sep_indices = [pos for pos, char in enumerate(temp) if char == ':'] # find indices of ':'
        Tini = nsp.eval(temp[:sep_indices[0]].replace('T1MX', str(T1MX)))
        Tend = nsp.eval(temp[sep_indices[0]+1:sep_indices[1]].replace('T1MX', str(T1MX)))
        npts = nsp.eval(temp[sep_indices[2]+1:].replace('T1MX', str(T1MX))) # number of points selected: can be ~= NBLK    
        if temp[sep_indices[1]+1:sep_indices[2]] == 'LIN':
            listx = np.linspace(Tini,Tend,npts);
        elif temp[sep_indices[1]+1:sep_indices[2]] == 'LOG':
            listx = np.logspace(np.log10(Tini),np.log10(Tend),npts);
        nrep = np.ceil(nblk/npts) # find if the time vector needs to be longer
        x = numpy.matlib.repmat(listx,1,nrep) # re-create the time vector
        x = x[:nblk] # select the portion corresponding to the number of blocs (needed if npts~=nblk)
    elif parameters['BGRD'] == 'LIN':
        print('BGRD = LIN')
        Tini = nsp.eval(parameters['BINI'].replace('T1MX', str(T1MX)))
        Tend = nsp.eval(parameters['BEND'].replace('T1MX', str(T1MX)))
        x = np.linspace(Tini, Tend, nblk) # re-create the time vector
    elif parameters['BGRD'] == 'LOG':
        print('BGRD = LOG')
        Tini_Tend = [0, 0]
        # This has to be so complicated b/c of mixed datatype that is possible for parameters['BGRD'].
        # Documentation would be beneficial, what can occur
        for nb, b in enumerate(['BINI', 'BEND']):
            if type(parameters[b]) == type(0.0):
                Tini_Tend[nb] = parameters[b]
            elif type(parameters[b]) == type('asdf'):
                if 'T1MX' in parameters[b]:
                    Tini_Tend[nb] = nsp.eval(parameters[b].replace('T1MX', str(T1MX)))
                else:
                    Tini_Tend[nb] = nsp.eval(parameters[b])
        Tini, Tend = Tini_Tend
        x = np.logspace(np.log10(Tini),np.log10(Tend),nblk) # re-create the time vector
    return x
