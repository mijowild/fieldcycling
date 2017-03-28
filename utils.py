import numpy.matlib
from num_string_eval import NumericStringParser

def get_x_axis(parameters):
    nsp = NumericStringParser()
    T1MX = parameters['T1MX'] # T1MX is used in the 'eval' expressions below
    # TODO: not tested yet for all cases
    if parameters['BGRD'] == 'LIST':
        temp = parameters['BLST']
        temp.replace(';', ':')
        sep_indices = [pos for pos, char in enumerate(ele) if char == ':'] # find indices of ':'
        Tini = nsp.eval(temp[:sep_indices[0]].replace('T1MX', str(T1MX)))
        Tend = nsp.eval(temp[sep_indices[0]+1:sep_indices[1]].replace('T1MX', str(T1MX)))
        npts = nsp.eval(temp[sep_indices[2]+1:].replace('T1MX', str(T1MX))) # number of points selected: can be ~= NBLK    
        if temp[sep[1]+1:sep[2]] == 'LIN':
            listx = linspace(Tini,Tend,npts);
        elif temp[sep[1]+1:sep[2]] == 'LOG':
            listx = np.logspace(np.log10(Tini),np.log10(Tend),npts);
        nrep = np.ceil(nblk/npts) # find if the time vector needs to be longer
        x = numpy.matlib.repmat(listx,1,nrep) # re-create the time vector
        x = x[:nblk] # select the portion corresponding to the number of blocs (needed if npts~=nblk)
    elif parameters['BGRD'] == 'LIN':
        Tini = nsp.eval(parameters['BINI'].replace('T1MX', str(T1MX)))
        Tend = nsp.eval(parameters['BEND'].replace('T1MX', str(T1MX)))
        x = np.linspace(Tini, Tend, nblk) # re-create the time vector
    elif parameters['BGRD'] == 'LOG':
        Tini = nsp.eval(parameters['BINI'].replace('T1MX', str(T1MX)))
        Tend = nsp.eval(parameters['BEND'].replace('T1MX', str(T1MX)))
        x = np.logspace(np.log10(Tini),np.log10(Tend),nblk) # re-create the time vector
    return x