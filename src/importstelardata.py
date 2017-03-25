@profile
def ImportStelarData(FileName, PathName, options=0):
    """import sdf data files from Stelar machines"""
    import os
    import sys
    import json
    import gzip
    import shutil
    import numpy
    import re
    olddir=os.getcwd()
    os.chdir(PathName)
    ie=0; #experiment number
    

    ###open and read the sdf file
    #the sdf is a repetition of numerical data, separated by a parameter list'
    with open(FileName+'.json','w') as fout:
                fout.close
    parameters=dict()
    with open(FileName,"r") as fid:
        line=fid.readline() #skip first line"
        while fid!=None:
            ie = ie + 1
            words=['']
            while 'DATA' not in words[0]:
                words = fid.readline().split('\t')
                if not words[0]:
                    print('escape')
                    break #escape the while loop
                if len(words)==2:
                    words[0]=re.sub('[=]','',words[0]) #get rid of '='
                    words[1]=re.sub('[\n]','',words[1])#get rid of trailing \n                
                    try:
                        words[1]=float(words[1])                                        #floats are converted
                        exec('parameters[\''+words[0].rstrip()+'\'] = '+str(words[1]))  #and stored as float
                    except ValueError:
                        exec('parameters[\''+words[0].rstrip()+'\'] = \''+words[1]+'\'')#else its stored as string
            if not words[0]:
                ie=ie-1
                break
            if parameters['NBLK']==0:
                parameters.update({'NBLK': 1})
            data=[]
            #data.append([[None]*6]*int(parameters['NBLK']*parameters['BS']))
            
            for i in range(0,int(parameters['NBLK']*parameters['BS'])):
                columns=fid.readline().replace('\n','').split('\t')
                data.append([])
                for c, ii in zip(columns, range(0,10)):
                    data[i].append(int(c))
            paramlist=(ie,parameters,data)
            with open(FileName+'.json','a') as fout:
                json.dump(paramlist,fout,ensure_ascii=False,indent=4)
                
            
    print('fool '+str(sys.getsizeof(paramlist)))
    with open(FileName+'.json','rb') as fin:
        with gzip.open(FileName+'.json.gz','wb') as fout:
            shutil.copyfileobj(fin, fout)
    os.chdir(olddir)
    

ImportStelarData('297K.sdf',r'D:\arbeit\python\fc data')
