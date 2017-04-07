import os
import sys
import re
import pandas as pd

class StelarDataFile:
    def __init__(self, FileName, PathName):
        self.FileName=FileName
        self.PathName=PathName
        self.options='standard'
        self.datas={}
                    
    def options(self, text):
        self.options=text
        
    def adddata(self, ie, parameter, data):
        self.datas[ie]=(parameter,data)
        
    def getexp(self,ie):
        return self.datas[ie]

    def getparameter(self,ie):
        parameter, data =self.datas[ie]
        return parameter

    def getdata(self,ie):
        parameter, data = self.datas[ie]
        return data

    def get_number_of_experiments(self):
        return len(self.datas)

    def addparameter(self, ie, par, val):
        parameter, data = self.datas[ie]
        parameter[par] = val
        self.adddata(ie,parameter,data)

    def sdfimport(self):
        ie=1
        olddir=os.getcwd()
        os.chdir(self.PathName)

        with open(self.FileName,"r") as fid:
            line=fid.readline() #skip first line"
            words=['bla']
            parameters=dict()
            while 'DATA' not in words[0]:
                words = fid.readline().split('\t')
                if not words[0]: #probably eof reached
                    print('probably end of file reached')
                    break #escape the while loop
                #read the parameters of file
                if len(words)==2:
                    words[0]=re.sub('[=]','',words[0]) #get rid of '='
                    words[1]=re.sub('[\n]','',words[1])#get rid of trailing \n
                    words[1]=re.sub('[\t]','',words[1])#get rid of \t
                    try:
                        words[1]=int(words[1])                                        #ints are converted
                        exec('parameters[\''+words[0].rstrip()+'\'] = '+str(words[1]))  #and stored as int
                    except ValueError:
                        try:
                            words[1]=float(words[1])                                        #floats are converted
                            exec('parameters[\''+words[0].rstrip()+'\'] = '+str(words[1]))  #and stored as float
                        except ValueError:
                            exec('parameters[\''+words[0].rstrip()+'\'] = \''+words[1]+'\'')#else its stored as string
                    if words[0]=='TIME':
                        try:
                            parameters['TIME']=pd.to_datetime(words[1])
                        except:
                            print('Time cannot be read')
                        

                try:
                     if 'DATA' in words[0]:
                        data=[]
                        if parameters['NBLK']==0:
                            parameters.update({'NBLK': 1})
                        for i in range(0,int(parameters['NBLK']*parameters['BS'])):
                            columns=fid.readline().replace('\n','').split('\t')
                            data.append([])
                            for c, ii in zip(columns, range(0,10)):
                                data[i].append(int(c))
                        self.adddata(ie,parameters,data)
                        #print(self.getexp(ie))
                        #print(ie)
                        ie = ie + 1
                        words[0]='bla'
                        parameters=dict()
                except KeyError:
                    print('fastforward') #no experiment found
        print(str(ie)+' experiments read')
        os.chdir(olddir)
