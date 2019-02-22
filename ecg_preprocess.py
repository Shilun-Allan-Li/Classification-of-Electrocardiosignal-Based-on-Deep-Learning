#coding=utf-8
import os
import glob
import wfdb
import numpy as np  
dirpath='E:\\pyproject\\lab9\\ECG-incart_pt\\*.atr'
datpath='E:\\pyproject\\lab9\\ECG-incart_pt\\*.dat'
atrs_pathlist=glob.glob(dirpath)
dats_pathlist=glob.glob(datpath)


#--------------------读取数据------------------
x=wfdb.rdsamp(atrs_pathlist[0][:-4])
print(x[0])


samplist=[];atrslist=[];labellist=[]
for i in dats_pathlist:
	tempsamp=wfdb.rdsamp(i[:-4])#数据
	tempatr=wfdb.rdann(i[:-4],'atr')#注释文件
	tempsamp=tempsamp[0]
	samplist.append(tempsamp)
	atrslist.append(tempatr.__dict__['sample'])
	labellist.append(wfdb.rdann(i[:-4],'atr').__dict__['symbol'])
# print(len(samplist),len(atrslist))

np.save('E:/pyproject/lab9/samplist_1.npy',np.array(samplist[:35]))
np.save('E:/pyproject/lab9/samplist_2.npy',np.array(samplist[35:]))
np.save('E:/pyproject/lab9/atrslist.npy',np.array(atrslist))
np.save('E:/pyproject/lab9/labelslist.npy',np.array(labellist))