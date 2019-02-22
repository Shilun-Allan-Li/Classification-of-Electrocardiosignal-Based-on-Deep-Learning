import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as signal
import pywt


samps=np.load('E:/pyproject/lab9/samplist_1.npy')
anns=np.load('E:/pyproject/lab9/annlist_1.npy')

print(samps.shape)
#(75,462600,12)
asamp=samps[0,:,0]
plt.plot(range(0,462600),asamp)
plt.show()

noise=signal.medfilt(asamp,257)
denoised_samp=asamp-noise
plt.plot(range(0,462),denoised_samp[:462])
plt.show()

# np.concatenate(s1,s2,s3,s4,axis=0)
#---------------中值滤波------------
medfilted_samp=[]
for i in range(75):
	medfilted_lead=[]
	for j in range(12):
		tempnoise=signal.medfilt(samps[i,:,j])
		temp_newsamp=samps[i,:,j]-tempnoise
		medfilted_lead.append(temp_newsamp)
	medfilted_samp.append(medfilted_lead)
print(len(medfilted_samp),len(medfilted_samp[0]),medfilted_lead[0].shape)
#75,12,(462600,)
print(anns[0].__dict__.keys())
#----------------心拍切分----------------
allheartbeats=[];alllabels=[]
for i in range(75):
	sampRids=anns[i].__dict__['sample']#--R波峰
	samlabels=anns[i].__dict__['symbol']#--R标签
	samp_beats=[]
	for j in range(len(sampRids)):
		if sampRids[j]<128:
			samlabels=samlabels[1:]
		else:
			# if sampRids[j]+128>len(samps[i,:,0]):
			# 	samlabels=samlabels[:j]
			# 	break
			# else:
			start=sampRids[j]-128
			samp_beats.append(samps[i,start:sampRids[j]+129,:])#--取R波峰前128个和后128点


	allheartbeats.extend(samp_beats)
	alllabels.extend(samlabels)
	print(len(samp_beats),len(samlabels))
print(len(allheartbeats),len(alllabels))

#----------------统计每类的样本数量-----------------

# kindsOFdiseases=[]
# for l in alllabels:
# 	if l not in kindsOFdiseases:
# 		kindsOFdiseases.append(l)
# print(kindsOFdiseases)
print(set(alllabels))
#['N','V','A','F','Q','n','R','B','S','j','+']
kindsOFdiseases=['N','V','A','F','Q','n','R','B','S','j','+']
numOFks={}
for item in kindsOFdiseases:
	numOFks[item]=alllabels.count(item)
print(numOFks)
# {'N':150361,'V':20009,"A":1943,'R':3173}

#N:正常；V:室性早搏；A：房性早搏；R：右束支传导阻滞

#-------------------丢弃数量少的样本---------------
newsamps=[];newlabels=[]
selectedls=['N','V','A','R']
for i in range(len(alllabels)):
	if alllabels[i] in selectedls:
		newsamps.append(allheartbeats[i])
		newlabels.append(alllabels[i])
allheartbeats=newsamps
alllabels=newlabels

# np.save('allheartbeats.npy',np.array(allheartbeats))
# np.save('alllabels.npy',np.array(alllabels))

#--------------------随机取样，统一样本长度并小波变换---------------
def shuffle_samp(samp,samlabels,label,num):
    '''
    leng:要多少个心拍
    '''
    import numpy as np
    import random
    tempsamps=[];templabels=[]
    leng=len(samp)
    for i in range(leng):
        if samlabels[i]==label and samp[i].shape[0]==257:
            tempsamps.append(samp[i])
            templabels.append(label)
    ids=random.sample( range(len(tempsamps)),num)#----随机选取num个下标
    tempsamps=list(np.array(tempsamps)[ids])
    print('**',len(tempsamps))#--应该输出num的值--1940
    return[tempsamps,templabels[:num]]

def get_samps(samps,labels,num):
    import random
    '''
    samps:样本
    labels:相应标签
    num:每一类保存的样本数量
    '''
    ls=['N','V','A','R']#
    shuffled_ss=[]; shuffled_ls=[]
    for i in ls:
        shuffled=shuffle_samp(samps,labels,i,num)
        shuffled_ss.extend(shuffled[0])
        shuffled_ls.extend(shuffled[1])
    ids=random.sample(range(num*4),num*4)
    
    
    print('ls,ss',len(shuffled_ls),len(shuffled_ss))
    nshuffled_ls=np.array(shuffled_ls)[ids]
    nshuffled_ss=np.array(shuffled_ss)[ids]
    print('ls,ss',nshuffled_ls.shape,nshuffled_ss.shape)
    return[shuffled_ls,shuffled_ss]
[ls,ss]=get_samps(allheartbeats,alllabels,1940)
print(len(ls),len(ss))


#--------------------小波变换-----------------
allheartbeats=np.load('allheartbeats.npy')
alllabels=np.load('alllabels.npy')
asamp=allheartbeats[0,:,0]
cA1,cD1=pywt.wavedec(asamp,'db1',level=1)#[cA1,cD1]
plt.plot(range(len(asamp)),asamp)
plt.show()
plt.plot(range(len(cA1)),cA1)
plt.show()

#-------------------对所有信号小波滤波----------------
wavelet_samps=[]
for i in range(allheartbeats.shape[0]) :
	wavelet_leads=[]
	for j in range(12):
		alead=allheartbeats[i,:,j]#---一个导联
		tempcoffs=pywt.wavedec(alead,'db1',level=1)
		wavelet_leads.append(tempcoffs[0])
	wavelet_samps.append(wavelet_leads)
print(wavelet_samps[0][0].shape)

