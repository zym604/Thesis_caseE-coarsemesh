import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# test area
sp = 0.018+0.02

# Openfoam path
mod = '/home/yangmo/Desktop/thesis/caseE-coarsemesh/ML/exp'
paths = [mod+str(i)+'/' for i in range(1,8)]

# DNS path
from numpy import genfromtxt
path_dns = '/home/yangmo/Desktop/thesis/caseE-coarsemesh/ML/'
xyz_dns = genfromtxt(path_dns+r'step1_data_mesh.csv')
U_dns   = genfromtxt(mod+'1/'+r'step1_data_U_hf.csv')
index   = xyz_dns[:,0]>=sp
xyz_dns = xyz_dns[index,:]
U_dns   = U_dns[index,:]

# process OpenFOAM data
U_pred_ofs = []
for path in paths:
    print(path)
    xyz_of = genfromtxt(path+'step4_data_mesh.csv')
    #xyz_of = xyz_of[index,:]
    U_of = genfromtxt(path+'step4_data_output.csv')
    #U_of = U_of[index,:]
    # get corped section area
    U_pred_ofs.append(U_of)

# get LF flow features
PF= genfromtxt(path_dns+'exp'+str(1)+'/'+'step4_data_input.csv')

# find similar points (similar in flow feature but different in position)
PF_model1 = PF[:,:15]
PF_model2 = PF[:,15:30]
PF_model3 = PF[:,30:]
PF_model12 = PF[:,:30]
PF_model123 = PF
U1 = U_pred_ofs[1]
U2 = U_pred_ofs[2]
U3 = U_pred_ofs[3]
U12 = U_pred_ofs[4]
U123 = U_pred_ofs[0]

ind_A = 10
pos_A = xyz_dns[ind_A]
PF1_A = PF_model1[ind_A]
PF2_A = PF_model2[ind_A]
PF3_A = PF_model3[ind_A]
PF12_A = PF_model12[ind_A]
PF123_A = PF_model123[ind_A]

# find point B that has similar PF_model2 like point A
ind_B = ind_A+51
pos_B = xyz_dns[ind_B]
PF1_B = PF_model1[ind_B]
PF2_B = PF_model2[ind_B]
PF3_B = PF_model3[ind_B]
PF12_B = PF_model12[ind_B]
PF123_B = PF_model123[ind_B]
fig = plt.figure(figsize=[9,8])
ax = plt.subplot(221)
plt.plot(PF1_A,PF1_B,'.-',label='$k-\epsilon$ model')
plt.plot(PF2_A,PF2_B,'.-',label='$k-\omega$ model')
plt.plot(PF3_A,PF3_B,'.-',label='laminar')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
lim_left = min(xlim[0],ylim[0])
lim_right = max(xlim[1],ylim[1])
x = np.linspace(lim_left,lim_right)
ax.plot(x, x, 'k--', label='45 degree line')
plt.xlim([x.min(),x.max()])
plt.ylim([x.min(),x.max()])
plt.xlabel('ML model input on point A')
plt.ylabel('ML model input on point B')
plt.legend(framealpha=0.5)
ax2 = plt.subplot(222)
plt.plot(U_dns[ind_A],U_dns[ind_B],'r.-',label='HF data')
plt.plot(U1[ind_A],U1[ind_B],'.-',label='$k-\epsilon$ model')
plt.plot(U2[ind_A],U2[ind_B],'.-',label='$k-\omega$ model')
plt.plot(U3[ind_A],U3[ind_B],'.-',label='laminar')
xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
lim_left = min(xlim[0],ylim[0])
lim_right = max(xlim[1],ylim[1])
x = np.linspace(lim_left,lim_right)
ax2.plot(x, x, 'k--', label='45 degree line')
plt.xlim([x.min(),x.max()])
plt.ylim([x.min(),x.max()])
plt.xlabel('ML model output on point A')
plt.ylabel('ML model output on point B')
plt.legend(framealpha=0.5)
# 3D
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(xyz_of[:,0],xyz_of[:,1],xyz_of[:,2],s=0.1)
ax3.scatter(pos_A[0],pos_A[1],pos_A[2],s=40,label='point A')
ax3.scatter(pos_B[0],pos_B[1],pos_B[2],s=40,label='point B')
plt.legend()
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
ax3.set_zlabel('z axis')
#ax3.set_ylim(pos_A[1]-0.0,pos_A[1]+0.01)
ax3.view_init(30, -35)
# bar
E_1_A = ((U_dns[ind_A]-U1[ind_A])**2).mean()
E_1_B = ((U_dns[ind_B]-U1[ind_B])**2).mean()
E_2_A = ((U_dns[ind_A]-U2[ind_A])**2).mean()
E_2_B = ((U_dns[ind_B]-U2[ind_B])**2).mean()
E_3_A = ((U_dns[ind_A]-U3[ind_A])**2).mean()
E_3_B = ((U_dns[ind_B]-U3[ind_B])**2).mean()
E_12_A = ((U_dns[ind_A]-U12[ind_A])**2).mean()
E_12_B = ((U_dns[ind_B]-U12[ind_B])**2).mean()
E_123_A = ((U_dns[ind_A]-U123[ind_A])**2).mean()
E_123_B = ((U_dns[ind_B]-U123[ind_B])**2).mean()
E_1 = np.mean([E_1_A,E_1_B])
E_2 = np.mean([E_2_A,E_2_B])
E_3 = np.mean([E_3_A,E_3_B])
E_12 = np.mean([E_12_A,E_12_B])
E_123 = np.mean([E_123_A,E_123_B])
ax4 = plt.subplot(224)
bar_width = 0.35
index = np.arange(4)
#plt.barh(index,[E_1_A,E_2_A,E_3_A,E_12_A,E_123_A],bar_width)
#plt.barh(index+bar_width,[E_1_B,E_2_B,E_3_B,E_12_B,E_123_B],bar_width)
plt.barh(index,[E_3,E_2,E_1,E_123],bar_width)
plt.yticks(index + bar_width/2, ('model3', 'model2', 'model1', 'model1+2+3'))
lossmessgae = "model1 : $k-\epsilon$ model \nmodel2 : $k-\omega$ model \nmodel3 : laminar"
plt.annotate(lossmessgae, xy=(0.5, 0.8), xycoords='axes fraction')
plt.tight_layout()
plt.savefig('regional_analysis')