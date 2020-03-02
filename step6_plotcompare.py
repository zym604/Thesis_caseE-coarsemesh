import matplotlib.pyplot as plt
from f_plot2 import getOFsectiondata,getDNSsectiondata,getlinedata,corpsection
import numpy as np

# Openfoam path
mod = '/home/yangmo/Desktop/thesis/caseE-coarsemesh/ML/exp'
paths = [mod+str(i)+'/' for i in range(1,8)]

# DNS path
path_dns = '/home/yangmo/Desktop/thesis/caseC-2-DNS_subchannel/DNSdata/Re11000/output_new/'
# extract location
pos_x = 0.0385
# extract line location
rad_x = 0.001*(6.3-4.57/np.sqrt(2))
nop = 5000
y_p_LF = np.linspace(rad_x, -rad_x, num=nop)
z_p_LF = np.linspace(-rad_x, rad_x, num=nop)

# process DNS data
x_dns,y_dns,z_dns,U_dns,R_dns = getDNSsectiondata(path_dns,pos_x)
y_dns,z_dns,U_dns = corpsection(y_dns,z_dns,U_dns,y_p_LF,z_p_LF,plotornot=1,figname='DNS')
liner_dns,U_pred_dns = getlinedata(y_dns,z_dns,U_dns,y_p_LF,z_p_LF)

# process OpenFOAM data
liner_ofs = []
U_pred_ofs = []
for path in paths:
    print(path)
    #x_of,y_of,z_of,U_of = getOFsectiondata(path,pos_x)
    mesh = np.loadtxt(path+"step1_data_mesh.csv")
    #U_of = np.loadtxt(path+"step3_prediction.csv")
    U_of = np.loadtxt(path+"step3_prediction.csv")
    x_of = mesh[:,0]
    y_of = mesh[:,1]
    z_of = mesh[:,2]
    #get section data
    U = U_of
    meshx_test = x_of
    meshy_test = y_of
    meshz_test = z_of
    index = (np.isclose(meshx_test,pos_x,0.02))
    if np.unique(meshx_test[index]).shape[0] > 1:
        uniquevalues = np.unique(meshx_test[index])
        uniquevalue  = uniquevalues[np.argmin(np.abs(uniquevalues-pos_x))]
        index = (np.equal(meshx_test,uniquevalue))
        print('ERROR! use pos_x = ',uniquevalue,' instead')
    y_p = meshy_test[index]    #size should be 2959(old DNS data) / 6402 for new DNS data
    z_p = meshz_test[index]
    x_p = meshx_test[index]
    U_p = U[index]
    U_of = U_p
    x_of = x_p
    y_of = y_p
    z_of = z_p
    # check if selected points contains overlap points with slightly different x position
    if np.unique(x_p).shape[0] == 1:
        print('yes')
        print('unique values for x are:',np.unique(x_p).shape[0])
    else:
        print('ERROR!')
        print('unique values for x are:',np.unique(x_p).shape[0])
    # get corped section area
    y_of,z_of,U_of = corpsection(y_of,z_of,U_of,y_p_LF,z_p_LF,plotornot=1,figname=path)
    # get line data
    liner_of,U_pred_of = getlinedata(y_of,z_of,U_of,y_p_LF,z_p_LF)
    liner_ofs.append(liner_of)
    U_pred_ofs.append(U_pred_of)


R_max = (6.3*np.sqrt(2)-4.57)*0.001

#plot
plt.figure()
for i in range(len(paths)):
    plt.plot(liner_ofs[i],U_pred_ofs[i][:,0],label=paths[i].replace('/home/yangmo/Desktop/thesis/caseE-coarsemesh/ML/','').replace('/',''))
plt.plot(liner_dns,U_pred_dns[:,0],label="HF DNS")
plt.xlabel('channel width (m)')
plt.ylabel('velocity (m/s)')
plt.legend()
plt.ylim([8,18])
plt.xlim([-0.005,0.005])
plt.savefig('fig2_line')

#plot
plt.figure()
for i in range(len(paths)):
    plt.semilogx(liner_ofs[i][:-1]+R_max,U_pred_ofs[i][:,0][:-1],label=paths[i].replace('/home/yangmo/Desktop/thesis/caseE-coarsemesh/ML/','').replace('/',''))
plt.semilogx(liner_dns[:-1]+R_max,U_pred_dns[:,0][:-1],label="HF DNS")
plt.xlabel('distance to the wall (m)')
plt.ylabel('velocity (m/s)')
plt.xlim([0.0000001,R_max])
plt.legend()
plt.savefig('fig2_line_log')

#plot
fig = plt.figure(figsize=[5,7])
ax1 = fig.add_subplot(211)
ax1.title.set_text('(a)')
for i in range(1):
    plt.plot(liner_ofs[i],U_pred_ofs[i][:,0],label='ML prediction')
plt.plot(liner_dns,U_pred_dns[:,0],label="HF DNS")
plt.xlabel('channel width (m)')
plt.ylabel('velocity (m/s)')
plt.legend()
plt.ylim([8,18])
plt.xlim([-0.005,0.005])

#plot
ax2 = fig.add_subplot(212)
ax2.title.set_text('(b)')
for i in range(1):
    plt.semilogx(liner_ofs[i][:-1]+R_max,U_pred_ofs[i][:,0][:-1],label='ML prediction')
plt.semilogx(liner_dns[:-1]+R_max,U_pred_dns[:,0][:-1],label="HF DNS")
plt.xlabel('distance to the wall (m)')
plt.ylabel('velocity (m/s)')
plt.xlim([0.0000001,R_max])
plt.legend()
fig.tight_layout()

plt.savefig('fig2_line_log_single')

