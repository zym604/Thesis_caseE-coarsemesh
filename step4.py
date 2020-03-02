import numpy as np
import matplotlib.pyplot as plt

# load data
#data_output = np.loadtxt("step1_data_output.csv")
data_output = np.loadtxt("step3_prediction.csv")
data_input = np.loadtxt("step1_data_input.csv")
mesh = np.loadtxt("step1_data_mesh.csv")
U_hf_overlapped = np.loadtxt("step1_data_U_hf.csv")
R_hf_overlapped = np.loadtxt("step1_data_output.csv")
print('finish loading HF data')

# read OpenFOAM data
path = '/home/yangmo/Desktop/thesis/caseC-2-DNS_subchannel/takeback_newest_onlypredictionregion/LF_mesh15_kepsilon_validation/'
# get LF features
def getfeatures(path):
    import Ofpp
    dir_xyz = '0'
    meshx_test = Ofpp.parse_internal_field(path+dir_xyz+'/ccx')
    meshy_test = Ofpp.parse_internal_field(path+dir_xyz+'/ccy')
    meshz_test = Ofpp.parse_internal_field(path+dir_xyz+'/ccz')
    meshx_test = meshx_test + 0.02 + 0.018
    mesh_test = np.vstack([meshx_test.T,meshy_test.T,meshz_test.T]).T
    return mesh_test

mesh_lf = getfeatures(path)
print('finish loading LF mesh data')

# get test region R
ind_total = []
for i in range(mesh_lf.shape[0]):
    if (i%1000 == 0):
        print(str(i)+'/'+str(mesh_lf.shape[0]))
    v = mesh_lf[i,:]
    res = mesh - v
    res_1d = (res[:,0]**2+res[:,1]**2+res[:,2]**2)**0.5
    ind = np.argmin(abs(res_1d))
    ind_total.append(ind)
    
mesh_testregion = mesh[ind_total]
pred_testregion = data_output[ind_total]
F_testregion = data_input[ind_total]
U_testregion = U_hf_overlapped[ind_total]
R_testregion = R_hf_overlapped[ind_total]

# save data
np.savetxt("step4_data_mesh.csv", mesh_testregion)
np.savetxt("step4_data_output.csv", pred_testregion)
np.savetxt("step4_data_input.csv", F_testregion)
np.savetxt("step4_data_U_hf.csv", U_testregion)
np.savetxt("step4_data_R_hf.csv", R_testregion)