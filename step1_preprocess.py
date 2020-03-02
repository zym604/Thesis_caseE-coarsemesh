import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# read OpenFOAM data
models_lf = ['LF_mesh20_kepsilon','LF_mesh20_komega','LF_mesh20_laminar']
path_lf = '/home/yangmo/Desktop/thesis/caseE-coarsemesh/LF_openfoam/'
# get LF features
def getfeatures(path):
    import os
    import Ofpp
    dir_xyz = '0'
    meshx_test = Ofpp.parse_internal_field(path+dir_xyz+'/ccx')
    meshy_test = Ofpp.parse_internal_field(path+dir_xyz+'/ccy')
    meshz_test = Ofpp.parse_internal_field(path+dir_xyz+'/ccz')
    meshx_test = meshx_test + 0.02
    V = Ofpp.parse_internal_field(path+dir_xyz+'/V')
    mesh_test = np.vstack([meshx_test.T,meshy_test.T,meshz_test.T]).T
    folders = next( os.walk(path) )[1]
    new_folders = [folder for folder in folders if folder.isdigit()]
    num_folders = [int(folder) for folder in new_folders]
    dir_result = str(max(num_folders))
    U = Ofpp.parse_internal_field(path+dir_result+'/U')
    gradU = Ofpp.parse_internal_field(path+dir_result+'/gradU')
    dwall = Ofpp.parse_internal_field(path+dir_result+'/dwall')
    if os.path.isfile(path+dir_result+'/k'):
        k = Ofpp.parse_internal_field(path+dir_result+'/k')
    else:
        k = np.copy(meshx_test)
        k[:] = 0
    feature = np.vstack([U.T,gradU.T,dwall.T,k.T,V.T]).T
    return feature,mesh_test

feature_lf = [getfeatures(path_lf+i+'/') for i in models_lf]
selectedfeature_lf = np.vstack([feature_lf[i][0].T for i in [0,1,2]]).T
mesh_lf = feature_lf[0][1]

# laod HF Reynolds stress
file_hf = '/home/yangmo/Desktop/thesis/caseC-2-DNS_subchannel/DNSdata/Re11000/output_new/HF_DNS.csv'
data_hf = genfromtxt(file_hf, delimiter=',')[1:,:]
mesh_hf = data_hf[:,:3]
#fileR_hf = '/home/yangmo/Desktop/thesis/caseC-2-DNS_subchannel/DNSdata/Re11000/output_new/RStress.csv'
#R_hf = genfromtxt(fileR_hf, delimiter=',')

# select overlapped region
#data_hf_overlapped = np.zeros([mesh_lf.shape[0],R_hf.shape[1]])
U_hf_overlapped = np.zeros([mesh_lf.shape[0],3])
eucd_least = np.zeros(mesh_lf.shape[0])
for i in range(mesh_lf.shape[0]):
    if i%1000==0:
        print(str(i)+'/'+str(mesh_lf.shape[0]))
    pos_lf = mesh_lf[i,:]
    dist = mesh_hf-pos_lf
    eucd = (dist[:,0]**2+dist[:,1]**2+dist[:,2]**2)**0.5
    #Rvector = R_hf[eucd.argmin(),:]
    Uvector = data_hf[eucd.argmin(),3:]
    #data_hf_overlapped[i,:] = Rvector
    U_hf_overlapped[i,:] = Uvector
    eucd_least[i] = eucd.min()
max_eucd_deviation = np.unique(eucd_least).max()
if max_eucd_deviation < 10**(-10):
    print('Good!, the maximum euclidean distance is '+str(max_eucd_deviation)+' compare tp 1e-6~10!')
    print("LF points are all covered in HF sampling data!")
else:
    print('Not good!, the maximum euclidean distance is '+str(max_eucd_deviation)+' compare tp 1e-6~10!')
    print("LF points are NOT totally covered in HF sampling data!")

#data_output = data_hf_overlapped.astype(np.float32)
data_input = selectedfeature_lf.astype(np.float32)
mesh = mesh_lf

# save data
np.savetxt("step1_data_output.csv", U_hf_overlapped)
np.savetxt("step1_data_input.csv", data_input)
np.savetxt("step1_data_mesh.csv", mesh)
np.savetxt("step1_data_U_hf.csv", U_hf_overlapped)
