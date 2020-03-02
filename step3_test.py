import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

# print start time
time_s = time.time()
print( "Start time = "+time.ctime())

# load step1 data and other HF data
data_output = np.loadtxt("step1_data_output.csv")
data_input = np.loadtxt("step1_data_input.csv")
data_input1 = data_input[:,:15]
data_input2 = data_input[:,30:]
data_input = np.vstack([data_input1.T,data_input2.T]).T
mesh = np.loadtxt("step1_data_mesh.csv")

# seperate training, development & test data
sp = 0.018+0.02
t_d_xyz  = mesh[mesh[:,0]<sp,:]
test_xyz = mesh[mesh[:,0]>=sp,:]
t_d_inp  = data_input[mesh[:,0]<sp,:]
test_inp = data_input[mesh[:,0]>=sp,:]
t_d_oup  = data_output[mesh[:,0]<sp,:]
test_oup = data_output[mesh[:,0]>=sp,:]
# random split training and development set
nop = t_d_xyz.shape[0]
indices = np.random.RandomState(seed=42).permutation(nop)
bp = np.int(nop*0.8)
train_idx, dev_idx = indices[:bp], indices[bp:]
train_xyz, dev_xyz = t_d_xyz[train_idx,:], t_d_xyz[dev_idx,:]
train_inp, dev_inp = t_d_inp[train_idx,:], t_d_inp[dev_idx,:]
train_oup, dev_oup = t_d_oup[train_idx,:], t_d_oup[dev_idx,:]
print( "finish load and seperating data!")

# read data
x_train = train_inp
y_train = train_oup
x_dev = dev_inp
y_dev = dev_oup
x_test = test_inp
y_test = test_oup
#inp = inp*[4,100,1,4,0.04,1]
#oup = oup*500
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_dev = x_dev.astype(np.float32)
y_dev = y_dev.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
# Hyper Parameters
input_size = x_train.shape[1]
hidden_size = 500
output_size = y_train.shape[1]
num_epochs = 5000
learning_rate = 0.001
error = 1
relativerange = 500
relativeerror = 0.0001
dropout_rate = 0.5
loadtrainedmodel = True
onlyplot = False

# Linear Regression Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.l1 = nn.ReLU()
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Tanh()
        self.l4 = nn.ELU()
        self.l5 = nn.Hardshrink()
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.l3(out)
        out = self.ln(out)
        #out = self.dp(out)
        out = self.l1(out)
        out = self.ln2(out)
        out = self.l1(out)
        out = self.fc2(out)
        return out

model = Net(input_size, hidden_size, output_size, dropout_rate)
# print model summary
nodata = np.prod(x_train.shape)
noparas = sum([param.nelement() for param in model.parameters()])
print("Total number of data elements:"+str(nodata))
print("Total number of parameters   :"+str(noparas))
for name, param in model.named_parameters():
    print( name, "\t", param.nelement(), "\t\t", param.data.shape)
if noparas>nodata:
    print( "Use too much neurons!!!")
else:
    print( "Network is OK!")

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# load trained model parameters
if loadtrainedmodel:
    model.load_state_dict(torch.load('step2_model.pkl'))
    optimizer.load_state_dict(torch.load('step2_optimizer.pkl'))
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

###### GPU
if torch.cuda.is_available():
    print( "We are using GPU now!!!")
    model = model.cuda()

loss_values = []
# Train the Model 
for epoch in range(num_epochs):
    if onlyplot:
        break
    # Convert numpy array to torch Variable
    if torch.cuda.is_available():
        inputs  = Variable(torch.from_numpy(x_train).cuda())
        targets = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs  = Variable(torch.from_numpy(x_train))
        targets = Variable(torch.from_numpy(y_train))
    # Forward + Backward + Optimize
    optimizer.zero_grad()  
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    # use another stop criterion
    loss_values.append(loss.item())
    avgloss = np.mean(loss_values[-relativerange:])
    lastloss = loss_values[-1]
    relat_error = abs(lastloss-avgloss)/avgloss
    if (epoch+1) % 100 == 0:
        print ('Epoch [%d/%d],\t Loss: %.4f,\t relative error: %.4f' 
               %(epoch+1, num_epochs, loss.item(), relat_error))
    if relat_error < relativeerror and epoch>relativerange and lastloss< error:
        print ('Epoch [%d/%d],\t Loss: %.4f,\t relative error: %.4f' 
               %(epoch+1, num_epochs, loss.item(), relat_error))
        break

# print end time
time_e = time.time()
print ("End time = "+time.ctime())
totaltime = time_e - time_s
print ("total used time (s) = "+str(totaltime))
print ("total used time (min) = "+str(totaltime/60.0))

# Save the Model
torch.save(model.state_dict(), 'step3_model.pkl')
torch.save(optimizer.state_dict(), 'step3_optimizer.pkl')

# Plot learning curve
plt.figure()
#plt.plot(np.array(loss_values), markersize=1, marker=".", linewidth=1)
plt.plot(np.array(loss_values))
plt.ylim(0,lastloss*4)
plt.ylabel("training loss")
plt.xlabel("iteration")
plt.savefig("step3_loss", bbox_inches='tight')
print( "***learning curve plotted!!!")

# plot the graph - training, development and test set
if torch.cuda.is_available():
    predicted1 = model(Variable(torch.from_numpy(x_train).cuda())).data.cpu().numpy()
    predicted2 = model(Variable(torch.from_numpy(x_dev  ).cuda())).data.cpu().numpy()
    predicted3 = model(Variable(torch.from_numpy(x_test ).cuda())).data.cpu().numpy()
else:
    predicted1 = model(Variable(torch.from_numpy(x_train))).data.numpy()
    predicted2 = model(Variable(torch.from_numpy(x_dev  ))).data.numpy()
    predicted3 = model(Variable(torch.from_numpy(x_test ))).data.numpy()
train_error = ((predicted1 - y_train)**2).mean()
dev_error = ((predicted2 - y_dev)**2).mean()
test_error = ((predicted3 - y_test)**2).mean()

plt.figure(figsize=(10,7))
titles=["Ux","Uy","Uz"]
for i in range(y_test.shape[1]):
    plt.subplot(2,y_test.shape[1],i+1)
    ms = 0.5
    plt.scatter(y_train[:,i],predicted1[:,i],s=ms,label="training set")
    plt.scatter(y_dev[:,i]  ,predicted2[:,i],s=ms,label="development set")
    plt.scatter(y_test[:,i] ,predicted3[:,i],s=ms,label="test set")
    plt.title(titles[i])
lgnd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
lossmessgae = "MSEs are:\n"+"Training set: "+str(train_error)+"\nDevelop set: "+str(dev_error)+"\nTest set      : "+str(test_error)
plt.annotate(lossmessgae, xy=(1.05, 0.2), xycoords='axes fraction')
plt.savefig("step3_compare", bbox_inches='tight')

# calculate variance
variance = dev_error/train_error
print( "variance = " + str(variance))

from torchviz import make_dot
# Convert numpy array to torch Variable
if torch.cuda.is_available():
    inputs  = Variable(torch.from_numpy(x_train).cuda())
    targets = Variable(torch.from_numpy(y_train).cuda())
else:
    inputs  = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))
dot = make_dot(outputs)
dot.format = 'png'
dot.render("step3_network_structure")

# save data
data_output[mesh[:,0]>=sp,:] = predicted3
np.savetxt("step3_prediction.csv", data_output)
