from __future__ import print_function, division, absolute_import
import marvin
from marvin.tools.maps import Maps
from marvin.tools.image import Image
from marvin import config
from marvin.tools.cube import Cube
from marvin.tools.query import Query
from marvin.utils.datamodel.query.MPL import DR15
from marvin.utils.general.general import getSpaxel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from six import b
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
# from .base import VACMixIn

# class GZVAC(VACMixIn):
    # """Provides access to the MaNGA Galaxy Zoo Morphology VAC.

    # VAC name: MaNGA Morphologies from Galaxy Zoo

    # URL: https://www.sdss.org/dr15/data_access/value-added-catalogs/?vac_id=manga-morphologies-from-galaxy-zoo

    # Description: Returns Galaxy Zoo morphology for MaNGA galaxies. 
    # The Galaxy Zoo (GZ) data for SDSS galaxies has been split over several iterations of www.galaxyzoo.org, 
    # with the MaNGA target galaxies being spread over five different GZ data sets. In this value added catalog 
    # we bring all of these galaxies into one single catalog and re-run the debiasing code (Hart et al. 2016) in 
    # a consistent manner across the all the galaxies. This catalog includes data from Galaxy Zoo 2 (previously 
    # published in Willett et al. 2013) and newer data from Galaxy Zoo 4 (currently unpublished).

    # Authors: Coleman Krawczyk, Karen Masters and the rest of the Galaxy Zoo Team.

    # """

    # # Required parameters
    # name = "galaxyzoo"
    # description = "Returns Galaxy Zoo morphology"
    # version = {"MPL-7": "v1_0_1", "MPL-8": "v1_0_1", "DR15": "v1_0_1", "DR16": "v1_0_1"}

    # # optional Marvin Tools to attach your vac to
    # include = (marvin.tools.cube.Cube, marvin.tools.maps.Maps, marvin.tools.modelcube.ModelCube)

    # # Required method
    # def set_summary_file(self, release):
    #     ''' Sets the path to the GalaxyZoo summary file '''

    #     # define the variables to build a unique path to your VAC file
    #     self.path_params = {"ver": self.version[release]}

    #     # get_path returns False if the files do not exist locally
    #     self.summary_file = self.get_path("mangagalaxyzoo", path_params=self.path_params)

    # # Required method
    # def get_target(self, parent_object):
    #     ''' Accesses VAC data for a specific target from a Marvin Tool object '''

    #     # get any parameters you need from the parent object
    #     mangaid = parent_object.mangaid

    #     # download the vac from the SAS if it does not already exist locally
    #     if not self.file_exists(self.summary_file):
    #         self.summary_file = self.download_vac("mangagalaxyzoo", path_params=self.path_params)

    #     # Open the file using fits.getdata for extension 1
    #     data = astropy.io.fits.getdata(self.summary_file, 1)

    #     # Return selected line(s)
    #     indata = mangaid in data["mangaid"]
    #     if not indata:
    #         return "No Galaxy Zoo data exists for {0}".format(mangaid)

    #     idx = data["mangaid"] == mangaid
    #     return data[idx]

#Defining class to do linear regession using torch NN 
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

plt.ion()

#set config attributes and turn on global downloads of Marvin data
config.setRelease('DR15')
config.mode = 'local'
config.download = True


########################################################### Question 1- Marvin ############################################

#Importing All MaNGA Data from DPRall Schema
data=pd.read_csv('CompleteTable.csv')


my_cube1=Cube('7957-12703')
central_spaxel1=my_cube1.getSpaxel(0,0)
map1=my_cube1.getMaps()
image1=Image('7957-12703')


central_spaxel1.flux.plot()
# plt.show()


image1.plot()
# plt.show()



my_cube2 =Cube('7443-12704')
central_spaxel2=my_cube2.getSpaxel(0,0)
central_spaxel2.flux.plot()
# plt.show()


image2=Image('7443-12704')
image2.plot()
# plt.show()


#Condition that galaxy be over mass 10^9 solar units and have redshift between 0 and 0.1 
sample=np.where((data.loc[:,'nsa_sersic_mass']>10**9) & (data.loc[:,'z']>0) & (data.loc[:,'z']<0.1))
sample=data.iloc[sample]

#Alternatively using marvin queries 


# filter='nsa.z>0 and nsa.z<0.1 and nsa.sersic_logmass > 9 and cube.quality == 0'
# return_paramter=['mangadapdb.dapall.mangaid']
# q=Query(search_filter=filter,return_params=return_paramter,release='DR15',return_all=True)
# results=q.run()
# galaxy_list=results.getListOf('mangadapdb.dapall.mangaid',to_ndarray=True)
# np.savetxt('Query Results',galaxy_list,fmt='%s')
 
#Saved results offline and commented out query to save time when running code

galaxy_list=np.loadtxt('Query Results',dtype=str)

#Galaxy Zoo 

################################################## Question 2- Stats for Morphology and Star-Formation Activity #####################################

#Pulling mass and SFR for galaxies from Cas Jobs table
galaxy_index=np.zeros(len(galaxy_list))

for i in range (len(galaxy_list)):
    galaxy_index[i]=np.where(data.loc[:,'mangaid']==galaxy_list[i])[0][0]

galaxy_index=np.array(galaxy_index,dtype=int)



galaxies=data.iloc[galaxy_index]

mass=galaxies.loc[:,'nsa_sersic_mass']
log_mass=np.log10(mass)

SFR=galaxies.loc[:,'sfr_tot']
log_SFR=np.log10(SFR)

ha_flux=galaxies.loc[:,'emline_gflux_tot_ha_6564']

n=galaxies.loc[:,'nsa_sersic_n']


bins_mass=np.arange(np.min(log_mass),np.max(log_mass),0.1)
bins_SFR=np.linspace(np.min(log_SFR),np.max(log_SFR),num=len(bins_mass))

sersic_n_array=np.zeros([len(bins_SFR),len(bins_mass)])
for j in range(len(bins_SFR)-1):
    for i in range(len(bins_mass)-1):
        if i and j <len(bins_mass):
            a=np.where((log_SFR>bins_SFR[j]) & (log_SFR<bins_SFR[j+1]) & (log_mass>bins_mass[i]) & (log_mass<bins_mass[i+1]))
        else:
            a=np.where((log_SFR>bins_SFR[j-1]) & (log_SFR<bins_SFR[j]) & (log_mass>bins_mass[i-1]) & (log_mass<bins_mass[i]))
        b=np.mean(n.iloc[a])
        sersic_n_array[i,j]=b



plt.figure()
#Plotting the relevant data 
plt.title('log SFR vs log Mass of Galaxies in MaNGA')
plt.xlabel(r'$log(M/M_{\odot})$')
plt.ylabel(r'$log(SFR/M_{\odot})$')
# plt.hist2d(log_mass,log_SFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.scatter(log_mass,log_SFR, c=np.log10(ha_flux), cmap='magma', alpha=0.9)
plt.colorbar().set_label('log(Ha Flux)')
plt.show()


plt.figure()
plt.title('log SFR vs log Mass of Galaxies in MaNGA')
plt.xlabel(r'$log(M/M_{\odot})$')
plt.ylabel(r'$log(SFR/M_{\odot})$')
# plt.hist2d(log_mass,log_SFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.scatter(log_mass,log_SFR, c=n.ravel(), cmap='viridis', alpha=0.9)
plt.colorbar().set_label('Sersic n')
plt.show()


####################################### Question 3- Machine Learning ################################################################### 

#Prepare Data

sSFR=SFR/mass #Calculating sSFR

log_sSFR=np.log10(sSFR)

log_sSFR=np.array(log_sSFR,dtype=np.float32).reshape(-1,1) #Changing data from pd dataframe to arrays to make it easier to feed to Torch 
n=np.array(n,dtype=np.float32).reshape(-1,1)

n=np.log10(n)


#Model 
inputDim=1
outputDim=1 
learningRate=0.1

model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

#Training Loop
epochs=2000

epoch_array=np.zeros(epochs)
loss_array=np.zeros(epochs)
for epoch in range(epochs): #Forward Pass and loss
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(log_sSFR).cuda())
        labels = Variable(torch.from_numpy(n).cuda())
    else:
        inputs = Variable(torch.from_numpy(log_sSFR))
        labels = Variable(torch.from_numpy(n))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters, (backward pass)
    loss.backward()

    # update parameters
    optimizer.step()

    epoch_array[epoch]=epoch 
    loss_array[epoch]=loss.item()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(log_sSFR).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(log_sSFR))).data.numpy()
    print(predicted)



plt.clf()
plt.plot(log_sSFR, n, 'go', label='True data', alpha=0.5)
plt.plot(log_sSFR, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
plt.figure()

plt.title('Loss vs Epoch for 1D Linear Regression')
plt.plot(epoch_array,loss_array)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.figure()


################################################ 2 input 1 ouput linear regression with no CUDA #########################################

#Two input and one output linear regression 

log_mass_sfr=pd.concat([log_mass,log_SFR],axis=1) #Combining both input variables into one df

log_mass_sfr=np.array(log_mass_sfr,dtype=np.float32).transpose() #Formatting properly for pytorch 

#Defining the model 
# First we define the trainable parameters A and b 
A = torch.randn((1, 2), requires_grad=True)
b = torch.randn(1, requires_grad=True)

def model(x_input):
    return A.mm(x_input) + b

#Chosing a loss function
loss= torch.nn.SmoothL1Loss()

### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = torch.optim.Adam([A, b], lr=0.1)

epoch_array2=np.zeros(2000)
loss_array2=np.zeros(2000)
# Main optimization loop
for t in range(2000): #t is epochs

    inputs = Variable(torch.from_numpy(log_mass_sfr))
    labels = Variable(torch.from_numpy(n))

    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(inputs)
    # See how far off the prediction is
    current_loss = loss(y_predicted, labels)
    # Compute the gradient of the loss with respect to A and b.
    current_loss.backward()
    # Update A and b accordingly.
    optimizer.step()

    epoch_array2[t]=t 
    loss_array2[t]=current_loss.item()
    print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")

plt.plot(log_mass_sfr[0,:], n, 'go', label='True data log n vs Mass', alpha=0.5)
plt.plot(log_mass_sfr[1,:], n, 'bo', label='True data log n vs SFR', alpha=0.5)
plt.plot(log_mass_sfr[0,:], A.detach().numpy()[0][0]*log_mass_sfr[0,:]+b.item(), '--', label='Predictions of log n vs Mass', alpha=0.5)
plt.plot(log_mass_sfr[1,:], A.detach().numpy()[0][1]*log_mass_sfr[1,:]+b.item(), '--', label='Predictions of log n vs SFR', alpha=0.5)
plt.legend(loc='best')
plt.show()
plt.figure()

plt.title('Loss vs Epoch for 2D Linear Regression')
plt.plot(epoch_array2,loss_array2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.figure()


################################################ 2 input 1 ouput linear regression with CUDA #########################################

#Model 
inputDim=2
outputDim=1 
learningRate=0.1

log_mass_sfr=pd.concat([log_mass,log_SFR],axis=1) #Combining both input variables into one df

log_mass_sfr=np.array(log_mass_sfr,dtype=np.float32) #Formatting properly for pytorch 


model = torch.nn.Linear(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

#Training Loop
epochs=2000

epoch_array3=np.zeros(epochs)
loss_array3=np.zeros(epochs)
for epoch in range(epochs): #Forward Pass and loss
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(log_mass_sfr).cuda())
        labels = Variable(torch.from_numpy(n).cuda())
    else:
        inputs = Variable(torch.from_numpy(log_mass_sfr))
        labels = Variable(torch.from_numpy(n))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters, (backward pass)
    loss.backward()

    # update parameters
    optimizer.step()

    epoch_array3[epoch]=epoch 
    loss_array3[epoch]=loss.item()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(log_mass_sfr).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(log_mass_sfr))).data.numpy()
    print(predicted)



plt.clf()
plt.plot(log_mass_sfr[:,1], n, 'go', label='True data (n vs Mass)', alpha=0.5)
plt.plot(log_mass_sfr[:,0], n, 'bo', label='True data (n vs SFR)', alpha=0.5)
plt.plot(log_mass_sfr[:,1], predicted, '--', label='Predictions (n vs Mass)', alpha=0.5)
plt.plot(log_mass_sfr[:,0], predicted, '--', label='Predictions (n vs SFR)', alpha=0.5)
plt.legend(loc='best')
plt.show()
plt.figure()

plt.title('Loss vs Epoch for 2D Linear Regression (CUDA)')
plt.plot(epoch_array3,loss_array3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.figure()

######################### 60-20-20 split for train-test-validate##################################

#Working with making a neural net with a hidden layer

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        x1 = self.linear(x)
        out = self.linear(x1)
        return out


#Importing clean data 

sSFR=SFR/mass #Calculating sSFR
log_sSFR=np.log10(sSFR)
log_sSFR=np.array(log_sSFR,dtype=np.float32)

n=galaxies.loc[:,'nsa_sersic_n']
n=np.log10(n)
n=np.array(n,dtype=np.float32)

#Creating paritions in the data to define the 60-20-20 split 
train_sSFR=torch.from_numpy(log_sSFR[0:2183].reshape(-1,1)).to('cuda:0')
test_sSFR=torch.from_numpy(log_sSFR[2183:2910].reshape(-1,1)).to('cuda:0')
validate_sSFR=torch.from_numpy(log_sSFR[2910:3638].reshape(-1,1)).to('cuda:0')

train_n=torch.from_numpy(n[0:2183].reshape(-1,1)).to('cuda:0')
test_n=torch.from_numpy(n[2183:2910].reshape(-1,1)).to('cuda:0')
validate_n=torch.from_numpy(n[2910:3638].reshape(-1,1)).to('cuda:0')

#Creating Tensor Datasets 
train_dataset= TensorDataset(train_sSFR,train_n)
test_dataset= TensorDataset(test_sSFR,test_n)
validate_dataset=TensorDataset(validate_sSFR,validate_n)

#Putting these partitions into a pytorch DataLoader 
train_dataloader= DataLoader(train_dataset,batch_size=len(train_sSFR),shuffle=True)
test_dataloader=DataLoader(test_dataset, batch_size=len(test_sSFR), shuffle=True)
validate_dataloader= DataLoader(validate_dataset, batch_size=len(validate_sSFR), shuffle=True)

#Model 
inputDim=1
outputDim=1 
learningRate=0.1

model = torch.nn.Linear(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

#Training Loop
epochs=2000

epoch_array4=np.zeros(epochs)
loss_array4=np.zeros(epochs)
for epoch in range(epochs): #Forward Pass and loss
    for xb,yb in train_dataloader:
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(xb)

        # get loss for the predicted output
        loss = criterion(outputs, yb)
        print(loss)
        # get gradients w.r.t to parameters, (backward pass)
        loss.backward()

        # update parameters
        optimizer.step()

        epoch_array4[epoch]=epoch 
        loss_array4[epoch]=loss.item()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(train_sSFR)
    print(predicted)
plt.plot(train_sSFR.cpu().numpy(), train_n.cpu().numpy(), 'go', label='Training data', alpha=0.5)
plt.plot(train_sSFR.cpu().numpy(), predicted.cpu().numpy(), '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
plt.figure()


#Testing the model with test datamodel 

predicted_test=model(test_sSFR)

plt.plot(test_sSFR.cpu().numpy(), test_n.cpu().numpy(), 'bo', label='Test data', alpha=0.5)
plt.plot(test_sSFR.cpu().numpy(), predicted_test.cpu().detach().numpy(), '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
plt.figure()

predicted_validate=model(validate_sSFR)

plt.plot(validate_sSFR.cpu().numpy(), validate_n.cpu().numpy(), 'ro', label='Validate data', alpha=0.5)
plt.plot(validate_sSFR.cpu().numpy(), predicted_validate.cpu().detach().numpy(), '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
plt.figure()

plt.title('Loss vs Epoch for 1D Linear Regression (CUDA) with 1 Hidden Layer')
plt.plot(epoch_array4,loss_array4)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.figure()
