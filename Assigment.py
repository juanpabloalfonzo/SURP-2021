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
import torch
from torch.autograd import Variable
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
config.setRelease('DR16')
config.mode = 'local'
config.download = True


#Question 1- Marvin 

#Importing All MaNGA Data from DPRall Schema
data=pd.read_csv('CompleteTable.csv')


my_cube1=Cube('7957-12703')
central_spaxel1=my_cube1.getSpaxel(0,0)
map1=my_cube1.getMaps()
image1=Image('7957-12703')


central_spaxel1.flux.plot()
plt.show()


image1.plot()
plt.show()



my_cube2 =Cube('7443-12704')
central_spaxel2=my_cube2.getSpaxel(0,0)
central_spaxel2.flux.plot()
plt.show()


image2=Image('7443-12704')
image2.plot()
plt.show()


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

#Question 2- Stats for Morphology and Star-Formation Activity 

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

#Plotting the relevant data 
plt.title('log SFR vs log Mass of Galaxies in MaNGA')
plt.xlabel(r'$log(M/M_{\odot})$')
plt.ylabel(r'$log(SFR/M_{\odot})$')
plt.scatter(log_mass,log_SFR, c=ha_flux, vmin=-2, vmax=-0.8, cmap='viridis', alpha=0.1)
plt.hist2d(log_mass,log_SFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.colorbar().set_label('Ha Flux')
plt.show()

plt.title('log SFR vs log Mass of Galaxies in MaNGA')
plt.xlabel(r'$log(M/M_{\odot})$')
plt.ylabel(r'$log(SFR/M_{\odot})$')
plt.scatter(log_mass,log_SFR, c=n, vmin=-2, vmax=-0.8, cmap='viridis', alpha=0.1)
plt.hist2d(log_mass,log_SFR, cmap='viridis', bins=(np.linspace(7,13,51),np.linspace(-5.5,1,51)))
plt.colorbar().set_label('Sersic n')
plt.show()



#Question 3- Machine Learning 
