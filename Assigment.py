import marvin
from marvin.tools.maps import Maps
from marvin.tools.image import Image
from marvin import config
from marvin.tools.cube import Cube
from marvin.tools.query import Query
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# set config attributes and turn on global downloads of Marvin data
config.setRelease('DR15')
config.mode = 'local'
config.download = True


#Question 1- Marvin 

#Importing All MaNGA Data from DPRall Schema
data=pd.read_csv('CompleteTable.csv')

# my_cube1=Cube('7957-12703')
maps=Maps('7957-12703')


# my_cube2 =Cube('7443-12704')



#Condition that galaxy be over mass 10^9 solar units and have redshift between 0 and 0.1 
sample=np.where((data.loc[:,'nsa_sersic_mass']>10**9) & (data.loc[:,'z']>0) & (data.loc[:,'z']<0.1))
sample=data.iloc[sample]

#Alternatively using marvin queries 
filter='nsa.z>0 and nsa.z<0.1 and nsa.sersic_logmass > 9 and cube.quality == 0'
q=Query(search_filter=filter)
r=q.run()
print(r) 



