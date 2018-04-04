import os

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from stompy.model.delft import waq_scenario
from stompy.model.delft import io as dio

## 

run_dir="runs/wy2013a-20171016-stormwater"

map_ds=dio.read_map(os.path.join(run_dir,'sfb_dfm_v2.map'),
                    hyd=os.path.join(run_dir,'com-sfb_dfm_v2.hyd'))
dio.map_add_z_coordinate(map_ds,total_depth='TotalDepth',coord_type='sigma',
                         layer_dim='layer')
##

g=unstructured_grid.UnstructuredGrid.from_ugrid(map_ds)

##

scal=map_ds['stormwater'].isel(time=-1).values.copy()
scal[ scal==-999 ] = np.nan

scal_davg=np.nanmean(scal,axis=0)


fig=plt.figure(1)
fig.clf()
ax=plt.gca()

coll=g.plot_cells(values=scal_davg,ax=ax)
coll.set_clim([0.8,1.2])

## 

# map_ds.mesh.attrs['face_dimension']='face'


# What is needed for this to be read into visit ugrid?
# The above, but also nothing is writing out proper sigma
# coordinates.  the plugin fails to find any match for "layer".
# The plugin can only handle independent 1-D variables for
# the vertical.
# So have to assemble a sigma coordinate system, which is of course
# assuming that the input conforms to that.

# This has some -999.  Not sure why.

# Testing this out -- 
# map_ds['ebda'].attrs['_FillValue']=-999

##

# Now it plots, but the data layout is all wrong. Maybe face/layer transposed?

# the scalars are all (time,layer,face), and attributes coordinates="sigma"
# vs. the DFM output, with time, nFlowElem,laydim.
# coordinates for DFM output are "FlowElem_xcc, FlowElem_ycc"
# should set FillValue=-999 !

nc_fn='stormwater_out.nc'
os.path.exists(nc_fn) and os.unlink(nc_fn)
map_ds.to_netcdf(nc_fn)
