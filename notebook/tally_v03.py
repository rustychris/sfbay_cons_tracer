"""
2017-11-28: v03 - rerun, using recent passive tracer runs on
 the DFM grid.  Update tally regions to v01 in order to properly
 cover some small tributaries which were omitted in previous grid.

v02: Sum for all outputs, then write fractions for several seasons.
v01: Switch to DWAQ output which has all of the POTWs, regenerate outputs.

"""
# coding: utf-8

from __future__ import print_function

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from shapely import geometry
import pandas as pd

from stompy.spatial import wkb2shp
from stompy.plot import plot_wkb, plot_utils
from stompy.model.delft import dfm_grid
import stompy.model.delft.io as dio

from stompy import utils
import stompy.plot.cmap as scmap


version='v03' # version of the output - should match this file

## 

region_shp='tally_regions_v01.shp'
regions=wkb2shp.shp2geom(region_shp)


## 

base_dir='/opt/data/dwaq/sfbay_constracer/runs/'

runs=glob.glob(os.path.join(base_dir,"wy2013a-*"))

run0=runs[0]

flowgeom=xr.open_dataset(os.path.join(run0,'flowgeom.nc'))
g=dfm_grid.DFMGrid(flowgeom)
dfm_grid.cleanup_multidomains(g)

if 0:
    plt.figure(1).clf()
    g.plot_edges(color='k',lw=0.6)
    for poly in regions['geom']:
        plot_wkb.plot_polygon(poly)

## 

# select cells for each region:
# choose based on centroid of each cell to avoid double-counting
cc=g.cells_centroid()
ccp=[geometry.Point(p) for p in cc]
cell_regions=np.zeros(g.Ncells(),'i4')-1

for region_i in range(len(regions)):
    poly=regions['geom'][region_i]
    in_poly=np.array([poly.contains(p) for p in ccp],'b1')
    cell_regions[in_poly]=region_i
    print("%s => %d cells"%( regions['name'][region_i], np.sum(in_poly)))

# little faster than scanning bitmask
cell_region_idxs=[np.nonzero(cell_regions==ri)[0]
                  for ri in range(len(regions))]

if 0:
    plt.figure(2).clf()
    g.plot_cells(values=cell_regions,lw=0.2,cmap='jet')
    for poly in regions['geom']:
        plot_wkb.plot_polygon(poly,fc='none',ec='k',lw=0.1)
    plt.axis('equal')

## 


skip=['localdepth','surf','totaldepth','depth','volume']
time_range=[np.datetime64('2012-10-01 00:00'),
            np.datetime64('2013-10-01 00:00')]


# iterate over the runs
def process_run(run):
    """
    run: path to directory containing dwaq run.
    return a list of dictionaries with region_i, region_name,
    timestamps,tracer names, mass in the region, and water volume
    in region.
    """
    print("Processing %s"%run)
    map_ds=dio.read_map(os.path.join(run,'sfb_dfm_v2.map'),
                        os.path.join(run,'com-sfb_dfm_v2.hyd'),
                        include_grid=False)
    scalars=[]

    for v in map_ds.data_vars:
        if v.lower() not in skip and map_ds[v].ndim == 3:
            scalars.append(v)
    print("Will create output for %d tracers: %s"%(len(scalars),
                                                   " ".join(scalars)))

    # Tally the mass contribution of one POTW to one region at one timestep:

    times_dt64=map_ds.time

    ti_start,ti_stop=np.searchsorted(times_dt64,time_range)
    recs=[]
    for ti in range(ti_start,ti_stop):
        print(times_dt64[ti].values,end=" ")
        for si in range(len(scalars)):
            print(scalars[si][:5],end=' ')
            # integrate water columns first
            scal=map_ds[scalars[si]]
            assert scal.dims == map_ds.volume.dims
            vol_var=map_ds.volume.isel(time=ti)
            vert_dim=vol_var.get_axis_num('layer')

            conc_global=scal.isel(time=ti).values

            vols=vol_var.values
            vols=np.ma.array(vols,mask=(vols<=1.0))

            mass_2d=(conc_global*vols).sum(axis=vert_dim)
            water_2d=vols.sum(axis=vert_dim)
            for ri in range(len(regions)):
                cells=cell_region_idxs[ri]
                mass_region=mass_2d[cells].sum()
                water_region=water_2d[cells].sum()
                recs.append(dict(region_i=ri,region_name=regions['name'][ri],
                                 time=times_dt64[ti],
                                 tracer=scalars[si],mass=mass_region,water=water_region))
        print()
    return recs

all_recs=[]

for run in runs:
    recs=process_run(run)
    all_recs.append(recs)

# delta is missing 3 days.
recs_concat=[]
[ recs_concat.extend(el) for el in all_recs] 
df=pd.DataFrame(recs_concat)

# repack times
t=[val.values for val in df.time.values]
df['time']=np.array(t)

# HDF is 4x more efficient in space and 10x+ in time
df.to_hdf('regions-daily.h5',key='raw')

## --------------------------------------------------------------------------------

# Re-read for nice clean starting point:
df1=pd.read_hdf('regions-daily.h5',key='raw')

## 

df2=df1.drop('region_i',axis=1) # name is good enough

# change to fraction before averaging.  otherwise the results are biased 
# by tidal phase.
df2['fraction']=df2['mass']/df2['water']

## 


# Group by 2 month blocks
period_breaks=[np.datetime64('2012-10-01 00:00:00'),
               np.datetime64('2012-12-01 00:00:00'),
               np.datetime64('2013-02-01 00:00:00'),
               np.datetime64('2013-04-01 00:00:00'),
               np.datetime64('2013-06-01 00:00:00'),
               np.datetime64('2013-08-01 00:00:00'),
               np.datetime64('2013-10-01 00:00:00')]
df2['period_i']=np.searchsorted(period_breaks,df2.time)

# average over the period:
df3=df2.groupby(['period_i','tracer','region_name']).mean()
df3=df3.drop(['mass','water'],axis=1)
print(df3.head())

## 


# Make the tracer names nicer -
flows=wkb2shp.shp2geom('inflows-v07.shp')
sel= (flows['category']=='potw') | (flows['category']=='refinery' )

originals=list(flows['name'][sel])
mangles=[s.lower().replace(' ','_').replace('/','_')
         for s in originals]
for orig,mangle in zip(originals,mangles):
    print("%s => %s"%(mangle,orig))
renames=dict(zip(mangles,originals))

# doctor that up a bit:
renames['lg']='Las Gallinas'
renames['american']='American Canyon'
renames['cccsd']='Central Contra Costa'
renames['ch']='C&H, Crockett'
renames['ddsd']='Delta Diablo'
renames['delta']='Delta'
renames['fs']='Fairfield-Suisun'
renames['marin5']='Marin SD5'
renames['millbrae_burlingame']='Millbrae,Burlingame'
renames['mt_view']='Mt. View'
renames['palo_alto']='Palo Alto'
renames['petaluma']='Petaluma'
renames['phillips66']='Phillips 66'
renames['pinole']='Pinole WWTP'
renames['san_jose']='San Jose WWTP'
renames['san_mateo']='San Mateo'
renames['sausalito']='Sausalito-Marin'
renames['sea']='sea'
renames['sonoma_valley']='Sonoma Valley POTW'
renames['south_sf']='South SF'
renames['st_helena']='St Helena'
renames['stormwater']='stormwater'
renames['sunnyvale']='Sunnyvale WPCP'

##     


df4=df3.unstack()
df4.columns=df4.columns.levels[1]
df4=df4.rename(renames)

# print(df4)

## 

col_renames={'carquinez':'Carquinez',
             'central bay':'Central Bay',
             'delta':'Delta',
             'lower south bay':'Lower South Bay',
             'san pablo bay':'San Pablo Bay',
             'south bay':'South Bay',
             'suisun bay':'Suisun Bay'}
df4.rename(columns=col_renames,inplace=True)

df4.sort_index(inplace=True)

print(df4.columns)

## 


# How much difference is there between periods?
# SJSC differs between 0.010 and 0.0117, so about 17% difference.
# In the north, take Fairfield-Suisun, and the difference is from 0.0022 down to 0.0010
# So it can be significant.
print( df4.loc[(slice(None),'San Jose WWTP'),('South Bay')] )
print( df4.loc[(slice(None),'Fairfield-Suisun'),('Suisun Bay')] )

##
from xlsxwriter.utility import xl_rowcol_to_cell

writer = pd.ExcelWriter('tally_%s.xlsx'%version,engine='xlsxwriter')

def hex(r,g,b):
    return "#%02x%02x%02x"%(r,g,b)

# add some styles
workbook=writer.book
# Add a percent format with 1 decimal point
percent_fmt = workbook.add_format({'num_format': '0.000%'})
region_fmt=workbook.add_format({'bg_color':hex(198, 224, 180),'align':'center'})
region_val_fmt=workbook.add_format({'bg_color':hex(226, 239, 218),'bottom':1})
bold_fmt=workbook.add_format({'bold':True})
discharge_fmt=workbook.add_format({'bg_color':hex(189, 215, 238)})
source_fmt=workbook.add_format({'bg_color':hex(221, 235, 247),'right':1})
load_fmt=workbook.add_format({'bg_color':hex(248, 203, 173),'align':'center'})
load_reg_fmt=workbook.add_format({'bg_color':hex(252, 228, 214),'bottom':1})
load_val_fmt=workbook.add_format({'bg_color':hex(255, 204, 153),'border':1})
load_calc_fmt=workbook.add_format({'num_format':'0.0000'})
title_fmt=workbook.add_format({'font_size':14,'bold':True})
para_fmt=workbook.add_format({'text_wrap':True,'valign':'top'})

intro=workbook.add_worksheet('Intro')
intro.write_string(0,1,'Conservative Tracer Estimate Worksheet',title_fmt)
intro.merge_range(1,1,9,11,
                  ("The sheets within this workbook allow for quick estimation "
                   "of ambient concentrations based on concentrations in POTW "
                   "and stormwater loads.  Estimated POTW and stormwater flows were "
                   "input into a hydrodynamic model which then simulated water year "
                   "2013, including a two month spin-up.  For each two month period of "
                   "WY2013 ambient concentrations for each source were summed within each "
                   "region.  These concnetrations are epressed as a percentage of the "
                   "water mass.\n\n"
                   "In each sheet a loads column is provided for entering "
                   "estimated concentrations in effluent and stormwater.  These concentrations "
                   "are multiplied through with the model output to estimate concentrations "
                   "integrated over each region, both in total and broken out by "
                   "discharge."),para_fmt)

intro.insert_image('B13','model-sources-map.png',{'x_scale': 0.45, 'y_scale': 0.45})
intro.insert_image('H13','regions-v00-labels.png',{'x_scale': 0.57, 'y_scale': 0.57})

for period_i in range(1,len(period_breaks)):
    pdf4=df4.loc[period_i,:].drop(['continuity','sea'])

    Nregion=len(pdf4.columns)
    Nsource=len(pdf4.index)
    
    ptime_range=[period_breaks[period_i-1],
                 period_breaks[period_i]-np.timedelta64(1,'D')]
    period_str="%s to %s"%(utils.to_datetime(ptime_range[0]).strftime('%Y-%m'),
                           utils.to_datetime(ptime_range[1]).strftime('%Y-%m'))
    print(period_str)

    pdf4.to_excel(writer,period_str,
                  startrow=2,startcol=1,
                  index_label='Discharge')
    # Manual styling
    sheet=writer.sheets[period_str]
    sheet.set_zoom(80) # initial zoom
    sheet.set_column('B:B',25) # source names
    sheet.set_column('C:I',12,percent_fmt) # set size of column, fmt
    sheet.set_column('M:S',12) # contributions columns
    
    sheet.write_string(1,1,period_str,bold_fmt) # 0-based indices, row,col
    sheet.merge_range(1,2,1,2+Nregion-1,'Subembayment',region_fmt)
    sheet.merge_range(1,5+Nregion,1,5+Nregion+Nregion-1,
                      'Contributions to Subembayment Concentration',load_fmt)    
    sheet.write_string(2,1,'Discharge',discharge_fmt)
    sheet.write_string(1,3+Nregion,'Concentration',load_fmt)
    sheet.write_string(2,3+Nregion,'in Load',load_fmt)
    sheet.set_column('K:K',18)
        
    for col_i,col_val in enumerate(pdf4.columns.values):
        sheet.write_string(2,2+col_i,col_val,region_val_fmt)
        sheet.write_string(2,Nregion+5+col_i,col_val,load_reg_fmt)
        sheet.write_string(5+Nsource,Nregion+5+col_i,col_val,load_reg_fmt)
    for row_i,row_val in enumerate(pdf4.index.values):
        sheet.write_string(3+row_i,1,row_val,source_fmt)
        sheet.write_number(3+row_i,Nregion+3,15.0,load_val_fmt)
    sheet.write(4+Nsource,5+Nregion,"Totals",load_fmt)

    for col_i,col_val in enumerate(pdf4.columns.values):
        for row_i,row_val in enumerate(pdf4.index.values):
            formula="=%s * %s"%(xl_rowcol_to_cell(3+row_i,col_i+2),
                                xl_rowcol_to_cell(3+row_i,3+Nregion,False,True))
            sheet.write_formula(3+row_i,5+Nregion+col_i,formula,
                                load_calc_fmt)
        col_start=xl_rowcol_to_cell(3,          5+Nregion+col_i)
        col_end  =xl_rowcol_to_cell(3+Nsource-1,5+Nregion+col_i)
        sheet.write_formula(6+Nsource,5+Nregion+col_i,"=SUM(%s:%s)"%(col_start,col_end),
                            load_calc_fmt)
    
df_specs=pd.Series(dict(start_time=df1.time.min(),
                        end_time=df1.time.max(),
                        region_shp=region_shp,
                        version=version,
                        waq_dir=base_dir))
pd.DataFrame(df_specs).to_excel(writer,'Metadata')

writer.save()

## --------------------------------------------------------------------------------

pd.DataFrame(df_specs)

# Sanity check:
lsb_sj=df4.loc[(slice(None),'San Jose WWTP'),'Lower South Bay']
sj_no3=600
lsb_no3=lsb_sj*sj_no3
print("Expected NO3 (uM) from San Jose in Lower South Bay",lsb_no3)
# that's the right ball park.  Note that total pct effluent is 
# 8.5%, so reality would be a little higher.
print("Observed NO3 is 40--110uM at Station 36 (Calaveras Point)")
print("                40--50uM  at Station 34 (Newark Slough)")
print("                20--40uM  at Station 32 (Ravenswood Pt, N of Dumbarton Br.)")

## 

if 0:
    # graphic representation
    g.edge_to_cells()
    grid_boundary=g.boundary_polygon()

    zoom=(533123.65957726236, 610471.18834887329, 4137693.3540862156, 4232681.5473145116)
    fig=plt.figure(figsize=(5.7,7))
    ax=fig.add_axes([0,0,1,1])
    ax.xaxis.set_visible(0) ; ax.yaxis.set_visible(0)
    grid_poly=plot_wkb.plot_polygon(grid_boundary,fc='0.7')
    region_polys=[]

    for poly in regions['geom']:
        gpoly=poly.intersection(grid_boundary)
        region_polys.append(  plot_wkb.plot_polygon(gpoly) )

    ax.axis('equal')
    ax.axis(zoom)

    cmap=gmtColormap.load_gradient('oc-sst.cpt')

    for ri in range(len(region_polys)):
        region_polys[ri].set_facecolor(cmap( ri/float(len(region_polys)-0.0)))
    fig.savefig('regions-%s-nolabels.png'%version,dpi=200)


    # In[184]:


    trans={'south bay':'South Bay',
           'lower south bay':'Lower South Bay',
          'central bay':'Central Bay',
          'san pablo bay':'San Pablo Bay',
          'carquinez':'Carquinez',
          'suisun bay':'Suisun Bay',
          'delta':'Delta'}
    offsets={'south bay':[8e3,2e3],
             'lower south bay':[1e3,3e3],
             'central bay':[8e3,0],
             'san pablo bay':[-6e3,8e3],
             'carquinez':[-3e3,-5e3],
             'suisun bay':[-8e3,-14e3],
             'delta':[-70e3,-18e3]}
    ax.texts=[]
    for ri in range(len(regions)):
        name=regions[ri]['name']
        xy=np.array(regions[ri]['geom'].centroid)
        xy=xy+offsets.get(name,[0,0])
        print name,xy
        txt=trans[name]
        ax.text(xy[0],xy[1],txt)
    plt.draw()
    fig.savefig('regions-%s-labels.png'%version,dpi=200)

