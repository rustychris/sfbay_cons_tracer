"""
Copied from
/home/rusty/models/delft/nbwaq/spinupdate/spinupdate_wy2013_D02cons_tracer.py

Moving on to 4/28/16, np=4, D02 hydrodynamics

this one does the full list of passive tracers

"""

import os
import shutil
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import numpy as np

from stompy import utils
from stompy.model.delft import waq_scenario

import pandas as pd

import ugrid
from stompy.spatial import (wkb2shp,proj_utils)
from stompy.grid import unstructured_grid

## 

# The 2017-10-16 runs used this older hydrodynamic data.
# hydro=waq_scenario.HydroFiles("../../delft/sfb_dfm_v2/runs/wy2013a/DFM_DELWAQ_wy2013a/wy2013a.hyd")
# This is the hydro described in the Intermim Model Validation Report,  with the adjusted
# fluxes for issues with that version of DFM.
hydro=waq_scenario.HydroFiles("../../delft/sfb_dfm_v2/runs/wy2013c/DFM_DELWAQ_wy2013c_adj/wy2013c.hyd")
hydro.enable_write_symlink=True

## 

PC=waq_scenario.ParameterConstant
Sub=waq_scenario.Substance
IC=waq_scenario.Initial

# Water quality setup
class Scen(waq_scenario.Scenario):
    name="sfb_dfm_v2"
    desc=('sfb_dfm_v2',
          'wy2013c',
          'conserv_tracer')
    # removed BALANCES-SOBEK-STYLE
    # IMPORTANT to have the NODISP-AT-BOUND in there.
    integration_option="""15 ;
    LOWER-ORDER-AT-BOUND NODISP-AT-BOUND
    BALANCES-OLD-STYLE BALANCES-GPP-STYLE 
    BAL_NOLUMPPROCESSES BAL_NOLUMPLOADS BAL_NOLUMPTRANSPORT
    BAL_NOSUPPRESSSPACE BAL_NOSUPPRESSTIME
    """
    base_path='runs/wy2013c-20180404'

    #maybe this will be more stable with shorter time step?
    # with 30 minutes, failed with non-convergence at 0.09% or so.
    # with 15 minutes, failed with non-convergence at 7.28%
    time_step=1000 # dwaq HHMMSS integer format

    # a bit less annoying than nefis
    map_formats=['binary']
    
    # stormwater is grouped into a single tracer
    storm_sources=['SCLARAVW2_flow',
                   'SCLARAVW1_flow',
                   'SCLARAVW4_flow',
                   'SCLARAVW3_flow',
                   'UALAMEDA_flow',
                   'EBAYS_flow',
                   'COYOTE_flow',
                   'PENINSULb1_flow',
                   'EBAYCc3_flow',
                   'USANLORZ_flow',
                   'PENINSULb3_flow',
                   'PENINSULb4_flow',
                   'EBAYCc2_flow',
                   'PENINSULb6_flow',
                   'PENINSULb2_flow',
                   'PENINSULb7_flow',
                   'PENINSULb5_flow',
                   'SCLARAVCc_flow',
                   'SCLARAVW5_flow',
                   'MARINS1_flow',
                   'EBAYCc6_flow',
                   'EBAYCc1_flow',
                   'EBAYCc5_flow',
                   'EBAYCc4_flow',
                   'MARINN_flow',
                   'NAPA_flow',
                   'CCOSTAW2_flow',
                   'CCOSTAW3_flow',
                   'MARINS3_flow',
                   'MARINS2_flow',
                   'PETALUMA_flow',
                   'SONOMA_flow',
                   'CCOSTAW1_flow',
                   'SOLANOWc_flow',
                   'CCOSTAC2_flow',
                   'EBAYN1_flow',
                   'EBAYN4_flow',
                   'EBAYN2_flow',
                   'EBAYN3_flow',
                   'SOLANOWa_flow',
                   'SOLANOWb_flow',
                   'CCOSTAC3_flow',
                   'CCOSTAC1_flow',
                   'CCOSTAC4_flow']
    
    delta_sources=['Jersey_flow',
                   'RioVista_flow']

    sea_sources=[ 'Sea_ssh' ]

    # run only a subset of substances
    sub_subset=None
    
    def init_substances(self):
        subs=super(Scen,self).init_substances()

        # with the DFM run, how do we get the labeling of boundary condition
        # and discharge flows?
        # previous code just used horizontal boundary flows.
        # what do the discharges look like the dwaq data?

        # there is a .bnd file, with 88 labeled entries, things like
        # "SCLARAVW2_flow".
        # each of those has what appears to be the numbers (negative)
        # for the boundary exchanges (sometime several boundary exchanges)
        # and some xy coordinates
        # maybe that's all we need.  

        link_groups=self.hydro.group_boundary_links()
        
        groups={} # just used for sanity checks to make sure that BCs that we're
        # trying to set exist.
        for link_group in link_groups:
            if link_group['id']>=0 and link_group['name'] not in groups:
                groups[ link_group['name'] ] = link_group
        
        # all src_tags default to a concentration BC of 1.0, which is exactly
        # what we want here.  no need to specify additional data.

        def check(name):
            return (self.sub_subset is None) or (name in self.sub_subset)
        
        for k in groups.keys():
            if k in self.delta_sources + self.storm_sources + self.sea_sources:
                # these are lumped below
                continue

            # if k!='ebda':#DBG
            #     continue
            
            name=k
            if name=='millbrae':
                # millbrae and burlingame got combined along the way, due to
                # entering in the same cell.
                name='millbrae_burlingame'

            if not check(name):
                continue
                
            print("Adding tracer for %s"%name)
            subs[name]=Sub(initial=IC(default=0.0))
            # any additional work required here?
            # hopefully waq_scenario is going to use these same names to label
            # boundaries
            self.src_tags.append(dict(tracer=name,items=[k]))

        if check('delta'):
            subs['delta']=Sub(initial=IC(default=0.0))
            self.src_tags.append( dict(tracer='delta',
                                       items=self.delta_sources) )

        if check('stormwater'):
            subs['stormwater']=Sub(initial=IC(default=0.0))
            self.src_tags.append( dict(tracer='stormwater',
                                       items=self.storm_sources) )

        if check('sea'):
            subs['sea']=Sub(initial=IC(default=0.0))
            self.src_tags.append( dict(tracer='sea',
                                       items=self.sea_sources) )

        if check('continuity'):
            subs['continuity']=Sub(initial=IC(default=1.0))
            self.src_tags.append( dict(tracer='continuity',
                                       items=list(groups.keys())))

        return subs

    def init_parameters(self):
        # choose which processes are enabled.  Includes some
        # parameters which are not currently used.
        params=super(Scen,self).init_parameters()
        
        params['NOTHREADS']=PC(24) # one processor.. or maybe two, or four

        # maybe defaulted to 1e-7?
        # 1e-6 failed to 0.09%, same as before
        params['Tolerance']=1e-5

        params['ACTIVE_DYNDEPTH']=1
        params['ACTIVE_TOTDEPTH']=1

        return params
        
    def cmd_default(self):
        self.cmd_write_hydro()
        self.cmd_write_inp()
        self.cmd_delwaq1()
        self.cmd_delwaq2()

    def __init__(self,*a,**k):
        super(Scen,self).__init__(*a,**k)

        self.map_output+=('TotalDepth',
                          'volume',
                          'depth')
    
## 
if __name__=='__main__':
    scen=Scen(hydro=hydro,
              sub_subset=['stormwater'])

    scen.start_time=scen.time0+scen.scu*hydro.t_secs[0]
    scen.stop_time =scen.time0+scen.scu*hydro.t_secs[-2]
    # real run, but just daily map output 
    scen.map_time_step=240000 # map output daily
    # debugging - hourly output
    # scen.map_time_step=10000 # map output hourly

    if 1:
        scen.cmd_default()
        # scen.cmd_write_nc()
    else:
        scen.main()

