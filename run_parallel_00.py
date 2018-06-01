"""
Following 
/opt/data/dwaq/cascade/sfbay_potw_py/run_long_multi.py

run one or a few tracers per each run
"""

import hashlib
import os
import glob
import logging
logging.basicConfig(level=logging.INFO)

import multiprocessing

import sfbay_cons_tracer_00
import pandas as pd

from stompy.model.delft import waq_scenario
import stompy.model.delft.io as dio

PC=waq_scenario.ParameterConstant

##


def load_hydro():
    # Needed to create a full scenario to in turn get the full list of tracers
    return sfbay_cons_tracer_00.hydro


def find_missing_tracers():
    logging.info("Finding missing tracers")
    hydro=load_hydro()
    scen=sfbay_cons_tracer_00.Scen(hydro=hydro)

    all_tracers=list(scen.substances.keys())

    # Which tracers have not yet been run?
    hists=glob.glob('%s-*/%s*-bal.his'%(scen.base_path,scen.name))

    past_tracers=[]

    for hist in hists:
        logging.info("Checking %s for completed tracers"%hist)
        df=dio.mon_his_file_dataframe(hist)
        for s in df.columns.levels[1]:
            try:
                s=s.decode()
            except AttributeError:
                pass
            past_tracers.append(s)

    missing_tracers=[t
                     for t in all_tracers
                     if t not in past_tracers]
    logging.info("Remaining tracers: %s"%(", ".join(missing_tracers)))
    
    return missing_tracers
## 

def worker(tracers):
    """
    Takes a list of tracers, creates and runs dwaq for those sources.
    """
    base_path=None
    try:
        # Unsafe to share hydro across multiple scenarios
        # this was an issue when using threads, no issue with multiprocessing
        hydro=load_hydro()

        if len(tracers)==1:
            tag=tracers[0]
        else:
            tag=hashlib.md5(str(tracers).encode()).hexdigest()
            
        base_path="%s-%s"%(sfbay_cons_tracer_00.Scen.base_path,
                           tag)

        if os.path.exists(base_path):
            return "Run %s exists at least in part"%base_path
        logging.info("Initiating run %s for %s"%(base_path,", ".join(tracers)))
        
        scen=sfbay_cons_tracer_00.Scen(sub_subset=tracers,
                                       hydro=hydro,
                                       base_path=base_path)
        scen.parameters['nothreads']=2
        scen.start_time=scen.time0+scen.scu*hydro.t_secs[0]
        scen.stop_time =scen.time0+scen.scu*hydro.t_secs[-2]
        # real run, but just daily map output 
        scen.map_time_step=240000 # map output daily
        
        scen.cmd_default()
        return "Run complete in %s"%base_path
    except Exception as exc:
        logging.error("Run in %s failed with exception:"%base_path)
        logging.exception(exc)
        return "Run %s failed with exception"%base_path


# with 4 here, it was down around 35-50x realtime, even early in the run.
per_run=1

from multiprocessing import Pool

if __name__=='__main__':
    missing_tracers=find_missing_tracers()
    N=len(missing_tracers)
    tracer_groups=[missing_tracers[i:min(i+per_run,N)]
                   for i in range(0,N,per_run)]
    logging.info("Runs:")
    for grp in tracer_groups:
        logging.info("  %s"%(", ".join(grp)))

    with Pool(10) as p:
        for ret in p.imap_unordered(worker,tracer_groups):
            print(ret)


