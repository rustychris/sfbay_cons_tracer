# IGNORE
# snippets used to test the boundary file parsing and interpretation

bnds=hydro.read_bnd()

g=hydro.grid()

##

fig=plt.figure(1)
fig.clf()
ax=fig.gca()

g.plot_edges(lw=0.5,color='0.5')

segs=[bnd[1]['x'].reshape((-1,2)) for bnd in bnds]
from matplotlib.collections import LineCollection
ax.add_collection(LineCollection(segs,color='r',lw=1.))

for bnd in bnds:
    ax.text(bnd[1]['x'][0,0,0],
            bnd[1]['x'][0,0,1],
            bnd[0],color='k')

##

# And do the exchanges make sense?
# in the bnd file, they are incrementally negative, -1 to -248.
# poi has down to -2480, though.
poi=hydro.pointers
hydro.infer_2d_elements()

elts=[]
elt_k=[]

for bnd in bnds:
    for bc_exch in bnd[1]['exch']:
        exch_i=np.nonzero(poi[:,0]==bc_exch)[0][0]
        seg_from,seg1_to=poi[exch_i,:2]

        seg0_to=seg1_to-1
        elts.append( hydro.seg_to_2d_element[seg0_to] )
        elt_k.append( hydro.seg_k[seg0_to] )
        
coll=g.plot_cells(mask=elts,ax=ax,values=elt_k,cmap='jet')
coll.set_clim([0,10])

##
tseries=[]

if 0:
    # Check on vertical distribution of flow at a river vs. outfall
    rio_vista=[bnd[0] for bnd in bnds].index('RioVista_flow')
    bnd=bnds[rio_vista] # river input
else:
    ebda=[bnd[0] for bnd in bnds].index('ebda')
    bnd=bnds[ebda] # outfall

for t_idx in [1000]: # range(960,1060):
    flow=hydro.flows(hydro.t_secs[t_idx])

    hydro.infer_2d_links() # complains about ambiguous connection

    total_flow=0.0
    for bc_exch in bnd[1]['exch']:
        print("Exch %i"%bc_exch)
        exch_i=np.nonzero(poi[:,0]==bc_exch)[0][0]
        assert exch_i<hydro.n_exch_x+hydro.n_exch_y
        seg_from,seg1_to=poi[exch_i,:2]

        seg0_to=seg1_to-1
        # not sure we can rely on infer_2d_links to match up with the identifiers
        # in bnd, but should be okay for extruding a surface exchange to get the rest.
        link=hydro.exch_to_2d_link[exch_i]
        exchanges_3d=np.nonzero( hydro.exch_to_2d_link==link )[0]

        for exch_3d in exchanges_3d:
            print("   %i:  flow=%.4f"%(exch_3d,flow[exch_3d]))
            total_flow += flow[exch_3d]
    dt=utils.to_datetime(hydro.t_dn[t_idx])
    tseries.append( [hydro.t_dn[t_idx],
                     total_flow] )
    print("Total flow(t=%s): %.4f"%(dt.strftime('%Y-%m-%d %H:%M'),total_flow))
##

fig=plt.figure(2) ; fig.clf()
ax=fig.gca()
tseries=np.array(tseries)
ax.plot_date(tseries[:,0],tseries[:,1]*35.314667,'b-o')

