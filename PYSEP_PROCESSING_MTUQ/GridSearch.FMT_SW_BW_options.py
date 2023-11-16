#!/usr/bin/env python

import os
from zlib import MAX_WBITS
import numpy as np
import argparse
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

def parse_args():

    parser = argparse.ArgumentParser(
    description="Input event info run MTUQ",
    formatter_class=argparse.RawTextHelpFormatter,
                                     )

    #parser.add_argument("-mdir",type=str,help="Main dir: -mdir /Users/felix/Documents/INVESTIGACION/2_FEB_JUL_2022/IRIS_WORKSHOP/MTUQ_INVERSIONS/FK_VS_1D_CUBIC_MESH/gs_mtuq")
    parser.add_argument("-event",type=str,help="Event ID (event directory must be in main dir): -event 20140823183304000 ")
    parser.add_argument("-evla",type=str,help="Latitude: -evla 64.68 ")
    parser.add_argument("-evlo",type=str,help="longitude: -evla -98.2 ")
    parser.add_argument("-evdp",type=str,help="Depth in m: -evdp 1000.0")
    parser.add_argument("-mw",type=float,help="Magnitude: -mw 4.9")
    parser.add_argument("-time", type=str,help="Origin time: -time 2014-08-25T16:19:03.00000Z")
    parser.add_argument("-np", type=int,help="Number of points per axis: -np 10")
    parser.add_argument("-fb",type=str,help="Frequency band for filtering data in seconds (body_waves/surface_waves): -fb 3-15/15-33")
    parser.add_argument("-wl",type=str,help="Window length in seconds (body_waves/surface_waves): -wl 30/200")
    parser.add_argument("-ts",type=str,help="Time shift limits in seconds (body_waves/surface_waves): -ts 5/15 (means a +/-5s and +/-15s time shift)")

    return parser.parse_args()


if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters for the catalog depth and magnitude of the 2020-04-04 SoCal event
    # Uses local database of Green's functions calculated via the FK method
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor_cat_SOCAL.py
    #

    param = parse_args()
    print(param.event)
    mdir = os.getcwd()

    path_data=    fullpath('{}/{}/*.[zrt]'.format(mdir,param.event))
    path_weights= fullpath('{}/{}/weights.dat'.format(mdir,param.event))
    event_id=     '{}'.format(param.event)
    #model=        'ir'
    model=        'ak135'
    #db = open_db('{}/greens/ir'.format(mdir),format='FK')

    #
    # Body and surface wave measurements will be made separately
    #

    freqs_bw = param.fb.split('/')[0].split('-')
    freqs_sw = param.fb.split('/')[1].split('-')
    wl_bw = float(param.wl.split('/')[0])
    wl_sw = float(param.wl.split('/')[1])

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 1/float(freqs_bw[1]),
        freq_max= 1/float(freqs_bw[0]),
        #pick_type='FK_metadata',
        pick_type='taup',
        taup_model=model,
        #FK_database='{}/greens/ir'.format(mdir),
        window_type='body_wave',
        window_length=wl_bw,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=1/float(freqs_sw[1]),
        freq_max=1/float(freqs_sw[0]),
        #pick_type='FK_metadata',
        pick_type='taup',
        taup_model=model,
        #FK_database='{}/greens/ir'.format(mdir),
        window_type='surface_wave',
        window_length=wl_sw,
        capuaf_file=path_weights,
        )

    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    ts_bw = int(param.ts.split('/')[0])
    ts_sw = int(param.ts.split('/')[1])

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-1*ts_bw,
        time_shift_max=ts_bw,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-1*ts_sw,
        time_shift_max=ts_sw,
        time_shift_groups=['ZR','T'],
        )

    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
    station_id_list = parse_station_codes(path_weights)

    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=param.np,
        #magnitudes= magnitudes.tolist()
        magnitudes=[float(param.mw)]
        )

    wavelet = Trapezoid(
        magnitude=float(param.mw))

    #
    # Origin time and location will be fixed. For an example in which they
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '{}'.format(param.time),
        'latitude': param.evla,
        'longitude': param.evlo,
        'depth_in_m': param.evdp,
        })

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity'])

        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        #greens = db.get_greens_tensors(stations,origin)
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')
    
    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, grid)

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)

    if comm.rank==0:

        results = results_bw + results_sw

        # array index corresponding to minimum misfit
        #idx = results.idxmin('source') #Old version
        idx = results.source_idxmin() #New version

        best_source = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = grid.get(idx).as_dict()

        #
        # Generate figures and save results
        #

        print('Generating figures...\n')

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
            misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

        plot_beachball(event_id+'FMT_beachball.png',
            best_source, stations, origin)

        plot_misfit_lune(event_id+'FMT_misfit.png', results)
        plot_misfit_lune(event_id+'FMT_misfit_mt.png', results, show_mt=True)
        plot_misfit_lune(event_id+'FMT_misfit_tradeoff.png', results, show_tradeoffs=True)

        print('Saving results...\n')

        merged_dict = merge_dicts(lune_dict, mt_dict, origin,
            {'M0': best_source.moment(), 'Mw': best_source.magnitude()})

        # save best-fitting source
        save_json(event_id+'FMT_solution.json', merged_dict)

        # save misfit surface
        results.save(event_id+'FMT_misfit.nc')


        print('\nFinished\n')