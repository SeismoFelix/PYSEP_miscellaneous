
import glob
from obspy import read
from obspy.core import UTCDateTime
import os

def rename_files(path):
    #2014-08-25T161903_ICELAND.Z7.DYSA..HHE.sac -> 20140825161903.Z7.DYSA..HH.e
    print(path)

    list_sac_files = glob.glob('{}/*.sac'.format(path))
    ot = list_sac_files[0].split('/')[-1].split('_')[0]
    year = ot.split('-')[0]
    month = ot.split('-')[1]
    day = ot.split('-')[2].split('T')[0]
    hour =  ot.split('-')[2].split('T')[1][0:2]
    min =  ot.split('-')[2].split('T')[1][2:4]
    sec =  ot.split('-')[2].split('T')[1][4:6]
    utc_ot = UTCDateTime(int(year),int(month),int(day),int(hour),int(min),int(sec))
    ev_id = '{}{}{}{}'.format(year,month,day,ot.split('-')[2].split('T')[1])
    mkdir_ev = 'mkdir {}'.format(ev_id)
    print(mkdir_ev)
    os.system(mkdir_ev)

    
    for file in list_sac_files:
        stream = read(file)
        new_name = '{}/{}.{}.{}.{}.{}.{}'.format(ev_id,ev_id,stream[0].stats.network,stream[0].stats.station,stream[0].stats.location,stream[0].stats.channel[0:2],stream[0].stats.channel[-1].lower())
        cp_file = 'cp {} {}'.format(file,new_name)
        print(cp_file)
        os.system(cp_file)
    return(ev_id,utc_ot)

def write_weight(ev_id):
    list_sac_files = glob.glob('{}/{}*.z'.format(ev_id,ev_id))
    open_weight = open('{}/weights.dat'.format(ev_id),'w')

    for i in range(len(list_sac_files)-1):
        file = list_sac_files[i]
        stream = read(file)
        st_name = file.split('/')[-1][0:-1]
        line = '{} {} 1 1 1 1 1 0.00   0.00      0      0      0\n'.format(st_name,stream[0].stats.sac.dist)
        open_weight.write(line)
    
    file = list_sac_files[-1]
    stream = read(file)
    st_name = file.split('/')[-1][0:-1]
    line = '{} {} 1 1 1 1 1 0.00   0.00      0      0      0'.format(st_name,stream[0].stats.sac.dist)
    open_weight.write(line)

def mtuq_syntax(ev_id,mw,utc_ot,nppa_dc,nppa_fmt,fb):
    list_sac_files = glob.glob('{}/{}*.z'.format(ev_id,ev_id))
    stream =  read(list_sac_files[0])
    event = list_sac_files[0].split('/')[-1].split('.')[0]
    evla = stream[0].stats.sac.evla
    evlo = stream[0].stats.sac.evlo
    evdp = float(stream[0].stats.sac.evdp)*1000
    time = str(utc_ot)
    freq = '{}-{}'.format(fb[0],fb[1])
    run_mtuq_DC = 'mpirun -np 8 python GridSearch.DoubleCouple_SW_BW_options.py -event {} -evla {} -evlo {} -evdp {} -mw {} -time {} -np {} -fb {}\n'.format(event,evla,evlo,evdp,mw,time,nppa_dc,freq)
    run_mtuq_FMT = 'mpirun -np 8 python GridSearch.FMT_SW_BW_options.py -event {} -evla {} -evlo {} -evdp {} -mw {} -time {} -np {} -fb {}'.format(event,evla,evlo,evdp,mw,time,nppa_fmt,freq)
    open_mtuq = open('{}/run_mtuq.txt'.format(ev_id),'w')
    open_mtuq.write(run_mtuq_DC)
    open_mtuq.write(run_mtuq_FMT)


path = '../2017-09-03T033001_NORTH_KOREA/SAC'
ev_id,utc_ot = rename_files(path)
write_weight(ev_id)
#mw = 4.5
mw = 5.2
nppa_dc = 50
nppa_fmt = 11
fb = [15,33]
mtuq_syntax(ev_id,mw,utc_ot,nppa_dc,nppa_fmt,fb)

    

