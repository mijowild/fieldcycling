# fieldcycling

## parameters often used in stelar data files (.sdf)
* DW, (dwell time) is the time step used in the digitilization in units of microseconds (1/sampling rate) (e.g. of FID)
* NS, (number of scans) number of repetition of experiment
* NBLK, (number of blocks) how many shots/blocks (e.g. FIDs) were recorded in one experiment
* BS, (block size) number of points in one shot/block (e.g. FID)
* BINI, time step of the first block in units of seconds
* BEND, time step of the last block in units of seconds
    * *watch out*: BINI and BEND can contain a string instead of a float number. 
        It has to be evaluated (e.g. BEND = '5 * T
* BGRD, is the grid of the time steps (lin, log or list)
* TIME, start time of experiment
* BRLX, relaxation field of the experiment (variable for the dispersion)

