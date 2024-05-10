import numpy as np
import argparse


def getMGRS():
    LON_STEP = 6
    LAT_STEP = 8   
    lons = np.arange(-180,180,LON_STEP)
    lats = np.arange(-80,80,LAT_STEP)
    lon_labels = np.arange(1,61)
    lat_labels = ['C','D','E','F','G','H','J','K','L','M',
                  'N','P','Q','R','S','T','U','V','W','X']        
    grid = {}
    for i in range(len(lats)):
        for j in range(len(lons)):
            grid[str(lon_labels[j]).zfill(2)+lat_labels[i]] = (lons[j],lats[i],lons[j]+LON_STEP,lats[i]+LAT_STEP)
    
    for i in lon_labels:
        idx = str(i).zfill(2)+'X'
        grid[idx] = (lons[i-1],72,lons[i-1]+LON_STEP,84) 
    grid['31V'] = (0,56,3,64)
    grid['32V'] = (3,56,12,64)
    grid['31X'] = (0,72,9,84)
    grid['33X'] = (9,72,21,84)
    grid['35X'] = (21,72,33,84)
    grid['37X'] = (33,72,42,84)   
    del grid['32X']
    del grid['34X']
    del grid['36X']
    return grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k','--key',default=None,type=str)
    args = parser.parse_args()
    if args.key:
        grid = getMGRS()
        print(grid[args.key])