"""
Module for simulating the ground track of a satellite in nadir pointing mode.

This module provides functions to calculate the ECI (Earth-Centered Inertial) orbit, convert it to ECEF (Earth-Centered Earth-Fixed) coordinates, and plot the ground track of the satellite.

Functions:
- get_eci_orbit: Get the ECI orbit coordinates.
- get_orbit: Get the orbit information in ECEF coordinates.
- get_ground_track: Calculate the latitude, longitude, and altitude of a satellite given its ECEF coordinates.
- plot_ground_track: Plot the ground track of the satellite.
"""

from orbit_gen import get_random_orbit, get_Rz, get_nadir_attitude, get_nadir_attitude_vectors
import numpy as np
from astropy.coordinates import EarthLocation
from matplotlib import pyplot as plt
from SatCam import SatCam, SatellitePose
from getMGRS import getMGRS
import cv2
from ultralytics import YOLO
import csv
from tqdm.contrib.concurrent import process_map
import os

def get_eci_orbit(tf=None):
    """
    Get the ECI (Earth-Centered Inertial) orbit.

    Returns:
        orbit_eci (list): List of ECI coordinates.
        tsamp (float): Time sampling interval.
        orbit_eci_q (list): List of ECI coordinates with quaternions.
    """
    if tf is not None:
        orbit_eci, tsamp, orbit_eci_q = get_random_orbit(tf=tf)
    else:
        orbit_eci, tsamp, orbit_eci_q = get_random_orbit()
    return orbit_eci, tsamp, orbit_eci_q

def get_orbit(tf=None):
    """
    Get the orbit information in ECEF coordinates.

    Returns:
        orbit_ecef (numpy.ndarray): The orbit position and attitude in ECEF coordinates with direction vectors.
        orbit_eci (numpy.ndarray): The orbit position in ECI coordinates with direction vectors.
        tsamp (numpy.ndarray): The time samples for Earth rotation.
        orbit_eci_q (numpy.ndarray): The orbit position and attitude in ECI coordinates with quaternions.
    """
    if tf is not None:
        orbit_eci, tsamp, orbit_eci_q = get_eci_orbit(tf)
    else:   
        orbit_eci, tsamp, orbit_eci_q = get_eci_orbit()
    # get time samples for Earth rotation
    Rzs = get_Rz(tsamp)
    # Get ECI position vector
    r_eci = orbit_eci[:, :3, np.newaxis]
    # Convert ECI position vector to ECEF
    r_ecef = np.matmul(Rzs, r_eci)

    # Stack position and attitude and convert to meters
    orbit_ecef = np.concatenate([r_ecef[:,:,0]*1000, orbit_eci[:,3:]], axis=1)
    return orbit_ecef, orbit_eci, tsamp, orbit_eci_q

def get_ground_track(orbit_ecef):
    """
    Calculates the latitude, longitude, and altitude of a satellite given its ECEF coordinates.
    
    Parameters:
        orbit_ecef (numpy.ndarray): Array of ECEF coordinates of the satellite.
    
    Returns:
        tuple: A tuple containing the latitude, longitude, and altitude of the satellite.
    """
    # Get the latitude, longitude, and altitude of the satellite
    loc = EarthLocation.from_geocentric(orbit_ecef[:,0], orbit_ecef[:,1], orbit_ecef[:,2], unit='m')
    lat = loc.lat.deg
    lon = loc.lon.deg
    alt = loc.height.value
    return lat, lon, alt

def plot_ground_track(lat, lon):
    """
    Plot the ground track of the satellite.
    
    Parameters:
        lat (numpy.ndarray): Array of latitude values.
        lon (numpy.ndarray): Array of longitude values.
    """
    fig, ax = plt.subplots()
    ax.scatter(lon, lat)
    plt.show()

def get_detections(img, region, window_transform, cam):
    model_path = 'models/' + region + '.pt'
    model = YOLO(model_path)
    conf_threshold = float(np.load('best_confs/' + region + '_best_conf.npy'))
    best_classes = np.load('best_classes/' + region + '_best_classes.npy')
    if os.path.exists('bad_classes/' + region + '_bad_classes.npy'):
        bad_classes = np.load('bad_classes/' + region + '_bad_classes.npy')
    else:
        bad_classes = []
    results = model.predict(img, conf=conf_threshold, classes=best_classes, imgsz=(2592, 4608), verbose=False)
    result = results[0]
    im_region_dets = []
    if len(result.boxes) > 0:
        for detection in result.boxes:
            cls = int(detection.cls.item())
            if cls not in bad_classes:
                xc, yc, w, h, = detection.xywh[0]
                xc, yc, w, h = xc.item(), yc.item(), w.item(), h.item()
                det_lon, det_lat = cam.convert_xc_yc_to_lon_lat(xc, yc, window_transform)
                cls_lon, cls_lat = get_lon_lat_from_cls(cls, region)
                det_pixel_location = cam.lonlat_to_pixel_coords(det_lon, det_lat)
                cls_pixel_location = cam.lonlat_to_pixel_coords(cls_lon, cls_lat)
                im_region_dets.append([cls, det_pixel_location[0], det_pixel_location[1], detection.conf.item()])
        return im_region_dets
    else:
        return None
    
def get_lon_lat_from_cls(cls, region):
    landmark_csv_path = 'landmark_csvs/' + region + '_top_salient.csv'
    with open(landmark_csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        dictlist = list(csv_reader)
    lon_lat_dict = dictlist[cls]
    lon = lon_lat_dict['Centroid Longitude']
    lat = lon_lat_dict['Centroid Latitude']
    return lon, lat

def eval_px_error(cls, lon, lat, xc, yc, cam):
    lon_px, lat_px = cam.lonlat_to_pixel_coords(lon, lat)
    x_err = np.abs(xc - lon_px)
    y_err = np.abs(yc - lat_px)
    return x_err, y_err, lon_px, lat_px

    
def run_sim(orbit_num):
    grid = getMGRS()
    orbit_ecef, orbit_eci, tsamp, orbit_eci_q = get_orbit(10800)
    nadir_attitude = get_nadir_attitude(orbit_eci)
    dir_vec, up_vec, right_vec = get_nadir_attitude_vectors(orbit_ecef)
    dir_vec_eci, up_vec_eci, right_vec_eci = get_nadir_attitude_vectors(orbit_eci)
    traj_positions = orbit_ecef[:,:3]
    nadir_orbit_ecef = np.concatenate([traj_positions, dir_vec, up_vec, right_vec],axis=1)
    nadir_orbit_eci = np.concatenate([traj_positions, dir_vec_eci, up_vec_eci, right_vec_eci], axis=1)
    orbit_path = 'orbits/pose'
    detection_path = 'orbits/detections'
    np.save(orbit_path + '/' + str(orbit_num).zfill(5) + '_orbit_eci_zyxvecs.npy', nadir_orbit_eci)
    np.save(orbit_path + '/' + str(orbit_num).zfill(5) + '_orbit_ecef_zyxvecs.npy', nadir_orbit_ecef)
    # json_data = json.dumps(nadir_orbit_ecef.tolist())
    # print(json_data)
    lats, lons, alt = get_ground_track(nadir_orbit_ecef)
    #in_timesteps = np.zeros(len(lats), dtype=bool)

    regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', 
                '32S', '32T', '33S', '33T', '52S', '53S', '54S', '54T']

    
    savevid = False
    showim = False
    check_err = True
    satcam = None
    im_w = 4608
    im_h = 2592
    vid_w = 576
    vid_h = 324
    vidname = str(orbit_num).zfill(5) + 'demo.avi'

    if savevid:
        video = None

    all_detections = []
    errs = []
    for i, pose in enumerate(nadir_orbit_ecef):    
        # if i == 0:
        #     if lat[i+1] < lat[i]:
        #         continue
        # else:
        #     if lat[i] < lat[i-1]:
        #         continue
        if satcam is None:
            satpose = SatellitePose(pose)
            satcam = SatCam(satpose, 66, im_w, im_h, regions=regions)
        else:
            satpose = SatellitePose(pose)
            satcam.update_pose(satpose) # implement update pose
        #cur_regions = satcam.find_current_regions()
        # 
        # for region in cur_regions:
        #     if region in regions:
        #         in_timesteps[i] = True
        #         # xyz_arr = satcam.get_xyz_array(32)
        #         # lonlat_arr = satcam.get_lonlat_array(xyz_arr, 32)
        #         img = satcam.get_image(8)
        # if ctr_region in regions:
        #     in_timesteps[i] = True
        #     img= satcam.get_image(1)
        if satcam.check_for_all_landmarks():
            # in_timesteps[i] = True
            #ctr_region = satcam.get_region(lons[i], lats[i])
            #satim, window_transform = satcam.get_image(ctr_region)      
            #if satim is not None:
            corner_lonlats = satcam.corner_lonlats
            tl_lon, tl_lat = corner_lonlats['tl']
            br_lon, br_lat = corner_lonlats['br']
            tr_lon, tr_lat = corner_lonlats['tr']
            bl_lon, bl_lat = corner_lonlats['bl']
            tl_reg = satcam.get_region(tl_lon, tl_lat)
            br_reg = satcam.get_region(br_lon, br_lat)
            tr_reg = satcam.get_region(tr_lon, tr_lat)
            bl_reg = satcam.get_region(bl_lon, bl_lat)
            ctr_reg = satcam.get_region(lons[i], lats[i])
            cur_regions = [tl_reg, tr_reg, br_reg, bl_reg, ctr_reg]
            big_satim = None
            for region in cur_regions:
                if region in regions:
                    satim, window_transform = satcam.get_image(region)
                    satim = cv2.cvtColor(satim, cv2.COLOR_RGB2BGR)
                    dets = get_detections(satim, region, window_transform, satcam)
                    if savevid:
                        if big_satim is None:
                            big_satim = satim.copy()
                        else:
                            big_satim += satim.copy()
                    if dets is not None:
                        for det in dets:
                            cls, xc, yc, conf = det
                            lon, lat = get_lon_lat_from_cls(int(cls), region)
                            i = int(i)
                            lon = float(lon)
                            lat = float(lat)
                            xc = float(xc)
                            yc = float(yc)
                            conf = float(conf)
                            all_detections.append([i, lon, lat, xc, yc, conf])
                            if check_err:
                                x_err, y_err, lon_px, lat_px = eval_px_error(cls, lon, lat, xc, yc, satcam)
                                errs.append([i, x_err, y_err, cls, region, conf])
                            if savevid or showim:
                                #outim = cv2.putText(outim, str(int(cls)), (int(xc*scale), int(yc*scale)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                                outim = cv2.circle(big_satim, (int(xc), int(yc)), 2, (0,255,0), -1)        
                                if check_err:
                                    outim = cv2.circle(big_satim, (int(lon_px), int(lat_px)), 2, (0,0,255), -1)
                                    outim = cv2.line(big_satim, (int(xc), int(yc)), (int(lon_px), int(lat_px)), (255,0,0), 1)                                  
            if savevid:
                if video is None:
                    video = cv2.VideoWriter('orbits/demos/' + vidname, 0, 5, (vid_w,vid_h))
                
                video.write(big_satim)
            if showim:
                cv2.imshow('satim', big_satim)
            satim = None


    np.save(detection_path + '/' + str(orbit_num).zfill(5) + '_all_detections.npy', all_detections)   
    err_arr = np.array(errs)
    if len(err_arr.shape) > 1:
        xs = np.float32(err_arr[:,1])
        ys = np.float32(err_arr[:,2])
        print('Orbit #', str(orbit_num).zfill(5))
        print('Mean x error:', np.mean(xs), 'Mean y error:', np.mean(ys))
        print('Median x error:', np.median(xs), 'Median y error:', np.median(ys))
        print('Max x error:', np.max(xs), 'Max y error:', np.max(ys))
    np.save(detection_path + '/' + str(orbit_num).zfill(5) + '_errs.npy', err_arr)
    #print(sum(in_timesteps))
        
    #satcam.find_current_regions()
    
    #print(satcam.corner_lonlats)
    #satcam.get_image()
    #print(satcam.get_all_vectors())
    # print(xyz_arr)
            
    # out = nadir_orbit_ecef[in_timesteps]
    # out = json.dumps(out.tolist())
    # with open(str(orbit_num).zfill(5) + '_pose_skip.json', 'w') as f:
    #     f.write(out)

if __name__ == '__main__':
    iterable = range(233, 236)
    process_map(run_sim, iterable, max_workers=1, chunksize=1)