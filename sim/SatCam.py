import numpy as np
from astropy.coordinates import EarthLocation
import cv2
import json
from getMGRS import getMGRS
from tqdm.contrib.concurrent import process_map
import rasterio
import os
from rasterio.merge import merge
from rasterio.windows import from_bounds, transform
import pyproj
import matplotlib.pyplot as plt
import csv

class Vector3D:
    def __init__(self, xyz):
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
    def get(self):
        return np.array((self.x, self.y, self.z))
    
class SatellitePose:
    def __init__(self, pose_array):
        self.position = Vector3D(pose_array[0:3])
        self.dir_vec = Vector3D(pose_array[3:6])
        self.up_vec = Vector3D(pose_array[6:9])
        self.right_vec = Vector3D(pose_array[9:12])
    def get_position(self):
        return self.position.get()
    def get_dir_vec(self):
        return self.dir_vec.get()
    def get_up_vec(self):
        return self.up_vec.get()
    def get_right_vec(self):
        return self.right_vec.get()

class SatCam:
    def __init__(self, sat_pose, hfov, w_px, h_px, regions=None):
        self.sat_pose = sat_pose
        self.hfov = hfov
        self.w_px = w_px
        self.h_px = h_px
        half_width = w_px/2
        half_height = h_px/2
        half_angle = np.deg2rad(hfov)/2
        f = half_width/np.tan(half_angle)
        self.f = f  
        self.vfov = np.rad2deg(2*np.arctan(half_height/self.f))
        self.sat_pos = sat_pose.get_position()
        self.dir_vec = sat_pose.get_dir_vec()
        self.up_vec = -sat_pose.get_up_vec()
        self.right_vec = sat_pose.get_right_vec()

        self.R_wc = np.stack((self.right_vec, self.up_vec, self.dir_vec), axis=1)
        self.R_cw = self.R_wc.T
        self.K = np.array([[f , 0, w_px/2],
                           [0, f, h_px/2],
                           [0, 0, 1]])
        self.K_hom = np.concatenate((self.K, np.zeros((3,1))),axis=1)
        self.K_inv = np.linalg.inv(self.K)
        self.K_inv_hom = np.concatenate((self.K_inv, np.zeros((1,3))),axis=0)
        if regions is None:
            self.regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S',
                            '32S', '32T', '33S', '33T', '52S', '53S', '54S', '54T']  
        else:
            self.regions = regions
        self.grid = getMGRS()
        self.C_cw = self.world_to_pixel_mat()
        self.region_im_dict = {}
        for key, value in self.grid.items():
            self.region_im_dict[key] = None
        self.region_ims = {}
        for region in self.regions:
            self.region_ims[region] = None

    def update_pose(self, sat_pose):
        self.sat_pose = sat_pose
        self.sat_pos = sat_pose.get_position()
        self.dir_vec = sat_pose.get_dir_vec()
        self.up_vec = -sat_pose.get_up_vec()
        self.right_vec = sat_pose.get_right_vec()
        self.R_wc = np.stack((self.right_vec, self.up_vec, self.dir_vec), axis=1)
        self.R_cw = self.R_wc.T
        self.C_cw = self.world_to_pixel_mat()

    def world_to_pixel_mat(self):
        t = self.R_cw @ self.sat_pos
        E = np.concatenate((self.R_cw, -t[:, np.newaxis]),axis=1)
        E_hom = np.concatenate((E, [[0,0,0,1]]), axis=0)
        C_cw = np.matmul(self.K_hom, E_hom)
        return C_cw

    def get_pixel_vector(self, x, y):
        vec = (self.R_wc @ self.K_inv @ np.array([x, y, 1]))
        return vec / np.linalg.norm(vec)

    def get_corner_vectors(self):
        tl_vec = self.get_pixel_vector(0, 0)
        tr_vec = self.get_pixel_vector(self.w_px, 0)
        br_vec = self.get_pixel_vector(self.w_px, self.h_px)
        bl_vec = self.get_pixel_vector(0, self.h_px)
        #print(tl_vec, tr_vec, br_vec, bl_vec)
        return {'tl':tl_vec, 'tr':tr_vec, 'br':br_vec, 'bl':bl_vec}
    
    def get_vector_subset(self, factor=3):
        vectors = {}
        for x in range(self.w_px//factor):
            for y in range(self.h_px//factor):
                vectors[(x, y)] = self.get_pixel_vector(x, y)
        return vectors, factor
    
    def get_xyz_array(self, factor=3):
        vectors, factor = self.get_vector_subset(factor=factor)
        im_w = self.w_px//factor
        im_h = self.h_px//factor
        xyz_array = np.zeros((im_h, im_w, 3))
        for key, vec in vectors.items():
            x, y = key
            p = self.cast_ray_to_earth(vec)
            xyz_array[y, x] = p
        return xyz_array
        

    def cast_ray_to_earth(self, vector):
        semimaj = a = b = 6378137
        semimin = c = 6356752.314245
        x, y, z = self.sat_pos
        u = vector[0]
        v = vector[1]
        w = vector[2]
        value = -a**2*b**2*w*z - a**2*c**2*v*y - b**2*c**2*u*x
        radical = a**2*b**2*w**2 + a**2*c**2*v**2 - a**2*v**2*z**2 + 2*a**2*v*w*y*z - a**2*w**2*y**2 + b**2*c**2*u**2 - b**2*u**2*z**2 + 2*b**2*u*w*x*z - b**2*w**2*x**2 - c**2*u**2*y**2 + 2*c**2*u*v*x*y - c**2*v**2*x**2
        magnitude = a**2*b**2*w**2 + a**2*c**2*v**2 + b**2*c**2*u**2

        if radical < 0:
            return None
        d = (value - a*b*c*np.sqrt(radical)) / magnitude

        if d < 0:
            return None

        return np.array([
            x + d * u,
            y + d * v,
            z + d * w,
        ])

    def ecef_pos_to_px(self, pos):
        x, y, z = pos
        pt = [x, y, z, 1]
        uvw = np.matmul(self.C_cw, pt)
        uv = uvw[:2]/uvw[2]
        return uv

    def convert_cls_to_lon_lat(self, cls, region):
        path = 'landmark_csvs/' + region + '_top_salient.csv'
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            dict_list = list(reader)
        cls_dict = dict_list[cls]
        lon = cls_dict['Centroid Longitude']
        lat = cls_dict['Centroid Latitude']
        return lon, lat

    def convert_xc_yc_to_lon_lat(self, xc, yc, window_transform):
        p = pyproj.Proj('EPSG:3857')
        # convert to pixel coordinates
        x, y = window_transform * (xc, yc)
        # convert to lonlat
        lon, lat = p(x, y, inverse=True)
        return lon, lat


    def get_corner_lonlats(self):
        corner_vectors = self.get_corner_vectors()
        corner_lonlats = {}
        for key, vec in corner_vectors.items():
            p = self.cast_ray_to_earth(vec)
            if p is not None:
                lonlat = EarthLocation.from_geocentric(p[0], p[1], p[2], unit='m')
                corner_lonlats[key] = (lonlat.lon.deg, lonlat.lat.deg)
            else:
                corner_lonlats[key] = None
        return corner_lonlats
    
    def get_region(self, lon, lat):
        for key, bounds in self.grid.items():
            if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
                return key
        return None
    
    def lonlat_to_pixel_coords(self, lon, lat):
        transformer = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'}       
        )
        zs = np.zeros(len(lon))
        x, y, z = transformer.transform(lon, lat, zs, radians=False)
        px = self.ecef_pos_to_px((x, y, z))
        return px

    def find_current_regions(self):
        self.corner_lonlats = corner_lonlats = self.get_corner_lonlats()
        regions = []
        for key, lonlat in corner_lonlats.items():
            if lonlat is not None:
                lon, lat = lonlat
                region = self.get_region(lon, lat)
                regions.append(region)
        num_bounds = []
        char_bounds = []
        for region in regions:
            if region is not None:
                num_bounds.append(int(region[:2]))
                char_bounds.append(region[2])
        if region is not None:
            if min(num_bounds) < 4 and max(num_bounds) > 57:
                num_range = [58, 59, 60, 1, 2, 3]
            else:
                num_range = range(min(num_bounds), max(num_bounds)+1)
            char_range = [chr(i) for i in range(ord(min(char_bounds)), ord(max(char_bounds))+1)]
            out_regions = []
            for num in num_range:
                for char in char_range:
                    out_regions.append(str(num).zfill(2) + char)
            self.current_regions = out_regions
        else:
            self.current_regions = regions
        return self.current_regions
    
    def check_for_landmarks_in_region(self, region):
        landmark_csv_path = 'landmark_csvs/' + region + '_top_salient.csv'
        best_landmarks = np.load('best_classes/' + region + '_best_classes.npy')
        num_landmarks = 0
        corner_lonlats = self.corner_lonlats
        if corner_lonlats['tl'] is None or corner_lonlats['br'] is None:
            return 0
        tl_lon, tl_lat = corner_lonlats['tl']
        br_lon, br_lat = corner_lonlats['br']
        with open(landmark_csv_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for i, row in enumerate(csv_reader):
                if i in best_landmarks:
                    center_lon = float(row['Centroid Longitude'])
                    center_lat = float(row['Centroid Latitude'])
                    if center_lon > tl_lon and center_lon < br_lon and center_lat > br_lat and center_lat < tl_lat:
                        num_landmarks +=1
                        if num_landmarks >= 3:
                            return num_landmarks
        return num_landmarks
        

    def check_for_all_landmarks(self):
        regions = self.find_current_regions()
        num_landmarks = 0
        for region in regions:
            if region in self.regions:
                num_landmarks += self.check_for_landmarks_in_region(region)
                if num_landmarks >= 3:
                    return True
        return False

    def get_lonlat_array(self, xyz_array, factor=3):
        # print(xyz_array[0,0], xyz_array[0,1])
        # xyz_array = cv2.resize(xyz_array, (self.w_px, self.h_px), interpolation=cv2.INTER_AREA)
        # print(xyz_array[0,0], xyz_array[0,1])
        lonlat_array = np.zeros((xyz_array.shape[0], xyz_array.shape[1], 2))
        transformer = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'}
        )
        lons, lats, alts = transformer.transform(xyz_array[:,:,0].ravel(), xyz_array[:,:,1].ravel(), xyz_array[:,:,2].ravel(), radians=False)
        lonlat_array[:,:,0] = lons.reshape(xyz_array[:,:,0].shape)
        lonlat_array[:,:,1] = lats.reshape(xyz_array[:,:,1].shape)
        return lonlat_array

    def choose_region_im(self, region):
        if self.region_im_dict[region] is None:
            self.region_im_dict[region] = np.random.choice(os.listdir('region_ims/' + region))
        return self.region_im_dict[region]

    def get_image(self, region):
        base_path = 'region_ims'
        if region in self.regions:
            if self.region_ims[region] is None:
                region_path = os.path.join(base_path, region)
                regim = self.choose_region_im(region)
                regim_path = os.path.join(region_path, regim)
                with rasterio.open(regim_path) as src:
                    t = src.transform
                    data = src.read((1,2,3))
                    data = np.moveaxis(data, 0, -1)
                    self.region_ims[region] = data, t
            else:
                data, t = self.region_ims[region]
        else:
            return None, None
        
        corner_lonlats = self.corner_lonlats
        tl_lon, tl_lat = corner_lonlats['tl']
        tr_lon, tr_lat = corner_lonlats['tr']
        bl_lon, bl_lat = corner_lonlats['bl']
        br_lon, br_lat = corner_lonlats['br']

        min_lon = min([tl_lon, tr_lon, bl_lon, br_lon])
        max_lon = max([tl_lon, tr_lon, bl_lon, br_lon])
        min_lat = min([tl_lat, tr_lat, bl_lat, br_lat])
        max_lat = max([tl_lat, tr_lat, bl_lat, br_lat])

        p = pyproj.Proj('EPSG:3857')
        min_x, min_y = p(min_lon, min_lat)
        max_x, max_y = p(max_lon, max_lat)

        satim, window_transform = self.get_windowed_image(data, min_x, min_y, max_x, max_y, t)
        return satim, window_transform

    def get_windowed_image(self, data, min_x, min_y, max_x, max_y, t):
        
        window = from_bounds(min_x, min_y, max_x, max_y, t)
        window_transform = rasterio.windows.transform(window, t)

        min_x_px, min_y_px = ~t*(min_x, max_y)
        max_x_px, max_y_px = ~t*(max_x, min_y)
        
        min_x_px = int(min_x_px)
        min_y_px = int(min_y_px)
        max_x_px = int(max_x_px)
        max_y_px = int(max_y_px)
        
        im_h = int(max_y_px - min_y_px)
        im_w = int(max_x_px - min_x_px)

        image = np.zeros((im_h, im_w, data.shape[2]), dtype=data.dtype)
        data_h, data_w, _ = data.shape

        data_min_x_px = max(0, min_x_px)
        data_max_x_px = min(data_w, max_x_px)
        data_min_y_px = max(0, min_y_px)
        data_max_y_px = min(data_h, max_y_px)

        if min_x_px < 0:
            im_min_x_px = -min_x_px
        else:
            im_min_x_px = 0
        if min_y_px < 0:
            im_min_y_px = -min_y_px
        else:
            im_min_y_px = 0
        if max_x_px > data_w:
            im_max_x_px = int(im_w - (max_x_px - data_w))
        else:
            im_max_x_px = im_w
        if max_y_px > data_h:
            im_max_y_px = int(im_h - (max_y_px - data_h))
        else:
            im_max_y_px = im_h

        image[im_min_y_px:im_max_y_px, im_min_x_px:im_max_x_px] = data[data_min_y_px:data_max_y_px, data_min_x_px:data_max_x_px]

        return image, window_transform
    
    
    # def get_image(self, ctr_region):
    #     base_path = 'region_ims'
    #     if ctr_region in self.regions:
    #         region_path = os.path.join(base_path, ctr_region)
    #         regim = self.choose_region_im(ctr_region)
    #         regim_path = os.path.join(region_path, regim)
    #     else:
    #         return None, None
    #     corner_lonlats = self.corner_lonlats
    #     tl_lon, tl_lat = corner_lonlats['tl']
    #     tr_lon, tr_lat = corner_lonlats['tr']
    #     bl_lon, bl_lat = corner_lonlats['bl']
    #     br_lon, br_lat = corner_lonlats['br']

    #     # Get minimum and maximum lonlats
    #     min_lon = min(tl_lon, bl_lon)
    #     max_lon = max(tr_lon, br_lon)
    #     min_lat = min(bl_lat, br_lat)
    #     max_lat = max(tl_lat, tr_lat)

    #     # make projector from epsg:4326 to epsg:3857
    #     p = pyproj.Proj('EPSG:3857')

    #     # Convert lonlats to xys
    #     min_x, min_y = p(min_lon, min_lat)
    #     max_x, max_y = p(max_lon, max_lat)

    #     # Open the raster image
    #     with rasterio.open(regim_path) as src:
    #         t = src.transform
    #         window = from_bounds(min_x, min_y, max_x, max_y, t) # get window from bounds
    #         window_transform = rasterio.windows.transform(window, t) # get transform from window
    #         windowed_im = src.read(window=window, boundless=True) # read the windowed image
    #         windowed_im = np.moveaxis(windowed_im, 0, -1)   # move axis to (h, w, 3)
        
    #     return windowed_im, window_transform
        


        


    # def get_image(self, ctr_region):
    #     base_path = 'region_ims'
    #     if ctr_region in self.regions:
    #         region_path = os.path.join(base_path, ctr_region)
    #         regim = self.choose_region_im(ctr_region)
    #         regim_path = os.path.join(region_path, regim)
    #     else:
    #         return None
    #     corner_lonlats = self.corner_lonlats
    #     if self.region_ims[ctr_region] is None:
    #         with rasterio.open(regim_path) as src:
    #             t = src.transform           
    #             data = src.read((1,2,3))
    #             data = np.moveaxis(data, 0, -1)
    #             self.region_ims[ctr_region] = [data, t]
    #     else:
    #         data, t = self.region_ims[ctr_region]
    #     transformer = rasterio.transform.AffineTransformer(t)
    #     tl_lon, tl_lat = corner_lonlats['tl']
    #     tr_lon, tr_lat = corner_lonlats['tr']
    #     bl_lon, bl_lat = corner_lonlats['bl']
    #     br_lon, br_lat = corner_lonlats['br']
    #     data_h, data_w, _ = data.shape
    #     p = pyproj.Proj('EPSG:3857')
    #     tl_x, tl_y = p(tl_lon, tl_lat)
    #     tr_x, tr_y = p(tr_lon, tr_lat)
    #     bl_x, bl_y = p(bl_lon, bl_lat)
    #     br_x, br_y = p(br_lon, br_lat)
    #     min_x = min(tl_x, bl_x)
    #     max_x = max(tr_x, br_x)
    #     min_y = min(tl_y, tr_y, bl_y, br_y)
    #     max_y = max(tl_y, tr_y, bl_y, br_y)

    #     y_px, x_px = transformer.rowcol([tl_x, tr_x, bl_x, br_x], [tl_y, tr_y, bl_y, br_y])
    #     min_x_px = min(x_px)
    #     max_x_px = max(x_px)
    #     min_y_px = min(y_px)
    #     max_y_px = max(y_px)

    #     im_w = max_x_px - min_x_px
    #     im_h = max_y_px - min_y_px

    #     im = np.zeros((im_h, im_w, 3), dtype=np.uint8)

    #     data_min_x_px = max(0, min_x_px)
    #     data_max_x_px = min(data_w, max_x_px)
    #     data_min_y_px = max(0, min_y_px)
    #     data_max_y_px = min(data_h, max_y_px)

    #     if min_x_px < 0:
    #         im_min_x_px = -min_x_px
    #     else:
    #         im_min_x_px = 0
    #     if min_y_px < 0:
    #         im_min_y_px = -min_y_px
    #     else:
    #         im_min_y_px = 0
    #     if max_x_px > data_w:
    #         im_max_x_px = im_w - (max_x_px - data_w)
    #     else:
    #         im_max_x_px = im_w
    #     if max_y_px > data_h:
    #         im_max_y_px = im_h - (max_y_px - data_h)
    #     else:
    #         im_max_y_px = im_h

    #     im[im_min_y_px:im_max_y_px, im_min_x_px:im_max_x_px] = data[data_min_y_px:data_max_y_px, data_min_x_px:data_max_x_px]
    #     window = from_bounds(min_x, min_y, max_x, max_y, t)
    #     window_transform = rasterio.windows.transform(window, t)
    #     transformer = rasterio.transform.AffineTransformer(window_transform)

    #     tl_y_px, tl_x_px = transformer.rowcol(tl_x, tl_y)
    #     tr_y_px, tr_x_px = transformer.rowcol(tr_x, tr_y)
    #     bl_y_px, bl_x_px = transformer.rowcol(bl_x, bl_y)
    #     br_y_px, br_x_px = transformer.rowcol(br_x, br_y)

    #     points_dst = np.array([[0, 0], [im_w, 0], [0, im_h], [im_w, im_h]], dtype=np.float32)
    #     points_src = np.array([[tl_x_px, tl_y_px], [tr_x_px, tr_y_px], [bl_x_px, bl_y_px], [br_x_px, br_y_px]], dtype=np.float32)

    #     M = cv2.getPerspectiveTransform(points_src, points_dst)
    #     warped = cv2.warpPerspective(im, M, (im_w, im_h))
    #     plt.imshow(warped)


    #     # window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
    #     # window_transform = rasterio.windows.transform(window, src.transform)
    #     # data = src.read((1,2,3), window=window, boundless = True)
    #     # data = np.moveaxis(data, 0, -1)
    #     # transformer = rasterio.transform.AffineTransformer(window_transform)
    #     # tl_y_px, tl_x_px = transformer.rowcol(tl_x, tl_y)
    #     # tr_y_px, tr_x_px = transformer.rowcol(tr_x, tr_y)
    #     # bl_y_px, bl_x_px = transformer.rowcol(bl_x, bl_y)
    #     # br_y_px, br_x_px = transformer.rowcol(br_x, br_y)
    #     # h,w = data.shape[:2]
    #     # points_dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    #     # points_src = np.array([[tl_x_px, tl_y_px], [tr_x_px, tr_y_px], [bl_x_px, bl_y_px], [br_x_px, br_y_px]], dtype=np.float32)
    #     # M = cv2.getPerspectiveTransform(points_src, points_dst)
    #     # warped = cv2.warpPerspective(data, M, (w, h))
    #     resized = cv2.resize(warped, (self.w_px, self.h_px))
    #     return resized

        #self.find_current_regions()
        # for region in self.current_regions:
        #     if region is None:
        #         continue
        #     region_path = os.path.join(base_path, region)
        #     if os.path.exists(region_path):
        #         random_regim = self.choose_region_im(region)
        #         regim_path = os.path.join(region_path, random_regim)
        #         regims.append(regim_path)
        # corner_lonlats = self.corner_lonlats
        # if len(regims) > 1:
        #     data, affine = rasterio.merge.merge(regims)
        #     data = np.moveaxis(data, 0, -1)
        #     tl_lon, tl_lat = corner_lonlats['tl']
        #     tr_lon, tr_lat = corner_lonlats['tr']
        #     bl_lon, bl_lat = corner_lonlats['bl']
        #     br_lon, br_lat = corner_lonlats['br']
        #     max_lon = max([tl_lon, tr_lon, bl_lon, br_lon])
        #     min_lon = min([tl_lon, tr_lon, bl_lon, br_lon])
        #     max_lat = max([tl_lat, tr_lat, bl_lat, br_lat])
        #     min_lat = min([tl_lat, tr_lat, bl_lat, br_lat])
        #     transformer = rasterio.transform.AffineTransformer(affine)
        #     tl_y_px, tl_x_px = transformer.rowcol(tl_lon, tl_lat)
        #     tr_y_px, tr_x_px = transformer.rowcol(tr_lon, tr_lat)
        #     bl_y_px, bl_x_px = transformer.rowcol(bl_lon, bl_lat)
        #     br_y_px, br_x_px = transformer.rowcol(br_lon, br_lat)
        #     max_lat_px, min_lon_px = transformer.rowcol(min_lon, min_lat)
        #     min_lat_px, max_lon_px = transformer.rowcol(max_lon, max_lat)

        #     dst_width = self.w_px
        #     dst_height = self.h_px
        #     h, w = data.shape[:2]
        #     points_dst = np.array([[min_lon_px, min_lat_px], [max_lon_px, min_lat_px], [max_lon_px, max_lat_px], [min_lon_px, max_lat_px]]).astype(np.float32)
        #     points_src = np.array([[tl_x_px, tl_y_px], [tr_x_px, tr_y_px], [br_x_px, br_y_px], [bl_x_px, bl_y_px]]).astype(np.float32)

        #     M = cv2.getPerspectiveTransform(points_dst, points_src)
        #     warped = cv2.warpPerspective(data, M, (w, h))
        #     cropped = warped[min_lat_px:max_lat_px, min_lon_px:max_lon_px]
        #     if len(cropped) == 0 or len(cropped[0]) == 0:
        #         print('why is this 0')
        #         return None
        #     resized = cv2.resize(cropped, (dst_width, dst_height))

        # elif len(regims) == 1:
        #     with rasterio.open(regims[0]) as src:
        #         tl_lon, tl_lat = corner_lonlats['tl']
        #         tr_lon, tr_lat = corner_lonlats['tr']
        #         bl_lon, bl_lat = corner_lonlats['bl']
        #         br_lon, br_lat = corner_lonlats['br']
        #         max_lon = max([tl_lon, tr_lon, bl_lon, br_lon])
        #         min_lon = min([tl_lon, tr_lon, bl_lon, br_lon])
        #         max_lat = max([tl_lat, tr_lat, bl_lat, br_lat])
        #         min_lat = min([tl_lat, tr_lat, bl_lat, br_lat])
        #         window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        #         window_transform = transform(window, src.transform)
        #         data = src.read((1,2,3), window=window, boundless = True)
        #         data = np.moveaxis(data, 0, -1)
        #         transformer = rasterio.transform.AffineTransformer(window_transform)
        #         tl_y_px, tl_x_px = transformer.rowcol(tl_lon, tl_lat)
        #         tr_y_px, tr_x_px = transformer.rowcol(tr_lon, tr_lat)
        #         bl_y_px, bl_x_px = transformer.rowcol(bl_lon, bl_lat)
        #         br_y_px, br_x_px = transformer.rowcol(br_lon, br_lat)

        #         dst_width = self.w_px
        #         dst_height = self.h_px
        #         h, w = data.shape[:2]
        #         points_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
        #         points_src = np.array([[tl_x_px, tl_y_px], [tr_x_px, tr_y_px],
        #                                 [br_x_px, br_y_px], [bl_x_px, bl_y_px]]).astype(np.float32)
        #         M = cv2.getPerspectiveTransform(points_dst, points_src)
        #         warped = cv2.warpPerspective(data, M, (w, h))
        #         resized = cv2.resize(warped, (dst_width, dst_height))
        # else:
        #     return None
        # return resized

    # def get_image(self, factor=3):
    #     base_path = 'sim\\region_ims'
    #     regims = []
    #     self.find_current_regions()
    #     for region in self.current_regions:
    #         if region in self.regions:
    #             region_path = os.path.join(base_path, region)
    #             regim_list = os.listdir(region_path)
    #             random_regim = np.random.choice(regim_list)
    #             regim_path = os.path.join(base_path, region, random_regim)
    #             regims.append(regim_path)
    #     if len(regims) > 1:
    #         xyz_arr = self.get_xyz_array(factor)
    #         lonlat_array = self.get_lonlat_array(xyz_arr, factor)
    #         data, transform = merge(regims)
    #         transformer = rasterio.transform.AffineTransformer(transform)
    #         data = np.moveaxis(data, 0, -1)
    #         lonravel = lonlat_array[:,:,0].ravel()
    #         latravel = lonlat_array[:,:,1].ravel()
    #         (rows, cols) = transformer.rowcol(lonravel, latravel) # input is (x, y) output is (row, col)
    #         rows, cols = np.array(rows), np.array(cols)
    #         rowcol = np.stack([rows, cols],axis=1)
    #         height, width = data.shape[0], data.shape[1]
    #         ltw = rowcol[:, 1] < width
    #         lth = rowcol[:, 0] < height
    #         gt0h = rowcol[:, 0] >= 0
    #         gt0w = rowcol[:, 1] >= 0
    #         inbounds_px = ltw*lth*gt0h*gt0w
    #         px = rowcol[inbounds_px]
    #         tmp = np.zeros((height, width), dtype=bool)
    #         for p in px:
    #             tmp[p[0], p[1]] = True
    #         h, w = lonlat_array.shape[0], lonlat_array.shape[1]
    #         out_im = np.zeros((h*w, 3), dtype=np.uint8)
    #         out_im[inbounds_px] = data[tmp]
    #         out_im = out_im.reshape((h, w, 3)) 
    #         plt.imshow(out_im)
    #         plt.show()
    #         # cv2.imshow('image', out_im)
    #         # cv2.waitKey(0)
    #         # print(rows, cols)
    #         # rows = rows.reshape(lonlat_array[:,:,0].shape)
    #         # cols = cols.reshape(lonlat_array[:,:,1].shape)
    #         # print(rows, cols)

    #     elif len(regims) == 1:
    #         with rasterio.open(regims[0]) as src:
    #             xyz_arr = self.get_xyz_array(factor)
    #             lonlat_array = self.get_lonlat_array(xyz_arr, factor)
    #             data = src.read()
    #             data = np.moveaxis(data, 0, -1)
    #             lonravel = lonlat_array[:,:,0].ravel()
    #             latravel = lonlat_array[:,:,1].ravel()
    #             (rows, cols) = src.index(lonravel, latravel) # input is (x, y) output is (row, col)
    #             rows, cols = np.array(rows), np.array(cols)
    #             rowcol = np.stack([rows, cols], axis=1)
    #             height, width = data.shape[0], data.shape[1]
    #             ltw = rowcol[:, 1] < width
    #             lth = rowcol[:, 0] < height
    #             gt0h = rowcol[:, 0] >= 0
    #             gt0w = rowcol[:, 1] >= 0
    #             inbounds_px = ltw*lth*gt0h*gt0w
    #             px = rowcol[inbounds_px]
    #             tmp = np.zeros((height, width), dtype=bool)
    #             for p in px:
    #                 tmp[p[0], p[1]] = True
                
    #             h, w = lonlat_array.shape[0], lonlat_array.shape[1]
    #             out_im = np.zeros((h*w, 3), dtype=np.uint8)
    #             out_im[inbounds_px] = data[tmp]
    #             out_im = out_im.reshape((h, w, 3)) 
    #             plt.imshow(out_im)
    #             plt.show()
    #             # cv2.imshow('image', out_im)
    #             # cv2.waitKey(0)
    #             # rows = rows.reshape(lonlat_array[:,:,0].shape)
    #             # cols = cols.reshape(lonlat_array[:,:,1].shape)
    #             # print(rows, cols)
    #     return out_im