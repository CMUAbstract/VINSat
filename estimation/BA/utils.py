import numpy as np
import math
import torch
# from lietorch.groups import SO3, SE3
import ipdb
from BA.BA_utils import *
"""
Computes the Bias-Precession-Nutation matrix transforming the GCRS to the 
CIRS intermediate reference frame. This transformation corrects for the 
bias, precession, and nutation of Celestial Intermediate Origin (CIO) with
respect to inertial space.

Arguments:
- `epc::Epoch`: Epoch of transformation

Returns:
- `rc2i::Matrix{<:Real}`: 3x3 Rotation matrix transforming GCRS -> CIRS
"""
def bias_precession_nutation(epc):
    # Constants of IAU 2006A transofrmation
    DMAS2R =  4.848136811095359935899141e-6 / 1.0e3
    dx06   =  0.0001750*DMAS2R
    dy06   = -0.0002259*DMAS2R

    # Compute X, Y, s terms using low-precision series terms
    x, y, s = iauXys00b(MJD_ZERO, mjd(epc, tsys="TT"))

    # Apply IAU2006 Offsets
    x += dx06
    y += dy06

    # Compute transformation and return
    rc2i = iauC2ixys(x, y, s)

    return rc2i


def get_r_sun_moon_PN(r_suns, r_moons, PNs, h, t):
    idx = int((2*t)/h)
    r_sun = r_suns[idx]
    r_moon = r_moons[idx]
    PN = PNs[idx]
    return r_sun, r_moon, PN


"""
Compute the Modified Julian Date for a specific epoch

Arguments:
- `epc::Epoch`: Epoch
- `tsys::String`: Time system to return output in

Returns:
- `mjd::Real`: Julian date of the epoch in the requested time system
"""
def mjd(epc, tsys):
    offset = time_system_offset(epc, "TAI", tsys)
    return (epc.days + (epc.seconds + epc.nanoseconds/1.0e9 + offset)/86400.0) - MJD_ZERO


"""
Offset of Modified Julian Days representation with respect to Julian Days. For 
a time, t, MJD_ZERO is equal to:

    MJD_ZERO = t_jd - t_mjd

Where t_jd is the epoch represented in Julian Days, and t_mjd is the epoch in
Modified Julian Days.

O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and 
Applications_, 2012.
"""
MJD_ZERO = 2400000.5

"""
Modified Julian Date of January 1, 2000 00:00:00. Value is independent of time
scale.

O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and 
Applications_, 2012.
"""
MJD2000 = 51544.0

"""
Earth's equatorial radius. [m]

GGM05s Gravity Model
"""
R_EARTH = 6.378136300e6               # [m] GGM05s Value

"""
Compute the Sun's position in the EME2000 inertial frame through the use
of low-precision analytical functions.

Argument:
- `epc::Epoch`: Epoch

Returns:
- `r_sun::AbstractArray{<:Real, 1}`: Position vector of the Sun in the Earth-centered inertial fame.

Notes:
1. The EME2000 inertial frame is for most purposes equivalent to the GCRF frame.

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.70-73.
"""
def sun_position(epc):
    # Constants
    mjd_tt  = mjd(epc, tsys="TT")      # MJD of epoch in TT
    epsilon = 23.43929111*np.pi/180.0     # Obliquity of J2000 ecliptic
    T       = (mjd_tt-MJD2000)/36525.0 # Julian cent. since J2000

    # Variables

    # Mean anomaly, ecliptic longitude and radius
    M = 2.0*np.pi * math.modf(0.9931267 + 99.9973583*T)[1]                 # [rad]
    L = 2.0*np.pi * math.modf(0.7859444 + M/(2.0*np.pi) + (6892.0*np.sin(M)+72.0*np.sin(2.0*M)) / 1296.0e3)[1] # [rad]
    r = 149.619e9 - 2.499e9*np.cos(M) - 0.021e9*np.cos(2*M)           # [m]

    # Equatorial position vector
    p_sun = Rx(-epsilon) * [r*np.cos(L), r*np.sin(L), 0.0]

    return p_sun

"""
Compute the Moon's position in the EME2000 inertial frame through the use
of low-precision analytical functions.

Argument:
- `epc::Epoch`: Epoch

Returns:
- `r_moon::AbstractArray{<:Real, 1}`: Position vector of the Moon in the Earth-centered inertial fame.

Notes:
1. The EME2000 inertial frame is for most purposes equivalent to the GCRF frame.

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.70-73.
"""
def moon_position(epc):
    # Constants
    mjd_tt  = mjd(epc, tsys="TT")      # MJD of epoch in TT
    epsilon = 23.43929111*np.pi/180.0     # Obliquity of J2000 ecliptic
    T       = (mjd_tt-MJD2000)/36525.0 # Julian cent. since J2000

    # Mean elements of lunar orbit
    L_0 =     math.modf(0.606433 + 1336.851344*T)[0] # Mean longitude [rev] w.r.t. J2000 equinox
    l   = 2.0*np.pi*math.modf(0.374897 + 1325.552410*T)[0] # Moon's mean anomaly [rad]
    lp  = 2.0*np.pi*math.modf(0.993133 +   99.997361*T)[0] # Sun's mean anomaly [rad]
    D   = 2.0*np.pi*math.modf(0.827361 + 1236.853086*T)[0] # Diff. long. Moon-Sun [rad]
    F   = 2.0*np.pi*math.modf(0.259086 + 1342.227825*T)[0] # Argument of latitude 


    # Ecliptic longitude (w.r.t. equinox of J2000)
    dL = + 22640*np.sin(l) - 4586*np.sin(l-2*D) + 2370*np.sin(2*D) +  769*np.sin(2*l) \
         - 668*np.sin(lp) - 412*np.sin(2*F) - 212*np.sin(2*l-2*D) - 206*np.sin(l+lp-2*D) \
         + 192*np.sin(l+2*D) - 165*np.sin(lp-2*D) - 125*np.sin(D) - 110*np.sin(l+lp) \
         + 148*np.sin(l-lp) - 55*np.sin(2*F-2*D)

    L = 2.0*np.pi * math.modf(L_0 + dL/1296.0e3)[0]  # [rad]

    # Ecliptic latitude
    S  = F + (dL+412*np.sin(2*F)+541*np.sin(lp)) * AS2RAD 
    h  = F-2*D
    N  = - 526*np.sin(h) + 44*np.sin(l+h) - 31*np.sin(-l+h) - 23*np.sin(lp+h) \
         + 11*np.sin(-lp+h) - 25*np.sin(-2*l+F) + 21*np.sin(-l+F)
    B  = (18520.0*np.sin(S) + N) * AS2RAD   # [rad]

    # Distance [m]
    r = + 385000e3 - 20905e3*np.cos(l) - 3699e3*np.cos(2*D-l) - 2956e3*np.cos(2*D) \
        - 570e3*np.cos(2*l) + 246e3*np.cos(2*l-2*D) - 205e3*np.cos(lp-2*D) \
        - 171e3*np.cos(l+2*D) - 152e3*np.cos(l+lp-2*D)   

    # Equatorial coordinates
    p_moon = Rx(-epsilon).dot(np.array([r*np.cos(L)*np.cos(B), r*np.sin(L)*np.cos(B), r*np.sin(B)]))

    return p_moon

"""
Rotation matrix, for a rotation about the x-axis.

Arguments:
- `angle::Real`: Counter-clockwise angle of rotation as viewed looking back along the postive direction of the rotation axis.
- `use_degrees:Bool`: If `true` interpret input as being in degrees.

Returns:
- `r::AbstractArray{<:Real, 2}`: Rotation matrix

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.27.
"""
def Rx(angle, use_degrees=False):
    
    if use_degrees:
        angle *= np.pi/180.0

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([[+1.0, 0.0, 0.0],
                     [0.0, +c, +s],
                     [0.0, -s, +c]])


"""
Computes the local density using the Harris-Priester density model.

Arguments:
- `x::AbstractArray{<:Real, 1}`: Satellite Cartesean state in the inertial reference frame [m; m/s]
- `r_sun::AbstractArray{<:Real, 1}`: Position of sun in inertial frame.

Returns:
- `rho:Float64`: Local atmospheric density [kg/m^3]

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.89-91.
"""
def density_harris_priester(x, r_sun):
    # Harris-Priester Constants
    hp_upper_limit =   1000.0          # Upper height limit [km]
    hp_lower_limit =    100.0          # Lower height limit [km]
    hp_ra_lag      = 0.523599          # Right ascension lag [rad]
    hp_n_prm       =        3          # Harris-Priester parameter 
                                        # 2(6) low(high) inclination
    hp_N           = 50                # Number of coefficients

    # Height [km]
    hp_h = torch.tensor([100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,     
            210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,     
            320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,     
            520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,     
            720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0,1000.0])

    # Minimum density [g/km^3]
    hp_c_min = torch.tensor([4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,         
                8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,         
                9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,         
                2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,         
                2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,         
                2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,         
                4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,         
                1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,         
                1.560e-03, 1.150e-03])

    # Maximum density [g/km^3]
    hp_c_max = torch.tensor([4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,         
                8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,         
                1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,         
                4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,         
                7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,         
                1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,         
                4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,         
                1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,         
                2.360e-02, 1.810e-02])

    # Satellite height
    geod   = sECEFtoGEOD(x[:,:3], use_degrees=True)
    height = geod[:,2]/1.0e3 # height in [km]

    # Exit with zero density outside height model limits
    if height.mean() > hp_upper_limit or height.mean() < hp_lower_limit:
        return 0.0
    
    # Sun right ascension, declination
    r_sun = torch.tensor(r_sun).float()
    ra_sun  = torch.atan( r_sun[1], r_sun[0] )
    dec_sun = torch.atan( r_sun[2], torch.sqrt( r_sun[0]**2 + r_sun[1]**2 ) )


    # Unit vector u towards the apex of the diurnal bulge
    # in inertial geocentric coordinates
    c_dec = torch.cos(dec_sun)
    u     = torch.tensor([c_dec * torch.cos(ra_sun + hp_ra_lag),
             c_dec * torch.sin(ra_sun + hp_ra_lag),
             torch.sin(dec_sun)]).float().unsqueeze(0).repeat(x.shape[0],1)


    # Cosine of half angle between satellite position vector and
    # apex of diurnal bulge
    c_psi2 = 0.5 + 0.5 * torch.dot(x[:,:3], u)/x[:,:3].norm(dim=-1)

    # Height index search and exponential density interpolation
    # ih = 0                            # section index reset
    # for i in range(hp_N):                   # loop over N_Coef height regimes
    #     if height >= hp_h[i] and height < hp_h[i+1]:
    #         ih = i                    # ih identifies height section
    #         break
    ih = torch.ge(height.unsqueeze(-1),hp_h[:1].unsqueeze(0)).sum(dim=-1).long() - 1

    h_min = ( hp_h[ih] - hp_h[ih+1] )/torch.log( hp_c_min[ih+1]/hp_c_min[ih] )
    h_max = ( hp_h[ih] - hp_h[ih+1] )/torch.log( hp_c_max[ih+1]/hp_c_max[ih] )

    d_min = hp_c_min[ih] * torch.exp( (hp_h[ih]-height)/h_min )
    d_max = hp_c_max[ih] * torch.exp( (hp_h[ih]-height)/h_max )

    # Density computation
    density = d_min + (d_max-d_min) * c_psi2**hp_n_prm

    # Convert from g/km^3 to kg/m^3
    density *= 1.0e-12

    # Finished
    return density

# def density_harris_priester_(epc, x):
#     r_sun = sun_position(epc)
#     return density_harris_priester(x, r_sun)


"""
Computes the perturbing, non-conservative acceleration caused by atmospheric
drag assuming that the ballistic properties of the spacecraft are captured by
the coefficient of drag.

Arguments:
- `x::AbstractArray{<:Real, 1}`: Satellite Cartesean state in the inertial reference frame [m; m/s]
- `rho::Real`: atmospheric density [kg/m^3]
- `mass::Real`: Spacecraft mass [kg]
- `area::Real`: Wind-facing cross-sectional area [m^2]
- `Cd::Real`: coefficient of drag [dimensionless]
- `T::AbstractArray{<:Real, 2}`: Rotation matrix from the inertial to the true-of-date frame

Return:
- `a::AbstractArray{<:Real, 1}`: Acceleration due to drag in the X, Y, and Z inertial directions. [m/s^2]

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.83-86.
"""
def accel_drag(x, rho, mass, area, Cd, T):

    # Constants
    omega = torch.tensor([0, 0, OMEGA_EARTH]).float()

    # Position and velocity in true-of-date system
    ipdb.set_trace()
    r_tod = (T * x[:,:3].unsqueeze(-2)).sum(dim=-1)
    v_tod = (T * x[:,3:].unsqueeze(-2)).sum(dim=-1)

    # Velocity relative to the Earth's atmosphere
    v_rel = v_tod - torch.cross(omega, r_tod)
    v_abs = v_rel.norm(dim=-1)

    # Acceleration 
    a_tod  = -0.5*Cd*(area/mass)*rho*v_abs*v_rel
    a_drag = (T*a_tod.unsqueeze(-1)).sum(dim=-2)

    return a_drag


"""
Convert geodetic coordinaties to Earth-fixed position

Arguments:
- `ecef::AbstractArray{<:Real, 1}`: Earth-fixed position [m]
- `use_degrees:Bool`: If `true` returns result in units of degrees

Returns:
- `geod::AbstractArray{<:Real, 1}`: Geocentric coordinates (lon, lat, altitude) [rad] / [deg]
"""
def sECEFtoGEOD(ecef, use_degrees=False):
    # Expand ECEF coordinates
    x, y, z = ecef[:,0], ecef[:,1], ecef[:,2]

    # Compute intermediate quantities
    epsilon  = 1e-8#eps(Float64) * 1.0e3 * WGS84_a # Convergence requirement as function of machine precision
    rho2 = x**2 + y**2                      # Square of the distance from the z-axis
    dz   = ECC2 * z
    N    = 0.0

    # Iteratively compute refine coordinates
    while True:
        zdz    = z + dz
        Nh     = torch.sqrt(rho2 + zdz**2)
        sinphi = zdz / Nh
        N      = WGS84_a / torch.sqrt(1.0 - ECC2 * sinphi**2)
        dz_new = N * ECC2 * sinphi

        # Check convergence requirement
        # ipdb.set_trace()
        if (dz - dz_new < epsilon).all():
            break

        dz = dz_new

    # Extract geodetic coordinates
    zdz = z + dz
    ipdb.set_trace()
    lat = torch.atan2(zdz, torch.sqrt(rho2))
    lon = torch.atan2(y, x)
    alt = torch.sqrt(rho2 + zdz**2) - N

    # Convert output to degrees
    if use_degrees:
        lat = lat*180.0/np.pi
        lon = lon*180.0/np.pi

    return torch.stack([lon, lat, alt], dim=-1)


"""
Nominal solar radiation pressure at 1 AU. [N/m^2]

O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and 
Applications_, 2012.
"""
P_SUN       = 4.560E-6                    # [N/m^2] (~1367 W/m^2) Solar radiation pressure at 1 AU


"""
Astronomical Unit. Equal to the mean distance of the Earth from the sun.
TDB-compatible value. [m]

P. Gérard and B. Luzum, IERS Technical Note 36, 2010
"""
AU          = 1.49597870700e11            # [m] Astronomical Unit IAU 2010


"""
Gravitational constant of the Sun. [m^3/s^2]

O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and 
Applications_, 2012.
"""
GM_SUN      = 132712440041.939400*1e9     # Gravitational constant of the Sun

"""
Gravitational constant of the Moon. [m^3/s^2]

O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and 
Applications_, 2012.
"""
GM_MOON     = 4902.800066*1e9

"""
Earth's Gravitational constant [m^3/s^2]

O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and 
Applications_, 2012.
"""
GM_EARTH    = 3.986004415e14              # [m^3/s^2] GGM05s Value

"""
Constant to convert arcseconds to radians. Equal to 2pi/(360*3600). [rad/as]
"""
AS2RAD = 2.0*np.pi/360.0/3600.0

"""
Earth's semi-major axis as defined by the WGS84 geodetic system. [m]

NIMA Technical Report TR8350.2
"""
WGS84_a     = 6378137.0                   # WGS-84 semi-major axis

"""
Earth's ellipsoidal flattening.  WGS84 Value.

NIMA Technical Report TR8350.2
"""
WGS84_f     = 1.0/298.257223563           # WGS-84 flattening

# Intermidiate calculations calculations
ECC2 = WGS84_f * (2.0 - WGS84_f) # Square of eccentricisty

"""
Earth axial rotation rate. [rad/s]

D. Vallado, _Fundamentals of Astrodynamics and Applications_ (4th Ed.), p. 222, 2010
"""
OMEGA_EARTH = 7.292115146706979e-5        # [rad/s] Taken from Vallado 4th Ed page 222

"""Computes the perturbing acceleration due to direct solar radiation 
pressure assuming the reflecting surface is a flat plate pointed directly at
the Sun.

Arguments:
- `x::AbstractArray{<:Real, 1}`: Satellite Cartesean state in the inertial reference frame [m; m/s]

Returns:
- `a::AbstractArray{<:Real, 1}`: Satellite acceleration due to solar radiation pressure [m/s^2]

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.77-79.
"""
def accel_srp(x, r_sun, mass=0.0, area=0.0, CR=1.8, p0=P_SUN, au=AU):
    # Spacecraft position vector
    r = x[:,:3]

    # Relative position vector of spacecraft w.r.t. Sun
    d = r - r_sun

    # Acceleration due to moon point mass
    a_srp = d * (CR*(area/mass)*p0*AU**2 / ((d.norm(dim=-1))**3))

    # Return
    return a_srp


"""
Computes the acceleration of a satellite in the inertial frame due to the
gravitational attraction of the Sun.

Arguments:
- `x::AbstractArray{<:Real, 1}`: Satellite Cartesean state in the inertial reference frame [m; m/s]
- `r_sun::AbstractArray{<:Real, 1}`: Position of sun in inertial frame.

Return:
- `a::AbstractArray{<:Real, 1}`: Acceleration due to the Sun's gravity in the inertial frame [m/s^2]

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.69-70.
"""
def accel_thirdbody_sun_(x, r_sun):
    # Acceleration due to sun point mass
    a_sun = accel_point_mass(x[:,:3], r_sun, GM_SUN)

    return a_sun

def accel_thirdbody_sun(epc, x):
    # Compute solar position
    r_sun = sun_position(epc)

    # Acceleration due to sun point mass
    a_sun = accel_point_mass(x[:,:3], r_sun, GM_SUN)

    return a_sun

"""
Computes the acceleration of a satellite in the inertial frame due to the
gravitational attraction of the Moon.

Arguments:
- `x::AbstractArray{<:Real, 1}`: Satellite Cartesean state in the inertial reference frame [m; m/s]
- `r_moon::AbstractArray{<:Real, 1}`: Position of moon in inertial frame.

Returns:
- `a::AbstractArray{<:Real, 1}`: Acceleration due to the Moon's gravity in the inertial frame [m/s^2]

References:
1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.69-70.
"""
def accel_thirdbody_moon_(x, r_moon):
    # Acceleration due to moon point mass
    a_moon = accel_point_mass(x[:,:3], r_moon, GM_MOON)

    return a_moon

def accel_thirdbody_moon(epc, x):
    # Compute solar position
    r_moon = moon_position(epc)

    # Acceleration due to moon point mass
    a_moon = accel_point_mass(x[:,:3], r_moon, GM_MOON)

    return a_moon
"""
Computes the acceleration of a satellite caused by a point-mass approximation 
of the central body. Returns the acceleration vector of the satellite.

Assumes the satellite is much, much less massive than the central body.

Arguments:
- `r_sat::AbstractArray{<:Real, 1}`: satellite position in a commonn inertial frame [m]
- `r_body::AbstractArray{<:Real, 1}`: position of body in a commonn inertial frame [m]
- `GM::AbstractArray{<:Real, 1}`: gravitational coeffient of attracting body [m^3/s^2] Default: SatelliteDynamics.GM_EARTH)
(Default: SatelliteDynamics.GM_EARTH

Return:
- `a::AbstractArray{<:Real, 1}`: Acceleration in X, Y, and Z inertial directions [m/s^2]
"""
def accel_point_mass(r_sat, r_body, gm_body=GM_EARTH):
    # Restrict inputs to position only
    r_sat  = r_sat[:,:3]
    r_body = r_body[:,:3]

    # Relative position vector of satellite w.r.t. the attraching body
    d = r_sat - r_body

    # Acceleration
    a = -gm_body * (d/d.norm()**3 + r_body/r_body.norm()**3)

    return a


"""
Given an orbital state expressed in osculating orbital elements compute the equivalent Cartesean position and velocity of the inertial state.

The osculating elements are assumed to be (in order):
1. _a_, Semi-major axis [m]
2. _e_, Eccentricity [dimensionless]
3. _i_, Inclination [rad]
4. _Ω_, Right Ascension of the Ascending Node (RAAN) [rad]
5. _ω_, Argument of Perigee [ramd]
6. _M_, Mean anomaly [rad]

Arguments:
- x_oe `x::AbstractArray{<:Real, 1}`: Osculating orbital elements. See above for desription of the elements and their required order.
- `use_degrees:Bool`: If `true` interpret input will be interpreted as being in degrees, and output will be returned in degrees.
- `GM::Real`: Gravitational constant of central body. Defaults to `SatelliteDynamics.GM_EARTH` if none is provided.

# Returns
- x `x::AbstractArray{<:Real, 1}`: Cartesean inertial state. Returns position and velocity. [m; m/s]
"""
def sOSCtoCART(x_oe, use_degrees=False, GM=GM_EARTH):

    if use_degrees == True:
        # Copy and convert input from degrees to radians if necessary
        oe = x_oe.clone().detach()
        oe[3:6] = oe[:,3:6]*np.pi/180.0
    else:
        oe = x_oe
    
    # Unpack input
    a, e, i, RAAN, omega, M = oe[:,0], oe[:,1], oe[:,2], oe[:,3], oe[:,4], oe[:,5]

    E = anomaly_mean_to_eccentric(M, e)

    # Create perifocal coordinate vectors
    P    = torch.zeros((oe.shape[0], 3))
    P[:,0] = torch.cos(omega)*torch.cos(RAAN) - torch.sin(omega)*torch.cos(i)*torch.sin(RAAN)
    P[:,1] = torch.cos(omega)*torch.sin(RAAN) + torch.sin(omega)*torch.cos(i)*torch.cos(RAAN)
    P[:,2] = torch.sin(omega)*torch.sin(i)

    Q    = torch.zeros((oe.shape[0], 3))
    Q[0] = -torch.sin(omega)*torch.cos(RAAN) - torch.cos(omega)*torch.cos(i)*torch.sin(RAAN)
    Q[1] = -torch.sin(omega)*torch.sin(RAAN) + torch.cos(omega)*torch.cos(i)*torch.cos(RAAN)
    Q[2] =  torch.cos(omega)*torch.sin(i)

    # Find 3-Dimensional Position
    x = torch.zeros((oe.shape[0], 6))
    x[:,:3] = a*(torch.cos(E)-e)*P + a*torch.sqrt(1-e*e)*torch.sin(E)*Q
    x[:,3:] = torch.sqrt(GM*a)/(x[:,:3].norm(dim=1))*(-torch.sin(E)*P + torch.sqrt(1-e*e)*torch.cos(E)*Q)

    return x


"""
Convert mean anomaly into eccentric anomaly.

Arguments:
- `M::Real`: Mean anomaly. [deg] or [deg]
- `e::Real`: Eccentricity. [dimensionless]
- `use_degrees:Bool`: If `true` interpret input will be interpreted as being in degrees, and output will be returned in degrees.

Returns:
- `E::Real`: Eccentric anomaly. [rad] or [deg]
"""
def anomaly_mean_to_eccentric(M, e, use_degrees=False):
    # Convert degree input
    if use_degrees == True:
        M *= np.pi/180.0

    # Convert mean to eccentric
    max_iter = 15
    epsilson = 1e-8#eps(Float64)*100.0

    # Initialize starting values
    M = M % 2.0*np.pi
    if e < 0.8:
        E = M
    else:
        E = M*0 + np.pi

    # Initialize working variable
    f = E - e*torch.sin(E) - M
    i = 0

    # Iterate until convergence
    while (f.abs() > epsilson).all():
        f = E - e*torch.sin(E) - M
        E = E - f / (1.0 - e*torch.cos(E))

        # Increase iteration counter
        i += 1
        if i == max_iter:
            raise Exception("Maximum number of iterations reached before convergence.")

    # Convert degree output
    if use_degrees == True:
        E *= 180.0/np.pi

    return E

"""
Compute the satellite orbital period given the semi-major axis.

Arguments:
- `a::Real`: Semi-major axis. [m]
- `GM::Real`: Gravitational constant of central body. Defaults to `SatelliteDynamics.GM_EARTH` if none is provided.

Returns:
- `T::Real`: Orbital period. [s]
"""
def orbit_period(a, GM=GM_EARTH):
    return 2.0*np.pi*np.sqrt(a**3/GM)

