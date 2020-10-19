# ============================================================================
# Name: raster.py
# Author: Stefan Ernst
# Date: 2018-03-16
# Updated: 2020-10-19
# Last update: improve write_raster
# Desc: Collection of functions for raster processing
# ============================================================================


def common_extent_asarray(rasterList, pixel_size=30):
    '''
    Reads the overlapping area of a set of rasters as arrays.
    !! Pixel size MUST use same unit as input rasters !!

    :param rasterList: list of rasters
    :param pixel_size: size of pixels of input rasters
    :return: list of arrays
    '''
    import gdal

    bbox = min_bounding_box(rasterList)
    num_xpix = int((bbox[3][0] - bbox[0][0]) / pixel_size)
    num_ypix = int((bbox[0][1] - bbox[3][1]) / pixel_size)

    ras_comm = [img.ReadAsArray(*index_from_coords(img, bbox[0]), num_xpix, num_ypix) for img in rasterList]

    return(ras_comm)


def create_raster(xSize, ySize, gt, srs, driver='Mem', out_fp='', bands=1, dtype=None):
    '''
    Creates a raster dataset in memory or on disk
    :param xSize: int - number of pixels in x-direction
    :param ySize: int - number of pixels in y-direction
    :param gt: tuple - GeoTransform in the format (origin x, pixel resolution w-e-direction, rotation,
                                                   origin y, rotation, pixel resolution n-s-direction)
    :param srs: osr.SpatialReference
    :param driver: optional - gdal.Driver, memory by default
    :param out_fp: str - filename of the dataset to be created
    :param bands: optional - int - number of bands of dataset
    :param dtype: optional - int/gdal naming - data type of raster values, gdal.GDT_Byte by default
    :return: gdal.Dataset
    '''
    import gdal

    if dtype is None:
        dtype=gdal.GDT_Byte
    ds = gdal.GetDriverByName(driver).Create(out_fp, xSize, ySize, bands, dtype)
    ds.SetGeoTransform(gt)
    ds.SetProjection(srs)

    return ds


def driver_selection(out_fn, driver_name):
    '''
    Small tool to select drivers from GDAL by file extension given in out_fn
    :param out_fn: filename of raster to be created
    :param driver_name: GDAL driver name
    :return: gdal.Driver
    '''
    import gdal

    if driver_name == None:
        if out_fn.endswith('.tif'):
            driver = gdal.GetDriverByName('GTiff')
        elif out_fn.endswith('.envi'):
            driver = gdal.GetDriverByName('ENVI')
        elif out_fn.endswith('.vrt'):
            driver = gdal.GetDriverByName('VRT')
        else:
            print('No driver specified or detected in filename. Please specify driver_name.'
                  '\n Refer to http://www.gdal.org/formats_list.html for a list of drivers supported by GDAL.')
    else:
        driver = gdal.GetDriverByName(driver_name)

    return driver


def extract_value(ras, point):
    '''
    Well...not much magic here
    :param ras:
    :param point:
    :return:
    '''
    import gdal

    x, y = point.GetX(), point.GetY()

    if isinstance(ras, list):
        values = []
        for file in ras:
            xoff, yoff = index_from_coords(file, (x, y))
            values.extend(file.ReadAsArray(xoff, yoff, 1, 1).flatten().tolist())
    elif isinstance(ras, gdal.Dataset):
        xoff, yoff = index_from_coords(ras, (x, y))
        values = ras.ReadAsArray(xoff, yoff, 1, 1).flatten().tolist()
    else:
        print("Argument features is not of type gdal.Dataset or List (of gdal.Datasets). Returning None.")
        return

    return values


def get_corner_coords(inputRaster, epsgCode=None, dict=False):
    '''
    This tool calculates the corner coordinate pairs (x,y) for a raster data set and does a
    re-projection if the epsgCode var is set (set to 'None' if no re-projection should be done).

    :param inputRaster: Input raster file
    :param epsgCode: EPSG string, used for re-projection
    :return: Coordinates of upper left, upper right, lower left, lower right. Tuple of tuples
    '''
    import gdal
    from geotools.vector import reproject_xy_coords

    # get upper left coords, resolution + x/y direction from input raster
    if isinstance(inputRaster, str):
        ras = gdal.Open(inputRaster)
    elif isinstance(inputRaster, gdal.Dataset):
        ras = inputRaster
    rasGT = ras.GetGeoTransform()
    UL_x, UL_y, res_x, res_y = rasGT[0], rasGT[3], rasGT[1], rasGT[5]

    # get length of rows and columns
    xSize = ras.RasterXSize
    ySize = ras.RasterYSize

    # calculate lower right coordinates
    LR_x = UL_x + (xSize * res_x)
    LR_y = UL_y + (ySize * res_y)

    if epsgCode == None:
        # create tuple of tuples containing x and y coordinates for each corner
        corners = ((UL_x, UL_y), (LR_x, UL_y), (UL_x, LR_y), (LR_x, LR_y))

    # re-project if EPSG_code is set
    elif isinstance(epsgCode, int):
        inEPSG = int(gdal.Info(ras, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
        UL_x, UL_y = reproject_xy_coords(UL_x, UL_y, inEPSG, epsgCode)
        LR_x, LR_y = reproject_xy_coords(LR_x, LR_y, inEPSG, epsgCode)
        corners = ((UL_x, UL_y), (LR_x, UL_y), (UL_x, LR_y), (LR_x, LR_y))

    else:
        print("epsgCode must be integer. See http://spatialreference.org/ref/epsg/ or http://epsg.io/")
        return

    if dict is True:
        corners = {"UL_x": UL_x, "UL_y": UL_y,
                   "UR_x": LR_x, "UR_y": UL_y,
                   "LL_x": UL_x, "LL_y": LR_y,
                   "LR_x": LR_x, "LR_y": LR_y}

    return corners


def get_image_header(inputRaster):
    '''
    Get basic info from raster file.
    '''
    import gdal

    if isinstance(inputRaster, str):
        inputRaster = gdal.Open(inputRaster, gdal.GA_ReadOnly)

    if inputRaster is not None:

        print(inputRaster)
        print("This image has", inputRaster.RasterCount, "bands.")

        for i in range(inputRaster.RasterCount):
            print("Band {0}: {1}, no data value: {2}".format(i+1, inputRaster.GetRasterBand(i+1).GetDescription(),
                                                             inputRaster.GetRasterBand(i+1).GetNoDataValue()))

        print("Image Size [{0}, {1}]".format(inputRaster.RasterXSize, inputRaster.RasterYSize))

        rasGT = inputRaster.GetGeoTransform()
        if rasGT is not None:
            print("Origin = ({0}, {1})".format(rasGT[0], rasGT[3]))
            print("Pixel size = ({0}, {1})".format(rasGT[1], rasGT[5]))
            print("Rotation = ({0}, {1})".format(rasGT[2], rasGT[4]))
        else:
            print("No geotransform detected for {}".format(inputRaster))

        srs = get_srs_ras(inputRaster)
        if srs is not None:
            print("Spatial referenfe: EPSG:{}".format(get_srs_ras(inputRaster, EPSG=True)))
            print(srs)

    else:
        print("Could not open input raster {}".format(inputRaster))


def get_map_proportions(in_arr, drop_vals=False):
    '''
    Get proportions of values in array - area proportions for maps

    :param in_arr: input array
    :param drop_vals: list - values to ignore when calculating. Total area will be reduced by the area of respective
                        classes. Example usecase: analysing classification results where water is masked.
    :return: list of lists: classes and proportions
    '''
    import numpy as np

    classes, count = np.unique(in_arr, return_counts=True)
    classes, count = classes.tolist(), count.tolist()
    arraySize = sum(count)

    if drop_vals:
        for drop in drop_vals:
            drop_index = classes.index(drop)  # get index of value to drop
            classes.remove(drop)
            arraySize = arraySize - count[drop_index]  # get update size of array without that value
            count.pop(drop_index)

    classes_proportions = count/arraySize

    results = [classes, classes_proportions.tolist()]

    return results


def get_netCDF_bands(filepath, ignore_bands=None):
    import h5py

    file = h5py.File(filepath)
    bands = [x for x in file]
    if ignore_bands is not None:
        bands = set(bands)
        ignore = set(ignore_bands)
        bands = list(bands.difference(ignore))

    return bands

def get_srs_ras(inputRaster, EPSG=False):
    '''
    Get the Spatial Reference from your raster data

    :param inputRaster: str(path to gdal.Dataset) or ogr.Dataset - object to query
    :param EPSG: optional, int - should an EPSG code be returned? (handle with care)
    :return: osr.SpatialReference
    '''
    import gdal, osr

    if type(inputRaster) == str:
        inputRaster = gdal.Open(inputRaster)
    proj = inputRaster.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)

    if EPSG is True:
        srs = srs.GetAttrValue('Authority', 1)

    return srs


def index_from_coords(inputRaster, coords, inverse=False):
    '''
    Calculates the raster index (row, column) from coordinates.

    :param inputRaster: Input raster
    :param coords: tuple: coordinates as x-y
    :return: Row and column
    '''
    import gdal

    if isinstance(inputRaster, str):
        inputRaster = gdal.Open(inputRaster)

    rasGT = inputRaster.GetGeoTransform()
    UL_x, UL_y, res_x, res_y = rasGT[0], rasGT[3], rasGT[1], rasGT[5]

    x_index = int((coords[0] - UL_x) / res_x)
    y_index = int((UL_y - coords[1]) / abs(res_y))

    if inverse is True:
        x_coord = UL_x + (coords[0] * res_x) + (res_x / 2)
        y_coord = UL_y + (coords[1] * res_y) + (res_y / 2)
        return x_coord, y_coord

    return x_index, y_index


def make_masked_array(inputArray, nodatavalue):
    '''
    Creates a numpy.maskedArray from numpy array.

    :param inputArray: numpy array
    :param nodatavalue: list or single value - value(s) to be masked
    :return: numpy masked array
    '''
    import numpy.ma as ma
    import numpy as np

    if isinstance(nodatavalue, list):
        maskedArray = ma.masked_array(inputArray, np.logical_or.reduce([inputArray == value for value in nodatavalue]))
    else:
        nodata = inputArray == nodatavalue
        maskedArray = ma.masked_array(inputArray, nodata)

    return maskedArray


def make_vrt(in_folder, patterns, out_name, opts = None):
    import gdal
    import glob
    from geotools.utils import glob_multipattern

    files = glob_multipattern(in_folder, patterns)
    vrt = gdal.BuildVRT(out_name, files, separate = False)
    vrt = None

    return None


def min_bounding_box(in_files, minmax=False):
    '''
    Calculates the minimum bounding box of input raster files.

    :param in_files: list - List of raster file paths (full path required).
    :return: Coordinates minimum bounding box of input rasters:
    Tuple of tuples (corners clockwise from UL) or tuple of minX, maxX, minY, maxY
    '''
    UL_x_list = []
    UL_y_list = []
    LR_x_list = []
    LR_y_list = []

    for file in in_files:
        corners = get_corner_coords(file)
        UL_x_list.append(corners[0][0])
        UL_y_list.append(corners[0][1])
        LR_x_list.append(corners[3][0])
        LR_y_list.append(corners[3][1])

    UL_x = max(UL_x_list)
    UL_y = min(UL_y_list)
    LR_x = min(LR_x_list)
    LR_y = max(LR_y_list)

    # UL, UR, LL, LR
    coords = ((UL_x, UL_y), (LR_x, UL_y), (UL_x, LR_y), (LR_x, LR_y))

    # return xmin/xmax, ymin/ymax only
    if minmax is True:
        return(UL_x, LR_x, LR_y, UL_y)

    return coords


def open_netCDF(filepath, band):
    import h5py
    import gdal

    expr = '{0}"{1}":{2}'.format('NETCDF:', filepath, band)
    open = gdal.Open(expr)
    return open


def rand_sample_array(in_arr, n_samples, return_vals=True):
    '''
    Draw random indices from array.

    :param in_arr: array to draw samples from
    :param n_samples: number of samples
    :param return_vals: shall values at sample locations be returned?
    :return: list of list with row, col pairs, values at locations if return_vals=True
    '''
    import numpy as np

    # draw random numbers within the range of rows and columns of the input array
    sample_row = np.random.choice(in_arr.shape[0], n_samples, replace=True)
    sample_col = np.random.choice(in_arr.shape[1], n_samples, replace=True)
    # create list of lists with row/col pairs
    samples = []
    for row, col in zip(sample_row, sample_col):
        samples.append([row,col])

    if return_vals is True:
        values = []
        for i in range(len(samples)):
            values.append(in_arr[samples[i][0], samples[i][1]])
        return samples, values

    return samples


def rand_sample_array_stratified(in_arr, n_samples, strata=None, return_values=False, by_area=False, drop_vals=False):
    '''
    Draws a stratified random sample from the input array.

    :param in_arr: 2d np.array
    :param strata: list of lists: minimum and maximum for each stratum, e.g. [[0, 50], [51, 100]]
    :param n_samples: list: number of samples for each stratum, e.g. [50, 80] - or number of samples if by_area=True
    :param return_values: bool: values of the input array at sample locations will be returned as list
    :return: list of lists: one list per stratum, array of row/col index for each sample
    '''
    import numpy as np
    from math import ceil

    if by_area is True:
        # get strata and n_samples from map proportions if by_area is True
        classes, proportions = get_map_proportions(in_arr, drop_vals)
            # duplicate list elements to have beginning and end of strata
        classes_dupl = [x for pair in zip(classes, classes) for x in pair]
        strata = []
        for i in range(0, len(classes)):
            strata.append(classes_dupl [0:2])
            classes_dupl = classes_dupl[2:]
        n_samples = [ceil(n_samples*prop) for prop in proportions]

    # set up return variables
    samples_allstrata = []
    values_allstrata = []

    # loop through each stratum
    for i, stratum in enumerate(strata):
        print(i, stratum)

        # create array that contains the row/col index pairs of all values that are within the current stratum
        strat_rows, strat_cols = np.where(np.logical_and(in_arr >= stratum[0], in_arr <= stratum[1]))
        sample_bowl = []
        for row, col in zip(strat_rows, strat_cols):
            sample_bowl.append([row, col])
        sample_bowl = np.column_stack((strat_rows, strat_cols))

        # draw random indices so select from the index pair array
        n_to_draw = n_samples[i]
        # break if number of values within the stratum is lower than n_samples
        if len(strat_rows) < n_to_draw:
            print("Warning: Sample size ({0}) is larger than the occurrence of values ({1}) within "
                  "stratum {2}({3}).".format(n_to_draw, len(strat_rows), i, stratum))
        random_draw = []
        while len(random_draw) < n_samples[i]:
            # use np.unique to avoid sampling same pixel several times, re-draw until we have number of samples we need
            random_draw.extend(np.unique(np.random.choice(sample_bowl.shape[0], n_to_draw, replace=True)).tolist())
            random_draw = np.unique(random_draw).tolist()
            if len(random_draw) == len(strat_rows): break
            n_to_draw = n_samples[i] - len(random_draw)

        samples = sample_bowl[random_draw]
        if return_values is True:
            values = [in_arr[sample[0], sample[1]] for sample in samples]

        samples_allstrata.append(samples)
        if return_values is True:
            values_allstrata.append(values)

    if return_values is True:
        return samples_allstrata, values_allstrata
    return samples_allstrata


def reproject_raster(inputRaster, outFilename, srs_or_refRas=None, px_size=None, driver='GTiff', resampling=None):
    '''
    Raster reprojection. Use Spatial Reference object and provide pixel size or reference raster. If reference raster is
    provided, the output raster will have exactly the same GT.

    :param inputRaster: gdal.Datasource
    :param outFilename: filename of reprojected raster
    :param srs_or_refRas: osr.SpatialReference or gdal.Datasource
    :param px_size: int - Pixel Size in unit of target projection
    :param driver: char
    :param resampling: gdal.GRA resampling method
    :return: None - writes raster dataset to disk
    '''
    import gdal
    from geotools.vector import reproject_xy_coords
    import osr

    if resampling is None:
        resampling = gdal.GRA_Bilinear

    in_srs = get_srs_ras(inputRaster)
    in_gt = inputRaster.GetGeoTransform()

    # use reference gdal.Dataset to extract SRS, GT, pixel size - output extent will match reference
    if isinstance(srs_or_refRas, gdal.Dataset):
        refRas = srs_or_refRas
        out_srs = get_srs_ras(refRas)
        out_gt = refRas.GetGeoTransform()
        xSize = refRas.RasterXSize
        ySize = refRas.RasterYSize

    # use spatial reference and reproject whole image - create GeoTransform values and x-/y-size for reprojected raster
    if isinstance(srs_or_refRas, osr.SpatialReference):
        out_srs = srs_or_refRas
        in_gt = inputRaster.GetGeoTransform()
        ul_x, ul_y = reproject_xy_coords(in_gt[0], in_gt[3], in_srs, out_srs)
        lr_x, lr_y = reproject_xy_coords(in_gt[0] + in_gt[1] * inputRaster.RasterXSize,
                                         in_gt[3] + in_gt[5] * inputRaster.RasterYSize,
                                         in_srs, out_srs)
        out_gt = (ul_x, px_size, in_gt[2], ul_y, in_gt[4], -px_size)
        xSize = int((lr_x - ul_x) / px_size)
        ySize = int((ul_y - lr_y) / px_size)

    in_dtype = inputRaster.GetRasterBand(1).DataType
    n_bands = inputRaster.RasterCount
    out_ras = gdal.GetDriverByName(driver).Create(outFilename, xSize, ySize, n_bands, in_dtype)
    out_ras.SetGeoTransform(out_gt)
    out_ras.SetProjection(out_srs.ExportToWkt())

    gdal.ReprojectImage(inputRaster, out_ras, in_srs.ExportToWkt(), out_srs.ExportToWkt(), resampling)

    del out_ras
    return gdal.Open(outFilename)



def write_raster(templateRaster, outputFilename, data, data_type, driver_name='GTiff', bandnames=None, gt=None, xsize=None, ysize=None, nodatavalue=None, co=None):
    '''
    This function creates raster files from a numpy arrays.
    :param templateRaster: datasource to copy projection and geotransform from
    :param outputFilename: path to the file to create
    :param data: NumPy array containing data to write
    :param data_type: output data type
    :param driver_name: GDAL driver name
    :param gt: GeoTransform, inherited from templateras if not specified
    :param xsize: inherited from templateras if not specified
    :param ysize: inherited from templateras if not specified
    :param nodatavalue: optional NoData value
    :param co: gdal creation options (compression, bigtiff, interleave, blocksize, etc):
               https://gdal.org/drivers/raster/gtiff.html
    '''

    import gdal
    
    gdal_dt_from_np = {
      'bool': 1,
      'uint8': 1,
      'int8': 1,
      'uint16': 2,
      'int16': 3,
      'uint32': 4,
      'int32': 5,
      'int64': 6,
      'float32': 6,
      'float64': 7,
      'complex64': 10,
      'complex128': 11,
    }

    if data_type is None:
        data_type = gdal_dt_from_np[str(data.dtype)]

    if driver_name is None:
        if outputFilename.endswith('.tif'):
            driver = gdal.GetDriverByName('GTiff')
        elif outputFilename.endswith('.envi'):
            driver = gdal.GetDriverByName('ENVI')
        elif outputFilename.endswith('.vrt'):
            driver = gdal.GetDriverByName('VRT')
        else:
            print('No driver specified or detected in filename. Please specify driver_name.'
                  '\n Refer to http://www.gdal.org/formats_list.html for a list of drivers supported by GDAL.')
    else:
        driver = gdal.GetDriverByName(driver_name)

    if xsize is None:
        xsize = templateRaster.RasterXSize
    if ysize is None:
        ysize = templateRaster.RasterYSize

    coptions = ['COMPRESS=LZW', 'PREDICTOR=2', 'INTERLEAVE=BAND']
    if co:
        coptions = co

    out_ras = driver.Create(outputFilename, xsize, ysize, 1, data_type, coptions)
    out_ras.SetProjection(templateRaster.GetProjection())

    if gt is None:
        out_ras.SetGeoTransform(templateRaster.GetGeoTransform())
    else:
        out_ras.SetGeoTransform(gt)


    if data.ndim > 2:
        n_bands = data.shape[0]
        for band in range(n_bands):
            out_band = out_ras.GetRasterBand(band + 1)
            if nodatavalue is not None:
                out_band.SetNoDataValue(nodatavalue)
            if bandnames:
                out_band.SetDescription(bandnames[band])
            out_band.WriteArray(data[band, :, :])
            out_band.ComputeStatistics(False)
            
    else:
        out_band = out_ras.GetRasterBand(1)
        if nodatavalue:
            out_band.SetNoDataValue(nodatavalue)
        if bandnames:
            out_band.SetDescription(bandnames)
        out_band.WriteArray(data)
        out_band.ComputeStatistics(False)
        
    #out_band.FlushCache()
    
    out_ras = out_band = None

    return



"""
def get_winsize(in_ras, coords):
    '''

    :param in_ras:
    :param coords:
    :return:
    '''

    import gdal
    ras = gdal.Open(in_ras)
    ras_gt = ras.GetGeoTransform

    x_max = int((coords[3][0] - UL_x) / res_x) - x_offset
    y_max = int((UL_y - coords[3][1]) / abs(res_y)) - y_offset
"""
