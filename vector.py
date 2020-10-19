# ============================================================================
# Name: vector.py
# Author: Stefan Ernst
# Date: 2018-05-22
# Updated: 2018-07-25
# Desc: Collection of functions for processing vector data
# ============================================================================

# Types of OGR geomeries:
#           0 Geometry
#           1 Point
#           2 Line
#           3 Polygon
#           4 MultiPoint
#           5 MultiLineString
#           6 MultiPolygon
#         100 No Geometry


def add_fields(ds_lyr, names, types):
    import ogr

    for i, name in enumerate(names):
        ds_lyr.CreateField(ogr.FieldDefn(name, types[i]))


def clip(in_lyr, clip_lyr=None, clip_feat=None, outName=None):

    import ogr
    import os

    if clip_lyr is None and clip_feat is None:
        print('No clip layer or clip feature defined. Exiting')
        return

    out_ds = ogr.GetDriverByName('Memory').CreateDataSource('mem')
    out_lyr = out_ds.CreateLayer('')

    clipper = ogr.GetDriverByName('Memory').CreateDataSource('mem')
    clipper_lyr = clipper.CreateLayer('')
    defn = clipper_lyr.GetLayerDefn()
    out_feat = ogr.Feature(defn)
    if clip_feat is None:
        clip_feat = clip_lyr.GetNextFeature()
    clip_geom = clip_feat.geometry().Clone()
    clip_geom_Wkb = ogr.CreateGeometryFromWkb(clip_geom.ExportToWkb())
    out_feat.SetGeometry(clip_geom_Wkb)
    clipper_lyr.CreateFeature(out_feat)

    ogr.Layer.Clip(in_lyr, clipper_lyr, out_lyr)

    if outName is not None:
        write_vector(out_ds, outName, srs=get_srs_vec(in_lyr))

    return out_ds


def feature_proximity(src_geom, target_lyr, distance_metric, buff_incr=None):
    '''
    This function takes a geometry and a layer as input and returns the min/max/mean distance between the geometry and
    the features in the target layer.
    When using distance_metric min, it is recommended to use the buff_incr variable for layers containing a large amount
    of geometries. The function will buffer the source geometry incrementally by the distance specified, until a feature
    is found in the buffer area. This makes sure we don't compute distances for all features in the large dataset.

    :param source_geom: Geometry
    :param target_lyr: Layer containing features to check
    :param buff_incr: Distance in m of the buffer size. With each iteration the buffer size increases by this value.
    :param distance_metric: float - distance to be returned - min/max/mean
    :return: float - in units of srs of the target layer
    '''
    import ogr, osr

    if srs_equal_check(src_geom, target_lyr) is False:
        src_geom = src_geom.Clone()
        src_geom.Transform(osr.CoordinateTransformation(get_srs_vec(src_geom), get_srs_vec(target_lyr)))

    features_found = 0
    buff_distance = buff_incr

    if buff_incr is not None:
        if distance_metric == 'max':
            print('Using distance_metric=max and buff_incr does not make sense for most applications. Please check '
                  'the description of the feature_proximity function.')
        while features_found < 1:
            # create buffer around feature and use as Spatial Filter to reduce the distances to calculate
            geom_filter = src_geom.Buffer(buff_distance)
            buff_distance += buff_incr
            target_lyr.SetSpatialFilter(geom_filter)
            features_found = target_lyr.GetFeatureCount()

    # only calculate distances if there is features in our buffer
    distance_list = []

    target_feat = target_lyr.GetNextFeature()
    while target_feat:
        feature_geom = target_feat.geometry().Clone()
        distance = src_geom.Distance(feature_geom)
        if distance != 0.0:
            distance_list.append(distance)
        target_feat = target_lyr.GetNextFeature()
    min_dist = min(distance_list)
    max_dist = max(distance_list)
    mean_dist = sum(distance_list) / len(distance_list)

    target_lyr.SetSpatialFilter(None)
    target_lyr.ResetReading()

    if distance_metric == 'min':
        distance_out = min_dist
    if distance_metric == 'max':
        distance_out = max_dist
    if distance_metric == 'mean':
        distance_out = mean_dist

    return distance_out


def copy_vector(in_folder, in_lyrname, out_filename, out_driver):
    '''
    Copies a vector to
    :param in_folder:
    :param in_lyrname:
    :param out_filename:
    :param out_driver:
    :return:
    '''
    import ogr
    ds = ogr.Open(in_folder, 1)
    if ds is None:
        return 'Could not open folder.'
    in_lyr = ds.GetLayer(in_lyrname)

    out_ds = create_vector(out_filename, out_driver)
    out_lyr = out_ds.CreateLayer(out_filename, in_lyr.GetSpatialRef(), ogr.wkbPolygon)
    out_lyr.CreateFields(in_lyr.schema)
    out_defn = out_lyr.GetLayerDefn()
    out_feat = ogr.Feature(out_defn)
    for in_feat in in_lyr:
        geom = in_feat.geometry()
        out_feat.SetGeometry(geom)
        for i in range(in_feat.GetFieldCount()):
            value = in_feat.GetField(i)
            out_feat.SetField(i, value)
        out_lyr.CreateFeature(out_feat)
    out_feat = None

    del ds, out_ds


def create_srs(epsgcode):
    '''
    Create SRS from EPSGCode
    :param epsgcode: int - EPSG code of SRS to be created
    :return: osr.SpatialReference object
    '''
    import osr

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(epsgcode)

    return out_srs


def create_vector(filename, driver='ESRI Shapefile'):
    '''
    Creates an ogr.DataSource
    :param filename: string - path to file or folder
    :param driver: ogr.Driver - optional
    :return: ogr.DataSource
    '''
    import ogr

    driver = ogr.GetDriverByName(driver)
    out_ds = driver.CreateDataSource(filename)
    if out_ds is None:
        print('Could not create {}'.format(out_ds))
        return

    return out_ds


def get_srs_vec(inputVector, EPSG=False):
    '''
    Get the Spatial Reference from your vector data
    :param inputVector: str(path to ogr.DataSource), ogr.DataSouce, ogr.Layer, ogr.Geometry - object to query
    :param EPSG: optional, int - should an EPSG code be returned? (handle with care)
    :return: osr.SpatialReference
    '''
    import ogr

    if isinstance(inputVector, str):
        ds = ogr.Open(inputVector)
        dsLyr = ds.GetLayer()
        srs = dsLyr.GetSpatialRef()
    elif isinstance(inputVector, ogr.DataSource):
        dsLyr = inputVector.GetLayer()
        srs = dsLyr.GetSpatialRef()
    elif isinstance(inputVector, ogr.Layer):
        srs = inputVector.GetSpatialRef()
    elif isinstance(inputVector, ogr.Feature):
        srs = inputVector.geometry.GetSpatialReference()
    elif isinstance(inputVector, ogr.Geometry):
        srs = inputVector.GetSpatialReference()
    else:
        print("Error: Input needs to be of class string, ogr.DataSource, ogr.Layer or ogr.Geometry.")

    if EPSG is True:
        srs = int(srs.GetAttrValue('Authority', 1))

    return srs


def get_attributes(ds, field, unique=False):
    '''
    Get all attributes from a field of a layer
    :param ds: ogr.DataSource
    :param field: str - name of the field to query
    :param unique: bool - only return unique attributes
    :return: list - all values/unique values of the queried field
    '''
    lyr = ds.GetLayer()
    ft = lyr.GetNextFeature()
    f_values = []
    while ft:
        f_values.append(ft.GetField(field))
        ft = lyr.GetNextFeature()
    if unique is True:
        f_values = sorted(set(f_values))
    lyr.ResetReading()

    return f_values


def get_fields(layer):
    '''
    Get all fields from a vector file
    :param layer: ogr.DataSource
    :return: list
    '''
    all_fields = [layer.GetLayerDefn().GetFieldDefn(i).name for i in range(layer.GetLayerDefn().GetFieldCount())]
    return all_fields


def polygon_from_UL_LR(xMin, xMax, yMin, yMax):
    '''
    Takes xMin/xMax and yMin/yMax to create a polygon geometry.

    :params xMin, xMax, yMin, yMay: int/float - minimum and maximum x/y coordinates
    :return: ogr.wkbPolygon geometry
    '''
    import ogr

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xMin, yMax)
    ring.AddPoint(xMax, yMax)
    ring.AddPoint(xMax, yMin)
    ring.AddPoint(xMin, yMin)
    box = ogr.Geometry(ogr.wkbPolygon)
    box.AddGeometry(ring)
    box.CloseRings()

    return box


def reproject_point(geom, target_obj):
    '''
    Reproject a point geometry
    :param geom: ogr.Geometry - geometry to reproject
    :param target_obj: ogr.DataSource, ogr.Layer, ogr.Feature, ogr.Geometry, gdal.Dataset
    :return: ogr.Geometry in srs of target_obj
    '''
    import ogr, osr
    import gdal
    from geotools.raster import get_srs_ras

    src_prj = get_srs_vec(geom)

    if isinstance(target_obj, (ogr.DataSource, ogr.Layer, ogr.Geometry)):
        target_prj = get_srs_vec(target_obj)
    elif isinstance(target_obj, gdal.Dataset):
        target_prj = get_srs_ras(target_obj)

    coordTrans = osr.CoordinateTransformation(src_prj, target_prj)

    # transform point
    geom.Transform(coordTrans)

    return(geom)


def reproject_vector(in_vector, target_srs, out_fp='mem', driver="MEMORY"):
    '''
    Reproject a vector file from one projection to another
    :param in_vector: string or ogr.DataSource - path to source ds or ds
    :param target_srs: osr.SpatialReference - SRS to project to
    :param out_fp: optional, str - output filepath
    :param driver: ogr.Driver - driver to be used for output
    :return: ogr.DataSource if driver is MEMORY or None and file written to disk
    '''
    import ogr, osr

    # open file to reproject
    if isinstance(in_vector, str):
        in_vector = ogr.Open(in_vector)

    in_layer = in_vector.GetLayer()

    # set spatial reference and transformation
    in_srs = in_layer.GetSpatialRef()
    if isinstance(target_srs, int):
        target_srs = create_srs(target_srs)
    transform = osr.CoordinateTransformation(in_srs, target_srs)

    # create layer to copy information into
    driver = ogr.GetDriverByName(driver)
    out_ds = driver.CreateDataSource(out_fp)
    # checks which geometry type we're looking at automatically layer.GetLayerDefn().GetGeomType()
    out_layer = out_ds.CreateLayer('', target_srs, in_layer.GetLayerDefn().GetGeomType())
    in_feat = in_layer.GetFeature(0)
    # create same fields as in input
    for i in range(in_feat.GetFieldCount()):
        out_layer.CreateField(in_feat.GetFieldDefnRef(i))

    # apply transformation
    for i, in_feat in enumerate(in_layer):
        transformed = in_feat.GetGeometryRef()
        transformed.Transform(transform)

        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        defn = out_layer.GetLayerDefn()
        out_feat = ogr.Feature(defn)
        out_feat.SetGeometry(geom)
        # write information to fields
        for i in range(in_feat.GetFieldCount()):
            value = in_feat.GetField(i)
            out_feat.SetField(i, value)
        out_layer.CreateFeature(out_feat)
        out_feat = None

    if driver.GetDescription() == 'Memory':
        return out_ds

    del out_ds


def reproject_xy_coords(in_x, in_y, in_srs=None, out_srs=None, EPSG=False):
    '''
    This tool transforms coordinates from one projection to another.
    Requires EPSG codes (int) for in- and output projection

    :param in_x: numeric - x-coordinate
    :param in_y: numeric - y-coordinate
    :param inEPSG: int - EPSG code for projection of input file
    :param outEPSG: int - EPSG code for projection of output file
    :return: Tuple of reprojected coordinates
    '''
    import ogr
    import osr

    # create geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(in_x, in_y)

    if EPSG is True:
        # create transformation
        src_prj = osr.SpatialReference()
        src_prj.ImportFromEPSG(in_srs)
        target_prj = osr.SpatialReference()
        target_prj.ImportFromEPSG(out_srs)
    else:
        src_prj = in_srs
        target_prj = out_srs

    coordTrans = osr.CoordinateTransformation(src_prj, target_prj)

    # transform point
    point.Transform(coordTrans)

    out_x = point.GetX()
    out_y = point.GetY()

    return out_x, out_y


def snap_to_grid(pointX, pointY, xRef, yRef, grid_spacing, pointIsMin=True):
    '''
    Shifts the x/y coordinates of a point to a grid defined by the x/y coordinates of a reference point.

    :param pointX, pointY: float/int - x/y coordinates of point to be shifted
    :param xRef, yRef:  float/int - x/y coordinates of reference point
    :param grid_spacing: Spacing of grid (size of grid cells)
    :param pointIsMin: bool - which direction should the snapping be to?
    :return: x/y coordinates of shifted point
    '''
    import math

    if xRef-pointX > 0:
        left = True
    elif xRef-pointX < 0:
        left = False
    elif xRef-pointX == 0:
        xNew = pointX
        return xNew

    # how many pixel fit in the space between reference point and extent of bounding box
    x_diff, y_diff = (xRef-pointX) / grid_spacing, (yRef - pointY) / grid_spacing
    # find closest pixel centroid to the right of pointX
    if left is True:
        xNew, yNew = xRef - math.ceil(x_diff) * grid_spacing, yRef - math.ceil(y_diff) * grid_spacing
        # shift one pixel left/down if input is the maximum values of a bounding box
        if pointIsMin is False:
            xNew, yNew = xNew + grid_spacing, yNew + grid_spacing

    if left is False:
        xNew, yNew = xRef - math.floor(x_diff) * grid_spacing, yRef - math.floor(y_diff) * grid_spacing
        if pointIsMin is False:
            xNew, yNew = xNew - grid_spacing, yNew - grid_spacing

    return xNew, yNew


def spatial_filter_by_extent(inputDS, filterDS, datatype=None):
    '''
    Filters an input vector by the extent of a raster or vector file. Automatically detects input type.

    :param inputPath: Path of the input vector file
    :param filterDS: Path of the file used as filter
    :param datatype:
    :return: ogr.Layer filtered by extent of input dataset
    '''
    import gdal
    import ogr
    from geotools.raster import get_corner_coords
    from geotools.raster import get_srs_ras
    from geotools.vector import get_srs_vec

    src_prj = srs_vec(inputDS)  # get projection of inputDS

    if isinstance(filter, gdal.Dataset):
        filterProj = get_srs_ras(filterDS)
        rasCoords = get_corner_coords(filterDS, dict=True)
        UL_x, UL_y, LR_x, LR_y = [rasCoords[key] for key in ["UL_x", "UL_y", "LR_x", "LR_y"]]

    if isinstance(filter, ogr.DataSource):
        inputLayer = inputDS.GetLayer()
        filterProj = get_srs_vec(filterDS)
        UL_x, LR_x, LR_y, UL_y = inputLayer.GetExtent()  # minX, maxX, minY, maxY

    if src_prj.IsSame(filterProj) is not 1:
        print("Projection of input files does not match. Expect very funky results")

    inputLayer.SetSpatialFilterRect(UL_x, LR_x, LR_y, UL_y)

    filteredDS = inputDS.GetLayer()

    return filteredDS


def sql_wrapper(ds, what, where, outName=None, filter=None, dialect='SQLite'):
    '''
    This function creates a new dataset based on an SQL statement and optionally writes it to disk.
    Only accounts for simple SELECT - FROM - WHERE requests at this point (pretty useless...).

    :param ds: ogr.DataSource to perform the SQL statement on
    :param what: what should be select from ds?
    :param where: condition for selection
    :param outName: optional - filename for output file
    :param filter: optional - geometry to be used as spatial filter
    :param dialect: SQL dialect - default is SQLite
    :return: ogr.DataSource - None if outName is defined
    '''
    import ogr
    import os

    ds_dir = os.path.dirname(ds.GetName())
    if ds_dir == '':
        ds_dir = './'
    fname = os.path.splitext(os.path.basename(ds.GetName()))[0]

    ds = ogr.Open(ds_dir, 1)
    sql = 'SELECT {0} FROM {1} WHERE {2}'.format(what, fname, where)
    in_lyr = ds.ExecuteSQL(sql, spatialFilter=filter, dialect=dialect)
    if outName is not None:
        out_lyr = ds.CopyLayer(in_lyr, outName)
        out_lyr = None
        ds = None
        return
    return in_lyr


def srs_equal_check(input1, input2, reproject=False):
    import gdal
    from geotools.raster import get_srs_ras
    import ogr

    if isinstance(input1, (ogr.DataSource, ogr.Layer, ogr.Geometry)):
        input1_srs = get_srs_vec(input1)
    elif isinstance(input1, gdal.Dataset):
        input1_srs = get_srs_ras(input1)

    if isinstance(input2, (ogr.DataSource, ogr.Layer, ogr.Geometry)):
        input2_srs = get_srs_vec(input2)
    elif isinstance(input2, gdal.Dataset):
        input2_srs = get_srs_ras(input2)

    return input1_srs.IsSame(input2_srs) is 1


def write_vector(in_data, out_fp, driver="ESRI Shapefile", srs=None):
    import ogr

    out_ds = create_vector(out_fp, driver)
    if out_ds is None:
        print("Could not create file. Check if file exists and is in use.")
        return

    if srs is not None:
        if isinstance(srs, int):
            srs = create_srs(srs)
    else: srs = get_srs_vec(in_data)

    # create layer, layer definition
    if isinstance(in_data, ogr.DataSource):
        in_lyr = in_data.GetLayer()
    else: in_lyr = in_data
    geomType = in_lyr.GetGeomType() # or in_data.GetLayerDefn().GetGeomType()
    out_lyr = out_ds.CreateLayer('', srs, geomType)
    # create/copy fiels from input
    in_feat = in_lyr.GetFeature(0)
    for i in range(in_feat.GetFieldCount()):
        out_lyr.CreateField(in_feat.GetFieldDefnRef(i))
    defn = out_lyr.GetLayerDefn()

    for i in range(in_lyr.GetFeatureCount()):
        in_feat = in_lyr.GetFeature(i)
        out_feat = ogr.Feature(defn)
        for i in range(in_feat.GetFieldCount()):
            value = in_feat.GetField(i)
            out_feat.SetField(i, value)
        geometry = in_feat.geometry().Clone()
        featureWkb = geometry.ExportToWkb()
        geometry = ogr.CreateGeometryFromWkb(featureWkb)
        out_feat.SetGeometry(geometry)
        out_lyr.CreateFeature(out_feat)

    #experimental change:
    #for feature in in_lyr:
    #    attribs = feature.items()
    #    out_feat = ogr.Feature(defn)
    #    for i, value in enumerate(attribs.values()):
    #        out_feat.SetField(i, value)
    #    geometry = in_feat.geometry().Clone()
    #    featureWkb = geometry.ExportToWkb()
    #    geometry = ogr.CreateGeometryFromWkb(featureWkb)
    #    out_feat.SetGeometry(geometry)
    #    out_lyr.CreateFeature(out_feat)


    out_feat = geometry = None  # destroy stuff
    out_ds = out_lyr = out_feat = geometry = None  # Save and close
