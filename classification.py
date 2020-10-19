# ============================================================================
# Name: classification.py
# Author: Stefan Ernst
# Date: 2018-07-19
# Updated: 2018-07-25
# Desc: Collection of functions for preparing data for classification with
#       scikit-learn
# ============================================================================


def extract_features(features, samples_ds, label_field):
    '''
    Use input features, ground truth and labels to create the np arrays in the format needed for scikit-learn.
    :param features: gdal.Dataset or list of gdal.Datasets - containing the features to extract
    :param samples_ds: ogr.DataSource - Shapefile etc containing the samples and labels
    :param label_field: str - name of the field in samples_ds containing the labels
    :return: two np.ndarrays - one for labels, one for extracted features
    '''

    import gdal
    from geotools.raster import extract_value
    import numpy as np

    # ==================================
    # get number of features
    if isinstance(features, list):
        n_features = sum([f.RasterCount for f in features])  # sum up the number of bands of input files
    elif isinstance(features, gdal.Dataset):
        n_features = features.RasterCount
    else:
        print("Argument features is not of type gdal.Dataset or List (of gdal.Datasets). Returning nada.")
        return

    samples_lyr = samples_ds.GetLayer()
    n_samples = samples_lyr.GetFeatureCount()  # get number of samples (=number of points in shapefile)

    # Set up the arrays for training features and labels
    train_feat = np.zeros((n_samples, n_features), dtype=np.float32)
    train_label = np.zeros((n_samples, 1), dtype=np.int8)

    # Iterate through all samples and write the labels + features to the arrays
    row = 0  # initiate row counter for the arrays
    samples_feat = samples_lyr.GetNextFeature()
    while samples_feat:

        point = samples_feat.geometry().Clone()
        # get labels
        train_label[row] = samples_feat.GetField(label_field)
        # get features
        feature_vals = extract_value(features, point)
        train_feat[row] = feature_vals

        row += 1
        samples_feat = samples_lyr.GetNextFeature()
    samples_lyr.ResetReading()

    train_label = train_label.ravel()

    return train_feat, train_label.ravel()


def train_rf(train_feat, train_label):

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(train_feat, train_label)

    return model

def predict_rf(data, model):

    from sklearn.ensemble import RandomForestClassifier

    prediction = model.predict(data)
