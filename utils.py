# ============================================================================
# Name: utils.py
# Author: Stefan Ernst
# Date: 2018-07-19
# Updated: 2018-07-25
# Desc: Collection of functions that make life easier - filetools, timing, ...
# ============================================================================


def csv_writer(fname, list_to_write, col_names=None, separator=","):
    '''
    Writes a list (or other iterable?) to csv file
    :param fname: str - output filename
    :param list_to_write: list - list containing stuff to write
    :param colNames: list, optional - provide second list containing column names (=first row in csv)
    :param separator: str - delimiter for columns
    :return: nada
    '''
    import csv

    with open(fname, "w", newline="") as file:
        csv.register_dialect("nerdtalk", delimiter=separator)
        writer = csv.writer(file, dialect="nerdtalk")
        if col_names is not None:
            writer.writerow(col_names)
        for i in range(len(list_to_write)):
            writer.writerow([str(x) for x in list_to_write[i]])


def find_in_other_list(target_list, source_list, search_item):
    '''
    Get element(s) from target list at index or indices where source_list matches search_item
    :param target_list: list containing values that should be returned
    :param source_list: list that should be matched against search_item
    :param search_item: value that should be searched for in source_list (str, int, whaever a list can hold)
    :return: values from target_list
    '''
    import numpy as np

    name = np.array(target_list)[np.where(np.array(source_list) == search_item)[0]].tolist()
    if len(name) < 2:
        name = name[0]
    return name


def glob_multipattern(path, patterns):
    '''
    Peform a file search for multiple patterns. Because more is better.
    :param path: str - path to search in
    :param patterns: list of patterns to search for
    :return: list of elements found
    '''
    from itertools import chain
    import glob

    if not path.endswith('/'):
        path = path + '/'

    if type(patterns) == str:
        patterns = [patterns]

    files_matched = []
    ext_lists = [glob.glob(path + pattern) for pattern in patterns]
    for list in chain(ext_lists):
        if list != []:
            files_matched.extend(list)

    return files_matched


def time_start():
    import time
    t = time.time()
    return t


def time_end(start):
    '''
    Returns time elapsed since start. Use with time_start() for incredible results.
    :param start:
    :return:
    '''
    import time
    f = time.time() - start
    h = int(f/3600)
    m = int(f/60 - h *60)
    s = int(f - int(f/60)*60)
    elapsed = '{0}h {1}m {2}s'.format(h, m, s)
    return elapsed


#def list_files(in_path, recursive=False, full_names=False, extensions=None):
#    import os
#
#
#    if recursive is True:
#        files_selected = []
#        for root, dirs, files in os.walk(in_path):
#            for ext in extensions:
#                for file in files:
#                    if file.endswith(ext):
#                        files_selected.append(os.path.join(root, file))
#
#def list_dirs(in_path, recursive=False, full_names=False, pattern=None):
#if not in_path.endswith("/"):
#    in_path = in_path+"/"
#
#from itertools import chain
#import glob
#import os
#
#roots = []
#for root, dirs, _ in os.walk(in_path):
#    roots.append(root)
#    file_list = {}
#    for folder in roots:
#        folder_content = [glob.glob(folder + "/*" + extension) for extension in extensions]
#        folder_content_sel = []
#        for i in chain(folder_content):
#            if i != []:
#                folder_content_sel.extend(i)
#        if folder_content_sel != []:
#            file_list[str.split(folder_content_sel[0], r"/")[-1]] = folder_content_sel
#
#
#        if folder_content_sel != []:
#            file_list.append(folder_content_sel)
