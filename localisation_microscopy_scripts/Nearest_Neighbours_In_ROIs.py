#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import omero.scripts as scripts
import omero.util.script_utils as script_util
from random import random
import math
from numpy import array
from omero.gateway import BlitzGateway
import omero
from omero.rtypes import *
from distance_functions import nearest_neighbour

FILE_TYPES = {'localizer':{'numColumns': 12, 'name': 'localizer', 'frame': 0, 'intensity': 1, 'z_col': None, 'psf_sigma': 2, 'headerlines': 5, 'x_col': 3, 'y_col': 4}, 
              'quickpalm':{'numColumns': 15, 'name': 'quickpalm', 'frame': 14, 'intensity': 1, 'z_col': 6, 'psf_sigma': None, 'headerlines': 1, 'x_col': 2, 'y_col': 3},
              'zeiss2d':{'numColumns': 13, 'name': 'zeiss2d', 'frame': 1, 'intensity': 7, 'z_col': None, 'psf_sigma': 6, 'headerlines': 1, 'x_col': 4, 'y_col': 5}}
PATH = os.path.join("/home/omero/OMERO.data/", "download")

def get_rectangles(conn, imageId, pix_size):
    """
        Returns a list of (x, y, width, height, zStart, zStop, tStart, tStop)
        of each rectange ROI in the image
    """

    rois = []

    roiService = conn.getRoiService()
    result = roiService.findByImage(imageId, None)

    for roi in result.rois:
        zStart = None
        zEnd = 0
        tStart = None
        tEnd = 0
        x = None
        for shape in roi.copyShapes():
            if type(shape) == omero.model.RectI:
                # check t range and z range for every rectangle
                t = shape.getTheT().getValue()
                z = shape.getTheZ().getValue()
                if tStart is None:
                    tStart = t
                if zStart is None:
                    zStart = z
                tStart = min(t, tStart)
                tEnd = max(t, tEnd)
                zStart = min(z, zStart)
                zEnd = max(z, zEnd)
                if x is None: # get x, y, width, height for first rect only
                    x = int(shape.getX().getValue())
                    y = int(shape.getY().getValue())
                    width = int(shape.getWidth().getValue())
                    height = int(shape.getHeight().getValue())
        # if we have found any rectangles at all...
        if zStart is not None:
            rois.append((x*pix_size, y*pix_size, width*pix_size, height*pix_size, zStart, zEnd, tStart, tEnd))

    return rois
    
def get_all_xycoords(all_data,xcol,ycol,pix_size):
    """
        Returns the xy coordinates from the input data in a numpy array
    """
    coords = np.zeros((all_data.shape[0],2))
    coords[:,0] = all_data[:,xcol]*pix_size #convert to nm --> camera pixel size
    coords[:,1] = all_data[:,ycol]*pix_size #convert to nm --> camera pixel size
    return coords
                    
def parse_sr_data(path,file_type,pix_size=95):
    """
        Parses all the data in the file being processed,
        and returns the xy coords in a numpy array
    """
    working_file_type = FILE_TYPES[file_type]
    headerlines = working_file_type['headerlines']
    num_columns = working_file_type['numColumns']
    xcol = working_file_type['x_col']
    ycol = working_file_type['y_col']
    s = time.time()
    try:
        data = np.genfromtxt(path,usecols=range(num_columns),skip_header=headerlines,dtype='float')
    except ValueError:
        data = np.genfromtxt(path,delimiter=',',usecols=range(num_columns),skip_header=headerlines,dtype='float')
    except:
        print 'there was a problem parsing localisation data'
        raise
    print 'reading the file took:',time.time()-s,'seconds'
    coords = get_all_xycoords(data,xcol,ycol,pix_size)
    return coords    

def download_data(ann):
    """
        Downloads the specified file to and returns the path on the server
    """ 
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    file_path = os.path.join(PATH, ann.getFile().getName())
    f = open(str(file_path), 'w')
    print "\nDownloading file to", file_path, "..."
    try:
        for chunk in ann.getFileInChunks():
            f.write(chunk)
    finally:
        f.close()
        print "File downloaded!"
    return file_path
    
def delete_downloaded_data(ann):
    file_path = os.path.join(PATH, ann.getFile().getName())
    try:
        os.remove(file_path)
    except OSError:
        pass
    
def get_coords_in_roi(all_coords,roi):
    """
        Returns the xy coordinates of the rectangular roi being processed
    """
    #convert to nm --> sr image pixel size
    xstart = roi[0]
    xstop = roi[0]+roi[2]
    ystart = roi[1]
    ystop = roi[1]+roi[3]
    return all_coords[(all_coords[:,0] > xstart) & (all_coords[:,0] < xstop)
                      & (all_coords[:,1] > ystart) & (all_coords[:,1] < ystop)]
    
def process_data(conn,image,rectangles,coords):
    """
        Calculates the neighbour distance in all the rectangular rois
    """    
    dist_bins = np.arange(0,200,1)
    num_rois = len(rectangles)
    nn_hist = np.zeros((dist_bins.shape[0]-1,num_rois))
    nn_data = []
    for i,rect in enumerate(rectangles):
        locs = get_coords_in_roi(coords,rect)
        nn = nearest_neighbour(locs)
        nn_data.append(nn)
        hist,edges = np.histogram(nn,bins=dist_bins)
        nn_hist[:,i] = hist
    return nn_data,nn_hist,edges
                            
def run_processing(conn,script_params):
    file_anns = []
    message = ""
    
    image_id = script_params['ImageID']
    image = conn.getObject("Image",image_id)
    if not image:
        message = 'Could not find specified image'
        return message
        
    file_id = script_params['FileID']
    ann = conn.getObject("Annotation",file_id)
    if not ann:
        message = 'Could not find specified annotation'
        return message
    
    #other parameters
    sr_pix_size = script_params['SR_pixel_size']
    cam_pix_size = script_params['Parent_Image_Pixel_Size']
    file_type = script_params['File_Type']
     
    path_to_ann = ann.getFile().getPath() + '/' + ann.getFile().getName()
    name,ext = os.path.splitext(path_to_ann)
    if ('txt' in ext) or ('csv' in ext):
        path_to_data = download_data(ann)
        coords = parse_sr_data(path_to_data,file_type,cam_pix_size)
        rectangles = get_rectangles(conn,image_id,sr_pix_size)
        nn_data,nn_hist,edges = process_data(conn,image,rectangles,coords)
        
        file_name = "near_neighbours_" + ann.getFile().getName()[:-4] + '.csv'
        print file_name
        try:
            f = open(file_name,'w')
            for r in range(len(nn_data)):
                row = nn_data[r]
                f.write(','.join([str(c) for c in row])+'\n')
        finally:
            f.close()

        new_file_ann, faMessage = script_util.createLinkFileAnnotation(
            conn, file_name, image, output="Wrote near neighbour csv (Excel) file",
            mimetype="text/csv", desc=None)
        if new_file_ann:
            file_anns.append(new_file_ann)

        if not file_anns:
            faMessage = "No Analysis files created. See 'Info' or 'Error' for"\
                " more details"
        elif len(file_anns) > 1:
            faMessage = "Created %s csv (Excel) files" % len(file_anns)
        message += faMessage
    else:
        message = 'file annotation must be txt or csv'
        return message
    # clean up
    delete_downloaded_data(ann)
    
    message += faMessage
    return message

def run_as_script():
    """
    The main entry point of the script, as called by the client via the scripting service, passing the required parameters.
    """

    dataTypes = [rstring('Image')]
    
    fileTypes = [k for k in FILE_TYPES.iterkeys()]

    client = scripts.client('Ripley_Lfunction.py', """This script searches an attached SR dataset for coords defined by an ROI""",

    scripts.String("Data_Type", optional=False, grouping="01",
        description="Choose source of images (only Image supported)", values=dataTypes, default="Image"),
        
    scripts.Int("ImageID", optional=False, grouping="02",
        description="ID of super resolved image to process"),
        
    scripts.Int("FileID", optional=False, grouping="03",
        description="ID of file to process"),
        
    scripts.String("File_Type", optional=False, grouping="04",
        description="Indicate the type of data being processed", values=fileTypes, default="localizer"),
        
    scripts.Int("SR_pixel_size", optional=False, grouping="05",
        description="Pixel size in super resolved image in nm"),

    scripts.Int("Parent_Image_Pixel_Size", optional=False, grouping="06",
        description="EMCCD pixel size in nm"),
        
    version = "5.0.2",
    authors = ["Daniel Matthews", "QBI"],
    institutions = ["University of Queensland"],
    contact = "d.matthews1@uq.edu.au",
    )

    try:

        # process the list of args above.
        scriptParams = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                scriptParams[key] = client.getInput(key, unwrap=True)

        print scriptParams

        # wrap client to use the Blitz Gateway
        conn = BlitzGateway(client_obj=client)

        # process images in Datasets
        message = run_processing(conn, scriptParams)
        client.setOutput("Message", rstring(message))
        
        #client.setOutput("Message", rstring("No plates created. See 'Error' or 'Info' for details"))
    finally:
        client.closeSession()

if __name__ == "__main__":
    run_as_script()
