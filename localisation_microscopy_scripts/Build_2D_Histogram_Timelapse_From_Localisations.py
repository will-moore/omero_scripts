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
from hist_data import calc_hist

FILE_TYPES = {'localizer':{'numColumns': 12, 'name': 'localizer', 'frame': 0, 'intensity': 1, 'z_col': None, 'psf_sigma': 2, 'headerlines': 5, 'x_col': 3, 'y_col': 4}, 
              'quickpalm':{'numColumns': 15, 'name': 'quickpalm', 'frame': 14, 'intensity': 1, 'z_col': 6, 'psf_sigma': None, 'headerlines': 1, 'x_col': 2, 'y_col': 3},
              'zeiss2d':{'numColumns': 13, 'name': 'zeiss2d', 'frame': 1, 'intensity': 7, 'z_col': None, 'psf_sigma': 6, 'headerlines': 1, 'x_col': 4, 'y_col': 5},
              'zeiss2d2chan':{'numColumns': 14, 'name': 'zeiss2d2chan', 'frame': 1, 'intensity': 7, 'z_col': None, 'psf_sigma': 6, 'headerlines': 1, 'x_col': 4, 'y_col': 5,'chan_col':14}}
PATH = os.path.join("/home/omero/OMERO.data/", "download")

def get_frame_indices(start,stop,duration,overlap):
    num_frames = (stop - duration)/(duration - overlap)
    start_frames = [start]
    for f in range(num_frames):
        start = start + (duration - overlap)
        start_frames.append(start)
        
    stop_frames = [sf + duration - 1 for sf in start_frames]
    return start_frames,stop_frames, num_frames      

def get_all_locs_in_chan(all_data,col,chan=0,chancol=None):
    if chancol:
        idx = np.where(all_data[:,chancol] == chan)[0]
    else:
        idx = np.ones((all_data.shape[0]),dtype=bool)
    coords = np.zeros((idx.shape[0]))
    coords[:] = all_data[idx,col]
    return coords
        
def get_all_xycoords(all_data,xcol,ycol,sizeC,chancol,pix_size):
    """
        Returns the xy coordinates from the input data in a numpy array
    """   
    coords = np.zeros((sizeC,all_data.shape[0],2))
    print coords.shape
    for c in range(sizeC):
        print get_all_locs_in_chan(all_data,xcol,c,chancol).shape
        coords[c,:,0] = get_all_locs_in_chan(all_data,xcol,c,chancol)*pix_size #convert to nm --> camera pixel size
        coords[c,:,1] = get_all_locs_in_chan(all_data,ycol,c,chancol)*pix_size #convert to nm --> camera pixel size       
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
    fcol = working_file_type['frame']
    if 'zeiss2d' in file_type:
        footerlines = 8
    else:
        footerlines = 0
    if 'zeiss2d2chan' in file_type:
        sizeC = 2
        chancol = FILE_TYPES[file_type]['chan_col']
    else:
        sizeC = 1
        chancol = None
    if 'zeiss' in file_type:
        pix_size = 1 #override pixel size when using zeiss data
                  
    s = time.time()
    print 'footerlines=',footerlines
    try:
        data = np.genfromtxt(path,usecols=range(num_columns),skip_header=headerlines,skip_footer=footerlines,dtype='float')
    except ValueError:
        data = np.genfromtxt(path,delimiter=',',usecols=range(num_columns),skip_header=headerlines,skip_footer=footerlines,dtype='float')
    except:
        print 'there was a problem parsing localisation data'
        raise
    print 'reading the file took:',time.time()-s,'seconds'
    coords = get_all_xycoords(data,xcol,ycol,sizeC,chancol,pix_size)
    frames = data[:,fcol]
    return coords,frames   

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
    
def process_data(conn,script_params,file_id,coords,nm_per_pixel,frames):
    """
        Calculates the neighbour distance in all the rectangular rois
    """    
    image_id = script_params['ImageID']
    image = conn.getObject("Image",image_id)
    if not image:
        message = 'Could not find specified image'
        return message
    
    #set parameters
    imageName = image.getName()
    name,ext = os.path.splitext(imageName)
    if 'ome' in name:
        name = name.split('.')[0]
        new_name = name + '_sr_timelapse_histogram.ome' + ext
    else:
        new_name = name + '_sr_timelapse_histogram' + ext
    parentDataset = image.getParent()
    parentProject = parentDataset.getParent()
    updateService = conn.getUpdateService()
    
    frame_width = image.getSizeX() 
    frame_height = image.getSizeY()
    sizeZ = 1
    if 'czi' in ext:
        num_frames = image.getSizeT()
    else:
        num_frames = image.getSizeZ()
    if 'zeiss2d2chan' in script_params['File_Type']:
        sizeC = 2
    else:
        sizeC = 1
        
    if script_params['Start_Frame'] == -1:
        start = 1
    else:
        start = script_params['Start_Frame']
    if script_params['Stop_Frame'] == -1:
        stop = num_frames
    else:
        stop = script_params['Stop_Frame']
    duration = script_params['Frame_Duration']
    if script_params['Overlap'] == -1:
        overlap = 0
    else:
        overlap = duration - script_params['Overlap']
    starts,stops,sizeT = get_frame_indices(start,stop,duration,overlap)
    sr_pix_size = script_params['SR_pixel_size']
    
    #calculate histogram
    binsx = (frame_width * nm_per_pixel) / sr_pix_size
    binsy = (frame_height * nm_per_pixel) / sr_pix_size
    hist_data = np.zeros((sizeC,binsy,binsx))
    hist_frames = []
    for c in range(sizeC):
        for f in range(sizeT):
            idx = np.where((frames >= starts[f]) & (frames <= stops[f]))
            coords_in_frames = coords[c,idx,:]
            hist = calc_hist('2d',coords_in_frames,binsy,binsx)
            hist_data[c,:,:] = hist
            hist_frames.append(hist_data)
        
    def plane_gen():
        for z in range(sizeZ):
            for c in range(sizeC):
                for t in range(sizeT):
                    plane = hist_frames[t]
                    yield plane[c,:,:]     
                    
    description = "Created from image:\n  Name: %s\n  File ID: %d" % (imageName, file_id)
    newImg = conn.createImageFromNumpySeq(
        plane_gen(), new_name,
        sizeZ=sizeZ, sizeC=sizeC, sizeT=sizeT,
        description=description)

    if newImg:
        iid = newImg.getId()
        print "New Image Id = %s" % iid
        # put new images in existing dataset
        dataset = None
        if parentDataset is not None and parentDataset.canLink():
            parentDataset = parentDataset._obj
        else:
            parentDataset = None
        parentProject = None    # don't add Dataset to parent.
    
        if parentDataset is None:
            link = None
            print "No dataset created or found for new images."\
                " Images will be orphans."
        else:
            dsLink = omero.model.DatasetImageLinkI()
            dsLink.parent = omero.model.DatasetI(
                parentDataset.id.val, False)
            dsLink.child = omero.model.ImageI(iid, False)
            updateService.saveObject(dsLink)
            if parentProject and parentProject.canLink():
                # and put it in the   current project
                projectLink = omero.model.ProjectDatasetLinkI()
                projectLink.parent = omero.model.ProjectI(
                    parentProject.getId(), False)
                projectLink.child = omero.model.DatasetI(
                    dataset.id.val, False)
                updateService.saveAndReturnObject(projectLink)   
        message = 'Super resolution histogram successfully created'
    else:
        message = 'Something went wrong, could not make super resolution histogram'
    return message
                   
def run_processing(conn,script_params):
    file_anns = []
    message = ""
           
    file_id = script_params['AnnotationID']
    ann = conn.getObject("Annotation",file_id)
    if not ann:
        message = 'Could not find specified annotation'
        return message
    
    #other parameters
    cam_pix_size = script_params['Parent_Image_Pixel_Size']
    file_type = script_params['File_Type']


    path_to_ann = ann.getFile().getPath() + '/' + ann.getFile().getName()
    name,ext = os.path.splitext(path_to_ann)
    if ('txt' in ext) or ('csv' in ext):
        path_to_data = download_data(ann)
        coords,frames = parse_sr_data(path_to_data,file_type,cam_pix_size)
        faMessage = process_data(conn,script_params,file_id,coords,cam_pix_size,frames)

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

    client = scripts.client('Build_2D_Histograms_From_Localisations.py', """This script creates a 2D histogram from a locaisation microscopy dataset""",

    scripts.String("Data_Type", optional=False, grouping="01",
        description="Choose source of images (only Image supported)", values=dataTypes, default="Image"),
        
    scripts.Int("ImageID", optional=False, grouping="02",
        description="ID of parent localisation microscopy movie"),
        
    scripts.Int("AnnotationID", optional=False, grouping="03",
        description="ID of file to process"),
        
    scripts.String("File_Type", optional=False, grouping="04",
        description="Indicate the type of data being processed", values=fileTypes, default="zeiss2d"),
        
    scripts.Int("SR_pixel_size", optional=False, grouping="05",
        description="Pixel size in super resolved image in nm"),
                            
    scripts.Int("Parent_Image_Pixel_Size", optional=False, grouping="06",
        description="Required to calculate number of xy bins in histogram"),
                            
    scripts.Int("Start_Frame", optional=False, grouping="07.1",
        description="starting frame in raw data (-1 for first frame)",default="-1"),                            

    scripts.Int("Stop_Frame", optional=False, grouping="07.2",
        description="stopping frame in raw data (-1 for final frame)",default="-1"), 
                            
    scripts.Int("Frame_Duration", optional=False, grouping="07.3",
        description="Number of raw frames used in each histogram frame",default="2000"),
                            
    scripts.Int("Overlap", optional=False, grouping="07.4",
        description="Number of frames to overlap for sliding average (referenced from frame start)",default="-1"),  
        
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
