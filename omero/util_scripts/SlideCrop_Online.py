#!/usr/bin/env python
# -*- coding: utf-8 -*-
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders
from email.Utils import formatdate
import smtplib
import re
import numpy as np
import math
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.gateway import ImageWrapper
from omero.rtypes import rstring, rlong, robject, rint
from omero.model import *
import omero.util.script_utils as script_utils
from omero.util.temp_files import create_path,remove_path
from omero.util.tiles import * 
from ome_tiff_utils.ome_metadata import OMETIFF
from ome_tiff_utils.slide_metadata import SlideImage
from PIL import Image, ImageSequence
import logging
logger = logging.getLogger(__name__)
import os

import time
startTime = 0

ADMIN_EMAIL = 'admin@omerocloud.qbi.uq.edu.au'
PATH = os.path.join("/home/omero/OMERO.data/", "download")
    
def printDuration(output=True):
    global startTime
    if startTime == 0:
        startTime = time.time()
    if output:
        print "Script timer = %s secs" % (time.time() - startTime)
        
def delete_roi_image(filepath):
    try:
        os.remove(filepath)
    except OSError:
        pass        

class PILTiffTileReader:
 
    def __init__(self, imageName):
        self.im = Image.open(imageName)
        self.seq = ImageSequence.Iterator(self.im)
        self.margin = 1
        
    def getTile(self, channel, tileX, tileY, tileWidth, tileHeight):
        box = (tileX, tileY, tileX + tileWidth, tileY + tileHeight)
        return self.seq[channel].crop(box)
    
    def close(self):
        del self.im
        
def create_image_from_tiles(conn,source,image_name,image_path,description,box):
        
    pixelsService = conn.getPixelsService()
    queryService = conn.getQueryService()
    xbox, ybox, wbox, hbox, z1box, z2box, t1box, t2box = box
    sizeX = wbox
    sizeY = hbox
    sizeZ = source.getSizeZ()
    sizeT = source.getSizeT()
    sizeC = source.getSizeC()
    tileWidth = 512
    tileHeight = 512
           
    def create_image():
        query = "from PixelsType as p where p.value='uint8'"
        pixelsType = queryService.findByQuery(query, None)
        channelList = range(sizeC)
        bytesPerPixel = pixelsType.bitSize.val / 8
        iId = pixelsService.createImage(
           sizeX,
           sizeY,
           sizeZ,
           sizeT,
           channelList,
           pixelsType,
           image_name,
           description,
           conn.SERVICE_OPTS)
        
        image = conn.getObject("Image", iId)
        return image

    def mktile(c, x, y, w, h):
        tile = tile_source.getTile(c, x, y, w, h)
        return list(np.asanyarray(tile).flatten())
        
    class Iteration(TileLoopIteration):

        def run(self, data, z, c, t, x, y, tileWidth, tileHeight, tileCount):
            tile2d = mktile(c, x, y, tileWidth, tileHeight)
            data.setTile(tile2d, z, c, t, x, y, tileWidth, tileHeight)
                  
    tile_source = PILTiffTileReader(image_path)
    new_image = create_image()
    pid = new_image.getPixelsId()
    loop = RPSTileLoop(conn.c.sf, PixelsI(pid, False))    
    loop.forEachTile(tileWidth, tileHeight, Iteration())

    for theC in range(sizeC):
        pixelsService.setChannelGlobalMinMax(pid, theC, float(0), float(255), conn.SERVICE_OPTS)
    tile_source.close()
    return new_image
    
def getRectangles(conn, imageId):
    """
    Returns a list of (x, y, width, height, zStart, zStop, tStart, tStop)
    of each rectangle ROI in the image
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
                if x is None:   # get x, y, width, height for first rect only
                    x = int(shape.getX().getValue())
                    y = int(shape.getY().getValue())
                    width = int(shape.getWidth().getValue())
                    height = int(shape.getHeight().getValue())
        # if we have found any rectangles at all...
        if zStart is not None:
            rois.append((x, y, width, height, zStart, zEnd, tStart, tEnd))

    return rois
    
def process_image(conn, imageId, parameterMap):
    """
    Process an image.
    Create a 5D image representing the ROI "cropping" the
    original image
    Image is put in a dataset if specified.
    """

    image = conn.getObject("Image", imageId)
    if image is None:
        print "No image found for ID: %s" % imageId
        return

    parentDataset = image.getParent()
    parentProject = parentDataset.getParent()

    imageName = image.getName()
    updateService = conn.getUpdateService()
    print image.countImportedImageFiles()
    # specifically count Fileset files
    fileCount = image.countFilesetFiles()
    fileset = image.getFileset() # will be None for pre-FS images
    fsId = fileset.getId()
    # list files
    if fileCount > 0:
        for origFile in image.getImportedImageFiles():
            name = origFile.getName()
            path = '/home/omero/OMERO.data/ManagedRepository/' + origFile.getPath()
    slide_name,slide_ext = os.path.splitext(name)        
    slide_path = os.path.abspath(path+name)
    print slide_path 
    slide = SlideImage(slide_path)

    # x, y, w, h, zStart, zEnd, tStart, tEnd
    rois = getRectangles(conn, imageId)

    imgW = image.getSizeX()
    imgH = image.getSizeY()
    
    for index, r in enumerate(rois):
        x, y, w, h, z1, z2, t1, t2 = r
        # Bounding box
        X = max(x, 0)
        Y = max(y, 0)
        X2 = min(x + w, imgW)
        Y2 = min(y + h, imgH)

        W = X2 - X
        H = Y2 - Y
        if (x, y, w, h) != (X, Y, W, H):
            print "\nCropping ROI (x, y, w, h) %s to be within image."\
                " New ROI: %s" % ((x, y, w, h), (X, Y, W, H))
            rois[index] = (X, Y, W, H, z1, z2, t1, t2)

    print "rois"
    print rois
    
    if len(rois) == 0:
        print "No rectangular ROIs found for image ID: %s" % imageId
        return

#make a new 5D image per ROI
    images = []
    iIds = []
    
    for i,r in enumerate(rois):
        
        x, y, w, h, z1, z2, t1, t2 = r
        print "  ROI x: %s y: %s w: %s h: %s z1: %s z2: %s t1: %s t2: %s"\
            % (x, y, w, h, z1, z2, t1, t2)
            
        description = "Image from ROIS on parent Image:\n  Name: %s\n"\
            "  Image ID: %d" % (imageName, imageId)
        print description
 
        if (i + 1) < 10:
            new_image_name = slide_name + '_' + '00' + str(i + 1) + '.ome.tif'
            outfilepath = PATH + '/' + new_image_name
        elif (i + 1) >= 10:
            new_image_name = slide_name + '_' + '0' + str(i + 1) + '.ome.tif'
            outfilepath = PATH + '/' + new_image_name
        print 'new image path', outfilepath
        s = time.time()
        OMETIFF(None, name, outfilepath, imageId, len(rois), slide, r, 0).process()
        print 'pylibtiff took:',time.time()-s,'seconds'
        p = time.time()
        newImg = create_image_from_tiles(conn,image,new_image_name,outfilepath,description,r)
        print 'tile uploading creation took:',time.time()-p,'seconds'
        delete_roi_image(outfilepath)
        images.append(newImg)
        iIds.append(newImg.getId())
        
    if slide:
        slide.close_slide()
        
    if len(iIds) == 0:
        print "No new images created."
        return

    if 'Container_Name' in parameterMap and \
       len(parameterMap['Container_Name'].strip()) > 0:
        # create a new dataset for new images
        datasetName = parameterMap['Container_Name']
        dataset = omero.model.DatasetI()
        dataset.name = rstring(datasetName)
        desc = "Images in this Dataset are from ROIs of parent Image:\n"\
            "  Name: %s\n  Image ID: %d" % (imageName, imageId)
        dataset.description = rstring(desc)
        dataset = updateService.saveAndReturnObject(dataset)
        parentDataset = dataset
    else:
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
        link = []
        for iid in iIds:
            dsLink = omero.model.DatasetImageLinkI()
            dsLink.parent = omero.model.DatasetI(
                parentDataset.id.val, False)
            dsLink.child = omero.model.ImageI(iid, False)
            updateService.saveObject(dsLink)
            link.append(dsLink)
        if parentProject and parentProject.canLink():
            # and put it in the   current project
            projectLink = omero.model.ProjectDatasetLinkI()
            projectLink.parent = omero.model.ProjectI(
                parentProject.getId(), False)
            projectLink.child = omero.model.DatasetI(
                dataset.id.val, False)
            updateService.saveAndReturnObject(projectLink)
    return images, dataset, link, iIds


def make_images_from_rois(conn, parameterMap):
    """
    Processes the list of Image_IDs, either making a new image-stack or a new
    dataset from each image, with new image planes coming from the regions in
    Rectangular ROIs on the parent images.
    """

    dataType = parameterMap["Data_Type"]

    message = ""

    # Get the images
    objects, logMessage = script_utils.getObjects(conn, parameterMap)
    message += logMessage
    if not objects:
        return None, message

    # Concatenate images from datasets
    if dataType == 'Image':
        images = objects
    else:
        images = []
        for ds in objects:
            images += ds.listChildren()

    # Check for rectangular ROIs and filter images list
    images = [image for image in images if image.getROICount("Rect") > 0]
    if not images:
        message += "No rectangle ROI found."
        return None, message

    total_rois = sum([i.getROICount("Rect") for i in images])
    if total_rois > 10:
        message += "Cannot start batch processing - too many rois (maximum is 10)."
        return None, message
        
    imageIds = [i.getId() for i in images]
    newImages = []
    newDatasets = []
    links = []
    for iId in imageIds:
        newImage, newDataset, link, new_ids = process_image(conn, iId, parameterMap)
        if newImage is not None:
            if isinstance(newImage, list):
                newImages.extend(newImage)
            else:
                newImages.append(newImage)
        if newDataset is not None:
            newDatasets.append(newDataset)
        if link is not None:
            if isinstance(link, list):
                links.extend(link)
            else:
                links.append(link)

    if newImages:
        if len(newImages) > 1:
            message += "Created %s new images" % len(newImages)
        else:
            message += "Created a new image"
    else:
        message += "No image created"

    if newDatasets:
        if len(newDatasets) > 1:
            message += " and %s new datasets" % len(newDatasets)
        else:
            message += " and a new dataset"
    
    print parameterMap['Email_Results']
    if parameterMap['Email_Results'] and (newImages or newDatasets):
        email_results(conn,parameterMap,new_ids)

    if not links or not len(links) == len(newImages):
        message += " but some images could not be attached"
    message += "."

    robj = (len(newImages) > 0) and newImages[0]._obj or None
    return robj, message


def list_image_names(conn, ids):
    """Builds a list of the image names"""
    image_names = []
    for image_id in ids:
        img = conn.getObject('Image', image_id)
        if not img:
            continue

        ds = img.getParent()
        if ds:
            pr = ds.getParent()
        else:
            pr = None

        image_names.append("[%s][%s] Image %d : %s" % (
                           pr and pr.getName() or '-',
                           ds and ds.getName() or '-',
                           image_id, os.path.basename(img.getName())))

    return image_names

def email_results(conn,params,image_ids):
    """
    E-mail the result to the user.

    @param conn: The BlitzGateway connection
    @param results: Dict of (imageId,text_result) pairs
    @param report: The results report
    @param params: The script parameters
    """
    print params['Email_Results']
    if not params['Email_Results']:
        return

    image_names = list_image_names(conn, image_ids)

    msg = MIMEMultipart()
    msg['From'] = ADMIN_EMAIL
    msg['To'] = params['Email_address']
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = '[OMERO Job] Slide Scanner image cropping'
    msg.attach(MIMEText("""
New images created from ROIs:

Format:
[parent project/datset][child dataset] new image id : parent image name

------------------------------------------------------------------------
%s""" % ("\n".join(image_names))))

    smtpObj = smtplib.SMTP('localhost')
    smtpObj.sendmail(ADMIN_EMAIL, [params['Email_address']], msg.as_string())
    smtpObj.quit()

def validate_email(conn, params):
    """
    Checks that a valid email address is present for the user_id

    @param conn: The BlitzGateway connection
    @param params: The script parameters
    """
    userEmail = ''
    if params['Email_address']:
        userEmail = params['Email_address']
    else:
        user = conn.getUser()
        user.getName() # Initialises the proxy object for simpleMarshal
        dic = user.simpleMarshal()
        if 'email' in dic and dic['email']:
            userEmail = dic['email']

    params['Email_address'] = userEmail
    print userEmail
    # Validate with a regular expression. Not perfect but it will do
    return re.match("^[a-zA-Z0-9._%-]+@[a-zA-Z0-9._%-]+.[a-zA-Z]{2,6}$",
                    userEmail)

def runAsScript():
    """
    The main entry point of the script, as called by the client via the
    scripting service, passing the required parameters.
    """
    printDuration(False)    # start timer
    dataTypes = [rstring('Dataset'), rstring('Image')]

    client = scripts.client(
        'Images_From_ROIs.py',
        """Crop rectangular regions from slide scanner images. WARNING: THIS PROCESS CAN TAKE A LONG TIME - APPROX MAXIMUM BATCH SIZE IS 10 ROIs!""",

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose Images via their 'Dataset' or directly by "
            " 'Image' IDs.", values=dataTypes, default="Image"),

        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Dataset IDs or Image IDs to "
            " process.").ofType(rlong(0)),

        scripts.String(
            "Container_Name", grouping="3",
            description="Option: put Images in new Dataset with this name",
            default="From_ROIs"),
                            
        scripts.Bool(
            "Email_Results", grouping="4", default=True,
            description="E-mail the results"),
                            
        scripts.String("Email_address", grouping="4.1",
        description="Specify e-mail address"),

        version="5.0.2",
        authors=["Daniel Matthews", "QBI"],
        institutions = ["University of Queensland"],
        contact = "d.matthews1@uq.edu.au",
    )

    try:
        parameterMap = client.getInputs(unwrap=True)
        print parameterMap

        # create a wrapper so we can use the Blitz Gateway.
        conn = BlitzGateway(client_obj=client)
        
        if parameterMap['Email_Results'] and not validate_email(conn, parameterMap):
            client.setOutput("Message", rstring("No valid email address"))
            return

        robj, message = make_images_from_rois(conn, parameterMap)

        client.setOutput("Message", rstring(message))
        if robj is not None:
            client.setOutput("Result", robject(robj))

    finally:
        client.closeSession()
        printDuration()

if __name__ == "__main__":
    runAsScript()
#     box = [150, 150, 75, 225]
#     Largeshape = [400, 400]
#     block_size = [50, 50]
#     mblocks = float(box[2])/float(block_size[0])
#     nblocks = float(box[3])/float(block_size[1])
#     largeimage = np.ones((400,400))
#     tile = tile_image_gen(largeimage,box)
#     print tile.shape