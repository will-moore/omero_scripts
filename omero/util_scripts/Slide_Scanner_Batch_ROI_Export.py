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
from omero.rtypes import rstring, rlong, robject, rint
import omero.util.script_utils as script_utils
import logging
logger = logging.getLogger(__name__)
import os

import time
startTime = 0

ADMIN_EMAIL = 'admin@omerocloud.qbi.uq.edu.au'

def createImageFromNumpyTileSeq (conn, zctPlanes, (x,y,w,h),imageName, sizeZ=1, sizeC=1, sizeT=1, description=None, dataset=None, sourceImageId=None, channelList=None):
    """
    This version of createImageFromNumpySeq is a temporary workaround for ticket 12459 - 
    original version does not allow creating of new images tile-by-tile
    D.matthews QBI
    
    Creates a new multi-dimensional image from the sequence of 2D np arrays in zctPlanes.
    zctPlanes should be a generator of np 2D arrays of shape (sizeY, sizeX) ordered
    to iterate through T first, then C then Z.
    Example usage:
    original = conn.getObject("Image", 1)
    sizeZ = original.getSizeZ()
    sizeC = original.getSizeC()
    sizeT = original.getSizeT()
    clist = range(sizeC)
    zctList = []
    for z in range(sizeZ):
    for c in clist:
    for t in range(sizeT):
    zctList.append( (z,c,t) )
    def planeGen():
    planes = original.getPrimaryPixels().getPlanes(zctList)
    for p in planes:
    # perform some manipulation on each plane
    yield p
    createImageFromNumpySeq (planeGen(), imageName, sizeZ=sizeZ, sizeC=sizeC, sizeT=sizeT, sourceImageId=1, channelList=clist)
    
    @param session An OMERO service factory or equivalent with getQueryService() etc.
    @param zctPlanes A generator of np 2D arrays, corresponding to Z-planes of new image.
    @param imageName Name of new image
    @param description Description for the new image
    @param dataset If specified, put the image in this dataset. omero.model.Dataset object
    @param sourceImageId If specified, copy this image with metadata, then add pixel data
    @param channelList Copies metadata from these channels in source image (if specified). E.g. [0,2]
    @return The new OMERO image: omero.model.ImageI
    """
    queryService = conn.getQueryService()
    pixelsService = conn.getPixelsService()
    rawPixelsStore = conn.c.sf.createRawPixelsStore() # Make sure we don't get an existing rpStore
    containerService = conn.getContainerService()
    updateService = conn.getUpdateService()
    
    #import np
    
    def createImage(firstPlane, channelList):
        """ Create our new Image once we have the first plane in hand """
        convertToType = None
        sizeY, sizeX = firstPlane.shape
        if sourceImageId is not None:
            if channelList is None:
                channelList = range(sizeC)
            iId = pixelsService.copyAndResizeImage(sourceImageId, rint(sizeX), rint(sizeY), rint(sizeZ), rint(sizeT), channelList, None, False, conn.SERVICE_OPTS)
            # need to ensure that the plane dtype matches the pixels type of our new image
            img = conn.getObject("Image", iId.getValue())
            newPtype = img.getPrimaryPixels().getPixelsType().getValue()
            omeroToNumpy = {'int8':'int8', 'uint8':'uint8', 'int16':'int16', 'uint16':'uint16', 'int32':'int32', 'uint32':'uint32', 'float':'float32', 'double':'double'}
            if omeroToNumpy[newPtype] != firstPlane.dtype.name:
                convertToType = getattr(np, omeroToNumpy[newPtype])
            img._obj.setName(rstring(imageName))
            updateService.saveObject(img._obj, conn.SERVICE_OPTS)
        else:
            # need to map np pixel types to omero - don't handle: bool_, character, int_, int64, object_
            pTypes = {'int8':'int8', 'int16':'int16', 'uint16':'uint16', 'int32':'int32', 'float_':'float', 'float8':'float',
                        'float16':'float', 'float32':'float', 'float64':'double', 'complex_':'complex', 'complex64':'complex'}
            dType = firstPlane.dtype.name
            if dType not in pTypes: # try to look up any not named above
                pType = dType
            else:
                pType = pTypes[dType]
            pixelsType = queryService.findByQuery("from PixelsType as p where p.value='%s'" % pType, None) # omero::model::PixelsType
            if pixelsType is None:
                raise Exception("Cannot create an image in omero from np array with dtype: %s" % dType)
            channelList = range(sizeC)
            iId = pixelsService.createImage(sizeX, sizeY, sizeZ, sizeT, channelList, pixelsType, imageName, description, conn.SERVICE_OPTS)
    
        imageId = iId.getValue()
        return containerService.getImages("Image", [imageId], None, conn.SERVICE_OPTS)[0], convertToType
    
    def uploadPlane(plane, z, c, t, x, y, w, h, convertToType):
        # if we're given a np dtype, need to convert plane to that dtype
        if convertToType is not None:
            p = np.zeros(plane.shape, dtype=convertToType)
            p += plane
            plane = p
        byteSwappedPlane = plane.byteswap()
        convertedPlane = byteSwappedPlane.tostring();
        #rawPixelsStore.setPlane(convertedPlane, z, c, t, conn.SERVICE_OPTS)
        rawPixelsStore.setTile(convertedPlane, z, c, t, x, y, w, h, conn.SERVICE_OPTS)
    
    image = None
    dtype = None
    channelsMinMax = []
    exc = None
    try:
        for theZ in range(sizeZ):
            for theC in range(sizeC):
                for theT in range(sizeT):
                    plane = zctPlanes.next()
                    if image == None: # use the first plane to create image.
                        image, dtype = createImage(plane, channelList)
                        pixelsId = image.getPrimaryPixels().getId().getValue()
                        rawPixelsStore.setPixelsId(pixelsId, True, conn.SERVICE_OPTS)
                    uploadPlane(plane, theZ, theC, theT, x, y, w, h, dtype)
                    # init or update min and max for this channel
                    minValue = plane.min()
                    maxValue = plane.max()
                    if len(channelsMinMax) < (theC +1): # first plane of each channel
                        channelsMinMax.append( [minValue, maxValue] )
                    else:
                        channelsMinMax[theC][0] = min(channelsMinMax[theC][0], minValue)
                        channelsMinMax[theC][1] = max(channelsMinMax[theC][1], maxValue)
    except Exception, e:
        logger.error("Failed to setPlane() on rawPixelsStore while creating Image", exc_info=True)
        exc = e
    try:
        rawPixelsStore.close(conn.SERVICE_OPTS)
    except Exception, e:
        logger.error("Failed to close rawPixelsStore", exc_info=True)
        if exc is None:
            exc = e
    if exc is not None:
        raise exc
    
    try: # simply completing the generator - to avoid a GeneratorExit error.
        zctPlanes.next()
    except StopIteration:
        pass
    
    for theC, mm in enumerate(channelsMinMax):
        pixelsService.setChannelGlobalMinMax(pixelsId, theC, float(mm[0]), float(mm[1]), conn.SERVICE_OPTS)
        #resetRenderingSettings(renderingEngine, pixelsId, theC, mm[0], mm[1])
    
    # put the image in dataset, if specified.
    if dataset:
        link = omero.model.DatasetImageLinkI()
        link.parent = omero.model.DatasetI(dataset.getId(), False)
        link.child = omero.model.ImageI(image.id.val, False)
        updateService.saveObject(link, conn.SERVICE_OPTS)
    
    return conn.ImageWrapper(image)
    
def printDuration(output=True):
    global startTime
    if startTime == 0:
        startTime = time.time()
    if output:
        print "Script timer = %s secs" % (time.time() - startTime)


def getRectangles(conn, imageId):
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
                if x is None:   # get x, y, width, height for first rect only
                    x = int(shape.getX().getValue())
                    y = int(shape.getY().getValue())
                    width = int(shape.getWidth().getValue())
                    height = int(shape.getHeight().getValue())
        # if we have found any rectangles at all...
        if zStart is not None:
            rois.append((x, y, width, height, zStart, zEnd, tStart, tEnd))

    return rois

def get_block_coords(row,col,block_size,box):

    # compute starting row/col in source image of block of data
    source_min_row = block_size[0] * row + box[1]
    source_min_col = block_size[1] * col + box[0]
    source_max_row = source_min_row + block_size[0]
    source_max_col = source_min_col + block_size[1]
    source_max_row = min(source_max_row,box[1] + box[3])
    source_max_col = min(source_max_col,box[0] + box[2])
    
    # setup indices for target block
    total_rows = source_max_row - source_min_row
    total_cols = source_max_col - source_min_col

    if (source_min_row >= 0) and (source_max_row <= box[1] + box[3]) and (source_min_col >= 0) and (source_max_col <= box[0] + box[2]):        
        pass
    else:
        # setup target indices variables
        target_min_row = 0
        target_max_row = total_rows
        target_min_col = 0
        target_max_col = total_cols
        # check each edge of the requested block for edge
        if source_min_row < 0:
            delta = 1 - source_min_row
            source_min_row = source_min_row + delta
            target_min_row = target_min_row + delta
        
        if source_max_row > box[3]:
            delta = source_max_row - box[3]
            source_max_row = source_max_row - delta
            target_max_row = target_max_row - delta
        
        if source_min_col < 0:
            delta = 1 - source_min_col
            source_min_col = source_min_col + delta
            target_min_col = target_min_col + delta
        
        if source_max_col > box[2]:
            delta = source_max_col - box[2]
            source_max_col = source_max_col - delta
            target_max_col = target_max_col - delta

    x = source_min_col
    y = source_min_row
    w = source_max_col - source_min_col
    h = source_max_row - source_min_row
    return (x,y,w,h)

def get_block(source,box,block):
    # need a tile generator to get all the planes within the ROI
    xbox, ybox, wbox, hbox, z1box, z2box, t1box, t2box = box
    x,y,w,h = block
    pixels = source.getPrimaryPixels()
    sizeZ = z2box-z1box + 1
    sizeT = t2box-t1box + 1
    sizeC = source.getSizeC()
    zctTileList = []
    tile = (x, y, w, h)
    for z in range(z1box, z2box+1):
        for c in range(sizeC):
            for t in range(t1box, t2box+1):
                zctTileList.append((z, c, t, tile))

    def tileGen():
        for i, t in enumerate(pixels.getTiles(zctTileList)):
            yield t

    return tileGen()

def put_blocks(source,tiles,source_coords,box):
    sizeZ = source.getSizeZ()
    sizeT = source.getSizeT()
    sizeC = source.getSizeC()
    tile_data = np.zeros((box[3],box[2],sizeC))
    for i,tile in enumerate(tiles):
        target_min_row = source_coords[i][1] - box[1]
        target_min_col = source_coords[i][0] - box[0]
        target_max_row = target_min_row + source_coords[i][3]
        target_max_col = target_min_col + source_coords[i][2]
        for i,t in enumerate(tile):
            tile_data[target_min_row:target_max_row,target_min_col:target_max_col,i] = t
            
    def plane_gen():
        for z in range(sizeZ):
            for c in range(sizeC):
                for t in range(sizeT):
                    plane = tile_data[:,:,c]
                    yield plane
                    
    return plane_gen()
        
def block_gen(source,box):
    block_size = (1000,1000)
    
    # total number of blocks we'll process (including partials)
    mblocks = int(math.ceil(float(box[2]) / float(block_size[0])))
    nblocks = int(math.ceil(float(box[3]) / float(block_size[1])))
    num_blocks = mblocks * nblocks
    tiles = []
    blk_coords = []
    for row in range(0,nblocks):
        for col in range(0,mblocks):
            x,y,w,h = get_block_coords(row,col,block_size,box)
            blk_coords.append((x,y,w,h))
            tile = get_block(source,box,(x,y,w,h))
    return put_blocks(source,tiles,blk_coords,box)
    
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

    pixels = image.getPrimaryPixels()
    # note pixel sizes (if available) to set for the new images
    physicalSizeX = pixels.getPhysicalSizeX()
    physicalSizeY = pixels.getPhysicalSizeY()

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
    for r in rois:
        
        x, y, w, h, z1, z2, t1, t2 = r
        print "  ROI x: %s y: %s w: %s h: %s z1: %s z2: %s t1: %s t2: %s"\
            % (x, y, w, h, z1, z2, t1, t2)

        # need a tile generator to get all the planes within the ROI
        sizeZ = z2-z1 + 1
        sizeT = t2-t1 + 1
        sizeC = image.getSizeC()
        zctTileList = []
        tile = (x, y, w, h)
        print "zctTileList..."
        for z in range(z1, z2+1):
            for c in range(sizeC):
                for t in range(t1, t2+1):
                    zctTileList.append((z, c, t, tile))
        
        tile = block_gen(image,r)

        print "sizeZ, sizeC, sizeT", sizeZ, sizeC, sizeT
        description = "Created from image:\n  Name: %s\n  Image ID: %d"\
            " \n x: %d y: %d" % (imageName, imageId, x, y)
        newImg = conn.createImageFromNumpyTileSeq(conn,
            tile, imageName,
            sizeZ=sizeZ, sizeC=sizeC, sizeT=sizeT,
            description=description)

        print "New Image Id = %s" % newImg.getId()

        images.append(newImg)
        iIds.append(newImg.getId())

    if len(iIds) == 0:
        print "No new images created."
        return

    if 'Container_Name' in parameterMap and \
       len(parameterMap['Container_Name'].strip()) > 0:
        # create a new dataset for new images
        datasetName = parameterMap['Container_Name']
        print "\nMaking Dataset '%s' of Images from ROIs of Image: %s" \
            % (datasetName, imageId)
        print "physicalSize X, Y:  %s, %s" \
            % (physicalSizeX, physicalSizeY)
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
        """Crop rectangular regions from slide scanner images. MAXIMUM BATCH SIZE IS 10 ROIs!""",

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
#     tile = block_gen(largeimage,box)
#     print tile.shape