#!/usr/bin/env python
# -*- coding: utf-8 -*-
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders
from email.Utils import formatdate
import smtplib
import re
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.rtypes import rstring, rlong, robject
import omero.util.script_utils as script_utils

import os

import time
startTime = 0

ADMIN_EMAIL = 'admin@omero.host.com'


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

        def tileGen():
            for i, t in enumerate(pixels.getTiles(zctTileList)):
                yield t

        print "sizeZ, sizeC, sizeT", sizeZ, sizeC, sizeT
        description = "Created from image:\n  Name: %s\n  Image ID: %d"\
            " \n x: %d y: %d" % (imageName, imageId, x, y)
        newImg = conn.createImageFromNumpySeq(
            tileGen(), imageName,
            sizeZ=sizeZ, sizeC=sizeC, sizeT=sizeT,
            description=description, sourceImageId=imageId)

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
    return images, dataset, link


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

    imageIds = [i.getId() for i in images]
    newImages = []
    newDatasets = []
    links = []
    for iId in imageIds:
        newImage, newDataset, link = process_image(conn, iId, parameterMap)
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
        email_results(conn,parameterMap)

    if not links or not len(links) == len(newImages):
        message += " but some images could not be attached"
    message += "."

    robj = (len(newImages) > 0) and newImages[0]._obj or None
    return robj, message

def email_results(conn,params):
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

    #image_names = list_image_names(conn, results)

    msg = MIMEMultipart()
    msg['From'] = ADMIN_EMAIL
    msg['To'] = params['Email_address']
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = '[OMERO Job] Slide Scanner image cropping'
    msg.attach(MIMEText("""New images created from ROIs:"""))

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
        """Create new Images from the regions defined by Rectangle ROIs on \
        images recorded on QBI Slide Scanner microscopes""",

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
                            
        scripts.String("Email_address", grouping="4.1", default="Email",
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
