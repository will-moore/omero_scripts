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
# import omero
# import omero.scripts as scripts
# from omero.gateway import BlitzGateway
# from omero.rtypes import rstring, rlong, robject
# import omero.util.script_utils as script_utils

import os

import time
startTime = 0

ADMIN_EMAIL = 'admin@omerocloud.qbi.uq.edu.au'


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

def get_block(source,block_struct,row,col,block_size):

    # compute starting row/col in source image of block of data
    source_min_row = 1 + block_size[0] * (row - 1)
    source_min_col = 1 + block_size[1] * (col - 1)
    source_max_row = source_min_row + block_size[0] - 1
    source_max_col = source_min_col + block_size[1] - 1
#     if ~options.PadPartialBlocks
#         source_max_row = min(source_max_row,source.shape[0]);
#         source_max_col = min(source_max_col,source.shape[1]);
#     end
    
    # set block struct location (before border pixels are considered)
    location = [source_min_row, source_min_col]
    
    # add border pixels around the block of data
#     source_min_row = source_min_row - options.BorderSize[0]
#     source_max_row = source_max_row + options.BorderSize[0];
#     source_min_col = source_min_col - options.BorderSize[1];
#     source_max_col = source_max_col + options.BorderSize[1];
    
    # setup indices for target block
    total_rows = source_max_row - source_min_row + 1
    total_cols = source_max_col - source_min_col + 1
    
    # for interior blocks
    if (source_min_row >= 1) and (source_max_row <= source.getSizeY()) and (source_min_col >= 1) and (source_max_col <= source.getSizeX()):
        
        # no padding necessary, just read data and return
        pixels = source.getPrimaryPixels()
        data = pixels.getTiles(zctTileList)
        
    elif:
        
        # setup target indices variables
        target_min_row = 1;
        target_max_row = total_rows;
        target_min_col = 1;
        target_max_col = total_cols;
        
        % check each edge of the requested block for edge
        if source_min_row < 1
            delta = 1 - source_min_row;
            source_min_row = source_min_row + delta;
            target_min_row = target_min_row + delta;
        end
        if source_max_row > source.shape[0]
            delta = source_max_row - source.shape[0];
            source_max_row = source_max_row - delta;
            target_max_row = target_max_row - delta;
        end
        if source_min_col < 1
            delta = 1 - source_min_col;
            source_min_col = source_min_col + delta;
            target_min_col = target_min_col + delta;
        end
        if source_max_col > source.shape[1]
            delta = source_max_col - source.shape[1];
            source_max_col = source_max_col - delta;
            target_max_col = target_max_col - delta;
        end
        
        % read source data
        source_data = source.readRegion(...
            [source_min_row                      source_min_col],...
            [source_max_row - source_min_row + 1 source_max_col - source_min_col + 1]);
        
        % allocate target block (this implicitly also handles constant value
        % padding around the edges of the partial blocks and boundary
        % blocks)
        inputClass = str2func(class(source_data));
        options.PadValue = inputClass(options.PadValue);
        block_struct.data = repmat(options.PadValue,[total_rows total_cols size(source_data,3)]);
        
        % copy valid data into target block
        target_rows = target_min_row:target_max_row;
        target_cols = target_min_col:target_max_col;
        block_struct.data(target_rows,target_cols,:) = source_data;
        
    else
        
        % in this code path, have are guaranteed to require *some* padding,
        % either options.PadPartialBlocks, a border, or both.
        
        % Compute padding indices for entire input image
        has_border = ~isequal(options.BorderSize,[0 0]);
        if ~has_border
            % options.PadPartialBlocks only
            aIdx = getPaddingIndices(source.shape(1:2),...
                options.Padding(1:2),options.PadMethod,'post');
            row_idx = aIdx{1};
            col_idx = aIdx{2};
            
        else
            % has a border...
            if  ~options.PadPartialBlocks
                % pad border only, around entire image
                aIdx = getPaddingIndices(source.shape(1:2),...
                    options.BorderSize,options.PadMethod,'both');
                row_idx = aIdx{1};
                col_idx = aIdx{2};
                
                
            else
                % both types of padding required
                aIdx_pre = getPaddingIndices(source.shape(1:2),...
                    options.BorderSize,options.PadMethod,'pre');
                post_padding = options.Padding(1:2) + options.BorderSize;
                aIdx_post = getPaddingIndices(source.shape(1:2),...
                    post_padding,options.PadMethod,'post');
                
                % concatenate the post padding onto the pre-padding results
                row_idx = [aIdx_pre{1} aIdx_post{1}(end-post_padding[0]+1:end)];
                col_idx = [aIdx_pre{2} aIdx_post{2}(end-post_padding[1]+1:end)];
                
            end
        end
        
        % offset the indices of our desired block to account for the
        % pre-padding in our padded index arrays
        source_min_row = source_min_row + options.BorderSize[0];
        source_max_row = source_max_row + options.BorderSize[0];
        source_min_col = source_min_col + options.BorderSize[1];
        source_max_col = source_max_col + options.BorderSize[1];
        
        % extract just the indices of our desired block
        block_row_ind = row_idx(source_min_row:source_max_row);
        block_col_ind = col_idx(source_min_col:source_max_col);
        
        % compute the absolute row/col limits containing all the necessary
        % data from our source image
        block_row_min = min(block_row_ind);
        block_row_max = max(block_row_ind);
        block_col_min = min(block_col_ind);
        block_col_max = max(block_col_ind);
        
        % read the block from the adapter object containing all necessary data
        source_data = source.readRegion(...
            [block_row_min                      block_col_min],...
            [block_row_max - block_row_min + 1  block_col_max - block_col_min + 1]);
        
        % offset our block_row/col_inds to align with the data read from the
        % adapter
        block_row_ind = block_row_ind - block_row_min + 1;
        block_col_ind = block_col_ind - block_col_min + 1;
        
        % finally index into our block of source data with the correctly
        % padding index lists
        block_struct.data = source_data(block_row_ind,block_col_ind,:);
        
    end
    
    data_size = [size(block_struct.data,1) size(block_struct.data,2)];
    block_struct.block_size = data_size - 2 * block_struct.border

def get_blocks(block_size,mblocks,nblocks,box):
    if float(box[2])%float(block_size[0]) != 0.0:
        mpartial = (float(box[2])%float(block_size[0]))/float(block_size[0])
    else:
        mpartial = 0.0
    if float(box[3])/float(block_size[1]) != 0.0:
        npartial = (float(box[3])%float(block_size[1]))/float(block_size[1])
    else:
        npartial = 0.0

    num_blocks = mblocks * nblocks
    blk_coords = np.zeros((num_blocks,2))
    blk_coords[0,:] = box[0:2]
    for row_blk in range(1,mblocks):
        row_blk_pos = 
    
def block_tile_gen(pixels,box):
    #pixels is the raw pixel store
    
    #first work out the location and number blocks required
    block_size = (500,500)
    
    # total number of blocks we'll process (including partials)
    mblocks = (box[2] + padding[0]) / block_size[0]
    nblocks = (box[3] + padding[1]) / block_size[1]
    
    # determine the block indices
    #get_block_indices(box,mblock,nblocks)
    
    """ Given [x,y,w,h] of the region in the image to crop,
    break up the region into blocks of a given size.
    
    Determine the [x,y,w,h] of each block required from the image
    
    Get the block from the image.
    
    Determine where in the new data set the block needs to go
    
    Make the new dataset
    
    Write the dataset """
    
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
                    
        block_tile_gen(pixels,r)

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
    #runAsScript()
    box = [150, 150, 100, 225]
    Largeshape = [400, 400]
    block_size = [50, 50]
    mblocks = float(box[2])/float(block_size[0])
    nblocks = float(box[3])/float(block_size[1])

    get_blocks(block_size,mblocks,nblocks,box)