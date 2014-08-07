import tables
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=tables.PerformanceWarning)

class SlideImage:
    def __init__(self, filename):
        self.filename = filename
        try:
            self.slide = tables.openFile(filename,'r')
        except IOError:
            print('Specified file does not exist')
            return

        self.dataSetInfo = self.slide.getNode('/DataSetInfo')
        print 'dataSetInfo:',self.dataSetInfo
        self.imageInfo = self.slide.getNode('/DataSetInfo/Image/')
        print 'imageInfo:',self.imageInfo
        self.resLevels = self.resolution_levels()
        print 'resLevels:',self.resLevels
        self.numChannels = self.channels()
        print self.numChannels

    def get_data_in_channel(self, selectedRes, channelNum, region=None):
        imSize = self.image_size_from_data()[selectedRes]
        if region:
            regionW = region[3]-region[2]
            regionH = region[1]-region[0]
        else:
            regionW = imSize[0]
            regionH = imSize[1]
        im_dtype = np.dtype('uint8')
        respath = '/DataSet/ResolutionLevel ' + str(selectedRes) + '/'
        imrespath = respath + 'TimePoint 0' + '/' + 'Channel ' + str(channelNum)
        self.imarray = np.zeros((1,regionH,regionW),dtype=im_dtype)
        data = self.slide.getNode(imrespath,'Data')
        if region:
            self.imarray[0,:,:] = data[0,region[0]:region[1],region[2]:region[3]]
        else:
            self.imarray[0:,:] = data[0,:,:]            
        return self.imarray  
    
    def get_data(self, selectedRes,channelNum,region=None):
        imSize = self.image_size_from_data()[selectedRes]
        if region:
            regionW = region[3]-region[2]
            regionH = region[1]-region[0]
        else:
            regionW = imSize[0]
            regionH = imSize[1]
        print 'regionw, regionh:',regionW,regionH
        im_dtype = np.dtype('uint8')
        if type(channelNum) == list: #if it is a list we are either pulling RGB or all channels for cropping
            numChannels = self.numChannels
            respath = '/DataSet/ResolutionLevel ' + str(selectedRes) + '/'
#            self.imarray = np.zeros((imSize[1],imSize[0],numChannels),dtype=im_dtype)
            self.imarray = np.zeros((regionH,regionW,numChannels),dtype=im_dtype)
            for ii in range(numChannels):
                imrespath = respath + 'TimePoint 0' + '/' + 'Channel ' + str(ii)
                data = self.slide.getNode(imrespath,'Data')
                if region:
                    self.imarray[:,:,ii] = data[0,region[0]:region[1],region[2]:region[3]]
                else:
                    self.imarray[:,:,ii] = data[0,:,:]
        else:
            numChannels = self.numChannels
            respath = '/DataSet/ResolutionLevel ' + str(selectedRes) + '/'
            imrespath = respath + 'TimePoint 0' + '/' + 'Channel ' + str(channelNum)
#            self.imarray = np.zeros((imSize[1],imSize[0],1))
            self.imarray = np.zeros((regionH,regionW,1),dtype=im_dtype)
            pathToData = self.slide[imrespath]
            data = pathToData["Data"]
            if region:
                self.imarray[:,:,0] = data[0,region[0]:region[1],region[2]:region[3]]
            else:
                self.imarray[:,:,0] = data[0,:,:]            
            #get just a single channel
        return self.imarray        
      


    def display_image(self):
        respath = '/DataSet/ResolutionLevel 7/'
        minrespath = respath + 'TimePoint 0/' + 'Channel 0'
        self.imres7 = self.slide[minrespath]["Data"]
        return self.imres7

#class SlideMeta(SlideImage):

##    def __init__(self, filename):
##        print(self.dataSetInfo)
##        self.resLevels = len(self.slide['/DataSet'])
##        self.numChannels = len(self.slide['/DataSet/ResolutionLevel 0/TimePoint 0/'])
        
    def micro_mode(self):
        MM = self.imageInfo._f_getAttr('MicroscopeMode')
        self.micromode = list2str(MM)
        return self.micromode
        
    def resolution_levels(self):
        levels = 0
        for node in self.slide.root.DataSet:
            levels = levels + 1
        return levels

    def time_points(self):
        addr = '/DataSet/ResolutionLevel 0/'
        node = self.slide.getNode(addr)
        time_points = 0
        for d in node:
            time_points += 1                
        return time_points
        
    def channels(self):
        addr = '/DataSet/ResolutionLevel 0/TimePoint 0/'
        node = self.slide.getNode(addr)
        channels = 0
        for d in node:
            channels += 1                
        return channels

    def image_size_from_data(self):
        imSizeArrayFromData = np.zeros((8,2))
        for r in range(self.resLevels):
            res_level = "ResolutionLevel " + str(r)
            path = "/DataSet/" + res_level + "/TimePoint 0/" + "Channel 0"
            data = self.slide.getNode(path,'Data')
            imSizeArrayFromData[r,0]= data.shape[2]
            imSizeArrayFromData[r,1]= data.shape[1]
        return imSizeArrayFromData    
        
    def image_size(self):
        imSizeArray = np.zeros((8,2))
        for r in range(self.resLevels):
            res_level = "ResolutionLevel " + str(r)
            path = "/DataSet/" + res_level + "/TimePoint 0/"
            folder = "Channel 0"
            chanAttribs = dict_builder(self.slide, path, folder, 0)
            sizeX = int(chanAttribs['ImageSizeX'])
            sizeY = int(chanAttribs['ImageSizeY'])
            imSizeArray[r,0] = sizeX
            imSizeArray[r,1] = sizeY

        return imSizeArray

#    def extents(self):
#        self.extMax = []
#        self.extMin = []
#        for e in range(3):
#            maxstr  = "ExtMax" + str(e)
#            extMaxVal = self.imageInfo.attrs[maxstr]
#            extMaxVal = int(float(list2str(extMaxVal)))
#            self.extMax.append(extMaxVal)
#            
#            minstr  = "ExtMin" + str(e)
#            extMinVal = self.imageInfo.attrs[minstr]
#            extMinVal = int(float(list2str(extMinVal)))
#            self.extMin.append(extMinVal)
#        return self.extMin, self.extMax
#
#    def scale(self):
#        self.xscale = []
#        self.yscale = []
#        slideSize = self.image_size()
#        maxressize = slideSize[0]
#        Xmax = maxressize[2]
#        Ymax = maxressize[1]
#        for r in range(self.resLevels):
#            slideshape = slideSize[r]
#            self.xscale.append(float(slideshape[2]) / float(Xmax))
#            self.yscale.append(float(slideshape[1]) / float(Ymax))
#            
#        return self.xscale, self.xscale

    def dataSet_channels(self):
        #extracted from DataSet group
        numchans = self.numChannels
        folder = "Channel "
        path = "/DataSet/ResolutionLevel 0/TimePoint 0/"
        self.DataSetChanList = dict_builder(self.slide, path, folder, numchans)
        return self.DataSetChanList

    def datasetinfo_channels(self):
        #extracted from DataSetInfo group
        numchans = self.numChannels

        folder = "Channel "
        path = "/DataSetInfo/"
        self.DataSetInfoChanList = dict_builder(self.slide, path, folder, numchans)
        return self.DataSetInfoChanList
    
    def get_channel_data(self):
        channelNames = []
        DataSetInfoChanList = self.datasetinfo_channels()
        for chan in range(self.numChannels):
            channelInfo = DataSetInfoChanList[chan]
            channelNames.append(channelInfo['Name'])
        return channelNames

    def datasetinfo_image(self):
        folder = "Image"
        path = "/DataSetInfo/"
        self.DataSetInfoImgList = dict_builder(self.slide, path, folder, 0)
        return self.DataSetInfoImgList
        
    def datasetinfo_imaris(self):
        folder = "Imaris"
        path = "/DataSetInfo/"
        self.DataSetInfoImarisList = dict_builder(self.slide, path, folder, 0)
        return self.DataSetInfoImarisList

    def datasetinfo_imarisdataset(self):
        folder = "ImarisDataSet"
        path = "/DataSetInfo/"
        self.DataSetInfoImarisDataSetList = dict_builder(self.slide, path, folder, 0)
        return self.DataSetInfoImarisDataSetList

    def datasetinfo_log(self):
        folder = "Log"
        path = "/DataSetInfo/"
        self.DataSetInfoLogList = dict_builder(self.slide, path, folder, 0)
        return self.DataSetInfoLogList

    def datasetinfo_capture(self):
        folder = "MF Capt Channel "
        path = "/DataSetInfo/"
        numchans = self.numChannels
        self.DataSetInfoCaptList = dict_builder(self.slide, path, folder, numchans)
        return self.DataSetInfoCaptList

    def datasetinfo_time(self):
        folder = "TimeInfo"
        path = "/DataSetInfo/"
        self.DataSetInfoTimeList = dict_builder(self.slide, path, folder, 0)        
        return self.DataSetInfoTimeList
    
    def close_slide(self):
        self.slide.close()
        
        
#Helper functions        
def dict_builder(inputfile,path, folder, numFolders):
        
    numattrs = []
    attrsKeys = []
    
    if numFolders == 0:
        metaDictList = []
        folderstr = folder + "/"
        folderpath = path + folderstr
        datagrp = inputfile.getNode(folderpath)
        attrsKeys = datagrp._v_attrs._v_attrnames
        numattrs = len(datagrp._v_attrs._v_attrnames)
        metaDict = {}
        for attr in range(numattrs):
            attrKey2str = str(attrsKeys[attr])
            attrValue = list2str(datagrp._f_getAttr(attrKey2str))
            metaDict[attrKey2str] = attrValue

        #sort_chandict = sorted(chandict.iterkeys())
        metaDictList = metaDict
    elif numFolders > 0:
        
        metaDictList = []
        for f in range(numFolders):
            folderstr = folder + str(f) + "/"
            folderpath = path + folderstr
            datagrp = inputfile.getNode(folderpath)
            attrsKeys.append(datagrp._v_attrs._v_attrnames)
            numattrs.append(len(datagrp._v_attrs._v_attrnames))
            metaDict = {}
            for attr in range(numattrs[f]):
                attrKey2str = str(attrsKeys[f][attr])
                attrValue = list2str(datagrp._f_getAttr(attrKey2str))
                metaDict[attrKey2str] = attrValue
                    
            #sort_chandict = sorted(chandict.iterkeys())
            metaDictList.append(metaDict)
             
    return metaDictList

def list2str(inputarray):
    str_list = []
    for iExt in range(len(inputarray)):
        str_list.append(inputarray[iExt])
        
    return ''.join(str_list)

if __name__ == '__main__':
    path = '/media/sf_uqdmatt2/Shared with Win7/test.ims'
    slide = SlideImage(path)
    #data = slide.get_data_in_channel(7,0)
    #print 'data shape:',data.shape
    mm = slide.micro_mode()
    print 'micro mode:',mm
    cl = slide.datasetinfo_channels()
    ImageData = slide.datasetinfo_image()
    sizeArray = slide.image_size()
    print sizeArray
    sizeT = slide.time_points()
    print sizeT