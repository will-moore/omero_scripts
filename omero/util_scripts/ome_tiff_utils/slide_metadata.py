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
        self.imageInfo = self.slide.getNode('/DataSetInfo/Image/')
        self.sizeR = self.get_size_r()
        self.sizeC = self.get_size_c()
        self.sizeT = self.get_size_t()

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
        numChannels = self.sizeC
        imSize = self.image_size_from_data()[selectedRes]
        if region:
            regionW = region[3]-region[2]
            regionH = region[1]-region[0]
        else:
            regionW = imSize[0]
            regionH = imSize[1]
        print 'regionw, regionh:',regionW,regionH
        im_dtype = np.dtype('uint8')
        if type(channelNum) == list:
            respath = '/DataSet/ResolutionLevel ' + str(selectedRes) + '/'
            self.imarray = np.zeros((regionH,regionW,numChannels),dtype=im_dtype)
            for ii in range(numChannels):
                imrespath = respath + 'TimePoint 0' + '/' + 'Channel ' + str(ii)
                data = self.slide.getNode(imrespath,'Data')
                if region:
                    self.imarray[:,:,ii] = data[0,region[0]:region[1],region[2]:region[3]]
                else:
                    self.imarray[:,:,ii] = data[0,:,:]
        else:
            respath = '/DataSet/ResolutionLevel ' + str(selectedRes) + '/'
            imrespath = respath + 'TimePoint 0' + '/' + 'Channel ' + str(channelNum)
            self.imarray = np.zeros((regionH,regionW,1),dtype=im_dtype)
            pathToData = self.slide[imrespath]
            data = pathToData["Data"]
            if region:
                self.imarray[:,:,0] = data[0,region[0]:region[1],region[2]:region[3]]
            else:
                self.imarray[:,:,0] = data[0,:,:]            
        return self.imarray        

    def display_image(self):
        respath = '/DataSet/ResolutionLevel 7/'
        minrespath = respath + 'TimePoint 0/' + 'Channel 0'
        self.imres7 = self.slide[minrespath]["Data"]
        return self.imres7
        
    def micro_mode(self):
        MM = self.imageInfo._f_getAttr('MicroscopeMode')
        self.micromode = list2str(MM)
        return self.micromode
        
    def get_size_r(self):
        levels = 0
        for node in self.slide.root.DataSet:
            levels = levels + 1
        return levels

    def get_size_t(self):
        addr = '/DataSet/ResolutionLevel 0/'
        node = self.slide.getNode(addr)
        size_t = 0
        for d in node:
            size_t += 1                
        return size_t
        
    def get_size_c(self):
        addr = '/DataSet/ResolutionLevel 0/TimePoint 0/'
        node = self.slide.getNode(addr)
        size_c = 0
        for d in node:
            size_c += 1                
        return size_c

    def image_size_from_data(self):
        imSizeArrayFromData = np.zeros((8,2))
        for r in range(self.sizeR):
            res_level = "ResolutionLevel " + str(r)
            path = "/DataSet/" + res_level + "/TimePoint 0/" + "Channel 0"
            data = self.slide.getNode(path,'Data')
            imSizeArrayFromData[r,0] = data.shape[2]
            imSizeArrayFromData[r,1] = data.shape[1]
        return imSizeArrayFromData    
        
    def image_size(self):
        imSizeArray = np.zeros((8,2))
        for r in range(self.sizeR):
            res_level = "ResolutionLevel " + str(r)
            path = "/DataSet/" + res_level + "/TimePoint 0/"
            folder = "Channel 0"
            chanAttribs = dict_list_builder(self.slide, path, folder, 1)[0]
            sizeX = int(chanAttribs['ImageSizeX'])
            sizeY = int(chanAttribs['ImageSizeY'])
            imSizeArray[r,0] = sizeX
            imSizeArray[r,1] = sizeY

        return imSizeArray

    def dataSet_channels(self):
        #extracted from DataSet group
        folder = "Channel "
        path = "/DataSet/ResolutionLevel 0/TimePoint 0/" 
        return dict_list_builder(self.slide, path, folder, self.sizeC)

    def datasetinfo_channels(self):
        #extracted from DataSetInfo group
        folder = "Channel "
        path = "/DataSetInfo/" 
        return dict_list_builder(self.slide, path, folder, self.sizeC)
    
    def get_channel_data(self):
        channelNames = []
        DataSetInfoChanList = self.datasetinfo_channels()
        for chan in range(self.sizeC):
            channelInfo = DataSetInfoChanList[chan]
            channelNames.append(channelInfo['Name'])
        return channelNames

    def datasetinfo_image(self):
        folder = "Image"
        path = "/DataSetInfo/"
        return dict_list_builder(self.slide, path, folder, 1)[0]
        
    def datasetinfo_imaris(self):
        folder = "Imaris"
        path = "/DataSetInfo/"
        return dict_list_builder(self.slide, path, folder, 1)[0]

    def datasetinfo_imarisdataset(self):
        folder = "ImarisDataSet"
        path = "/DataSetInfo/" 
        return dict_list_builder(self.slide, path, folder, 1)[0]

    def datasetinfo_log(self):
        folder = "Log"
        path = "/DataSetInfo/"
        return dict_list_builder(self.slide, path, folder, 1)[0]

    def datasetinfo_capture(self):
        folder = "MF Capt Channel "
        path = "/DataSetInfo/"
        return dict_list_builder(self.slide, path, folder, self.sizeC)

    def datasetinfo_time(self):
        folder = "TimeInfo"
        path = "/DataSetInfo/"      
        return dict_list_builder(self.slide, path, folder, 1)[0]  
    
    def close_slide(self):
        self.slide.close()
        
        
#Helper functions        
def dict_list_builder(inputfile, path, folder, numFolders):
    
    numattrs = []
    attrsKeys = []
    metaDictList = []

    for f in range(numFolders):
        if numFolders == 1:
            folderstr = folder + "/"
        else:
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
        metaDictList.append(metaDict)     
            
    return metaDictList

def list2str(inputarray):
    str_list = []
    for iExt in range(len(inputarray)):
        str_list.append(inputarray[iExt])
        
    return ''.join(str_list)
