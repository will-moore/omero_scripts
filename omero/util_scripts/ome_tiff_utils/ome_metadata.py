#from __future__ import print_function
import math as m
import numpy as np
from ome_tiff_utils.ome_generator import ome_elem, OMEBase
from ome_tiff_utils.slide_metadata import SlideImage

groups = dict(
              QBIgroup=dict(name = 'QBIgroup', contact_id = 'QBIuser'))
people = dict(
              QBIuser=dict(first_name = 'First', last_name = 'Last',
              email = 'QBIuser@uq.edu.au', user_name = 'QBIuser',
              institution='Queensland Brain Institute', groups='QBIgroup')  
               )

class FileSizeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)                

class OMETIFF(OMEBase):

    prefix = 'ome'
    def __init__(self, userDetails, slide_name, outpath, section_id, total_sections, input_data, pixelregion, scalefact):
        OMEBase.__init__(self)
        self.outputtif_path = outpath
        self.ID = section_id
        self.Total = total_sections
        self.InputFilename = slide_name
        self.slide = input_data
        self.roi = pixelregion
        self.mode = input_data.micro_mode()
        self.scalefact = scalefact
        self.tile_width = 1024
        self.tile_height = 1024
        if userDetails is None:
            current_user = 'QBIuser'
            current_group = 'QBIgroup'
        else:
            current_user = userDetails[0]
            current_group = userDetails[1]
            
        if current_user not in people:
            people[current_user] = dict(first_name = userDetails[2], last_name = userDetails[3],
                                    email = userDetails[0]+'@uq.edu.au', user_name = userDetails[0],
                                    institution='Queensland Brain Institute', groups = current_group)  
        if current_group not in groups:
            groups[current_group] = dict(name = userDetails[1], contact_id = userDetails[0])
        
        self.current_user = current_user
        self.current_group = current_group
    
    def iter_Image(self, func):
#        get the meta data dictionaries
        meta = self.slide
         
        if self.mode == 'MetaCyte TL':
            self.instrument = 'Bright-field'
        elif self.mode == 'MetaCyte FL':
            self.instrument = 'Fluorescence'
                 
        channelData = meta.datasetinfo_channels()
        ImageData = meta.datasetinfo_image()
        self.sizeArray = meta.image_size()
        fullressize = self.sizeArray[0]
        cropressize = self.sizeArray[self.scalefact]
        scaleX = float(fullressize[0]) / float(cropressize[0])
        scaleY = float(fullressize[1]) / float(cropressize[1])
        classifier = ImageData['Metafer Classifer']
        if ' ' in str(self.InputFilename):
            DatasetID = str(self.InputFilename).replace(' ','_')
        else:
            DatasetID = str(self.InputFilename)
        self.sizeX = self.roi[2]
        self.sizeY = self.roi[3]          
        self.sizeZ = int(ImageData['Z'])
        self.sizeC = meta.sizeC
        self.sizeT = meta.sizeT

        if self.sizeT == 1:
            TimeIncrement = '0.0'
        else:
            TimeIncrement = str(self.sizeT)
            
        dtype = np.dtype('uint8')

        ExtminX = float(ImageData['ExtMin0'])
        ExtmaxX = float(ImageData['ExtMax0'])
        ExtminY = float(ImageData['ExtMin1'])
        ExtmaxY = float(ImageData['ExtMax1'])
        ExtminZ = float(ImageData['ExtMin2'])
        ExtmaxZ = float(ImageData['ExtMax2'])
        Width = m.fabs(ExtminX) + m.fabs(ExtmaxX)
        Height = m.fabs(ExtminY) + m.fabs(ExtmaxY)
        Depth = 0.32 #m.fabs(ExtminZ) + m.fabs(ExtmaxZ)
        Xres = ((Width / fullressize[0]) * scaleX)
        Yres = ((Height / fullressize[1]) * scaleY)
        Zres = Depth
        pixels_d = {}
        pixels_d['PhysicalSizeX'] = str(Xres)
        pixels_d['PhysicalSizeY'] = str(Yres)
        pixels_d['PhysicalSizeZ'] = str(Zres)
        pixels_d['TimeIncrement'] = TimeIncrement
        self.PhysSize = (1/Xres,1/Yres,1/Zres)
        order = 'XYZCT'
        channel_d = dict(SamplesPerPixel='1')
        channel_d['AcquisitionMode'] = 'WideField'
        lpath_l = []

        self.tif_uuid = self._mk_uuid()
        self.tif_filename = self.outputtif_path      
#        self.tif_images[self.instrument,self.tif_filename,self.tif_uuid,self.PhysSize] = tif_data        

        pixels = ome_elem.Pixels(
                    DimensionOrder=order, ID='Pixels:%s' % (self.instrument),
                    SizeX = str(self.sizeX), SizeY = str(self.sizeY), SizeZ = str(self.sizeZ), SizeT=str(self.sizeT), SizeC = str(self.sizeC),
                    Type = self.dtype2PixelIType (dtype), **pixels_d
                    )
        
        for chan in range(self.sizeC):           
            chan_dict = channelData[chan]
            channel_d['Name'] = chan_dict['Name']
            chan_color = chan_dict['Color']
            channel = ome_elem.Channel(ID='Channel:0:%s' %(chan), **channel_d)
            lpath = ome_elem.LightPath(*lpath_l)
            channel.append(lpath)
            pixels.append(channel)

        plane_l = []    
        for idx_chan in range(self.sizeC):    
            d = dict(IFD=str(idx_chan),FirstC=str(idx_chan), FirstZ='0',FirstT='0', PlaneCount='1')
            plane_l.append(d)
    
            tiffdata = ome_elem.TiffData(ome_elem.UUID (self.tif_uuid, FileName=self.tif_filename.split("\\")[-1]), **d)
            pixels.append(tiffdata)
                             
        date = ImageData['RecordingDate']
        date = date.replace(' ','T')
        description = 'This is section number ' + str(self.ID) + ' of ' + str(self.Total) + \
                        ' on ' + self.InputFilename + ' acquired using classifier: ' + classifier
            
        image = ome_elem.Image (ome_elem.AcquiredDate (date), 
                           #ome_elem.ExperimenterRef(ID='Experimenter:%s' % (self.current_user)),
                           ome_elem.Description('Description:%s' %(description)),
                           #ome.GroupRef(ID='Group:%s' %(self.current_group)),
                           ome_elem.DatasetRef(ID='Dataset:%s' % (DatasetID)),
                           pixels, ID='Image:%s' % (self.instrument))
        yield image
        return            
    
    def iter_Dataset(self, func):         
        if ' ' in str(self.InputFilename):
            DatasetID = str(self.InputFilename).replace(' ','_')
        else:
            DatasetID = str(self.InputFilename)
            
        e = func (ID='Dataset:%s' % (DatasetID))
        yield e
        
    def iter_Experimenter(self, func):
        expid = self.current_user
        d = people[expid]
        d1 = dict (Email=d['email'], Institution=d['institution'], UserName=expid)
        if 'first_name' in d:
            d1['FirstName'] = d['first_name']
            d1['LastName'] = d['last_name']
            d1['DisplayName'] = '%(first_name)s %(last_name)s' % d
        else:
            d1['DisplayName'] = expid
        e = func (ID='Experimenter:%s' % (expid), **d1)
        g = d['groups']
        
        e.append (ome_elem.GroupRef(ID='Group:%s' % g))
        yield e 
        
    def iter_Group(self, func):
        groupid = self.current_group
        d = groups[groupid]
        e = func(ID='Group:%s' % groupid, Name=d['name'])
        if 'descr' in d:
            e.append(ome_elem.Description (d['descr']))
        if 'contact_id' in d:
            e.append (ome_elem.Contact(ID='Experimenter:%s' % (d['contact_id'])))
        yield e
            
    def iter_Instrument(self,func):
        yield self.get_Instrument(func)
        
    def get_Instrument(self,func):
        if self.mode == 'MetaCyte TL':
            self.instrument = 'Bright-field Slide Scanner'
            scope = 'MetaCyteTL'
        elif self.mode == 'MetaCyte FL':
            self.instrument = 'Fluorescence Slide Scanner'
            scope = 'MetaCyteFL'
        ImarisInfo = self.slide.datasetinfo_imaris()
        e = func(ID='Instrument:0')
        e.append(ome_elem.Microscope(Manufacturer='%s' %ImarisInfo['ManufactorString'],
                                Model=self.instrument,Type='Upright'))    
        return e

            
   
