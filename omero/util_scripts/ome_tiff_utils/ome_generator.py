import os
import sys
from uuid import uuid1 as uuid
from lxml import etree
from lxml.builder import ElementMaker
from libtiff import TIFF
import numpy as np

namespace_map=dict(bf = "http://www.openmicroscopy.org/Schemas/BinaryFile/2010-06",
                   ome = "http://www.openmicroscopy.org/Schemas/OME/2010-06",
                   xsi = "http://www.w3.org/2001/XMLSchema-instance",
                   sa = "http://www.openmicroscopy.org/Schemas/SA/2010-06",
                   spw = "http://www.openmicroscopy.org/Schemas/SPW/2010-06")

# create element makers: bf, ome, xsi
default_validate = False
if default_validate:
    # use this when validating
    ome_elem = ElementMaker (namespace = namespace_map['ome'], nsmap = namespace_map) 
else:
    # use this for creating imagej readable ome.tiff files.
    ome_elem = ElementMaker (nsmap = namespace_map) 

bf = ElementMaker (namespace = namespace_map['bf'], nsmap = namespace_map)
sa = ElementMaker (namespace = namespace_map['sa'], nsmap = namespace_map)
spw = ElementMaker (namespace = namespace_map['spw'], nsmap = namespace_map)

def ATTR(namespace, name, value):
    return {'{%s}%s' % (namespace_map[namespace], name): value}

class FileSizeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)  

def validate_xml(xml):
    if getattr(sys,'frozen',None):
        ome_xsd_path = os.path.dirname(sys.executable)
    elif __file__:  
        ome_xsd_path = os.path.dirname(__file__)
        
    ome_xsd = os.path.join(ome_xsd_path,'ome.xsd')    

    if os.path.isfile (ome_xsd):
        ome_xsd = os.path.join(namespace_map['ome'],'ome.xsd')
        f = open (ome_xsd) 
    else:
        import urllib2
        ome_xsd = os.path.join(namespace_map['ome'],'ome.xsd')
        f = urllib2.urlopen(ome_xsd)
    sys.stdout.write('Validating XML content against %r...' % (ome_xsd))
    xmlschema_doc = etree.parse(f)
    
    xmlschema = etree.XMLSchema(xmlschema_doc)
    if isinstance (xml, basestring):
        xml = etree.parse(xml)
    result = xmlschema.validate(xml)
    if not result:
        sys.stdout.write('FAILED:\n')
        for error in xmlschema.error_log:
            s = str (error)
            for k,v in namespace_map.items():
                s = s.replace ('{%s}' % v, '%s:' % k)
        sys.stdout.write('-----\n')
    else:
        sys.stdout.write('SUCCESS!\n')
    return result

class ElementBase:

    def __init__ (self, parent, root):
        self.parent = parent
        self.root = root
        
        n = self.__class__.__name__
        iter_mth = getattr(parent, 'iter_%s' % (n), None)
        nsn = 'ome_elem'
        nm = n
        if '_' in n:
            nsn, nm = n.split('_',1)
            nsn = nsn.lower()
        ns = eval(nsn)    
        ome_el = getattr (ns, nm, None)

        if iter_mth is not None:
            for element in iter_mth(ome_el):
                root.append(element)
        elif 0:
            print 'NotImplemented: %s.iter_%s(<%s.%s callable>)' % (parent.__class__.__name__, n, nsn, nm)

class TiffImageGenerator:
    
    def __init__(self,instrument,filename,input_data,scalefact):
        self.instrument = instrument
        self.filename = filename
        self.scale = scalefact
        self.data = input_data
        
    def create_tiles(self,roi,sizeX, sizeY, sizeZ, sizeC, sizeT, tileWidth, tileHeight, description):
        tif_image = TIFF.open(self.filename, 'w')
        tile_count = 0
        for c in range(0, sizeC):
            if c == 0:
                tif_image.set_description(description)
                
            tif_image.tile_image_params(sizeX,sizeY,sizeC,tileWidth,tileHeight)
            
            for tileOffsetY in range(
                    0, ((sizeY + tileHeight - 1) / tileHeight)):

                for tileOffsetX in range(
                        0, ((sizeX + tileWidth - 1) / tileWidth)):

                    x = tileOffsetX * tileWidth
                    y = tileOffsetY * tileHeight
                    w = tileWidth

                    if (w + x > sizeX):
                        w = sizeX - x

                    h = tileHeight
                    if (h + y > sizeY):
                        h = sizeY - y

                    tile_count += 1
                    tile_data = self.mktile(roi,c,x,y,w,h)
                    tile_dtype = tile_data.dtype
                    tile = np.zeros((1,tileWidth,tileHeight),dtype=tile_dtype)
                    tile[0,:h,:w] = tile_data[0,:,:]                     
                    tif_image.write_tile(tile,x,y)
                    
            tif_image.WriteDirectory()
        tif_image.close()
        return tile_count
    
    def mktile(self,roi,channel,x,y,w,h):
        row_start = y + roi[1]
        row_end = row_start + h
        col_start = x + roi[0]
        col_end = col_start + w
        roi = [row_start,row_end,col_start,col_end]
        tile_data = self.data.get_data_in_channel(self.scale,channel,roi)
        return tile_data
        
    def create_plane(self,roi,sizeX,sizeY,sizeC,description):
        tif_image = TIFF.open(self.filename, 'w')
        im_dtype = np.dtype('uint8')
        image_data = np.zeros((sizeC,sizeY,sizeX),dtype=im_dtype)
        print 'num channels=',sizeC
        for c in range(sizeC):
            if c == 0:
                tif_image.set_description(description)
            imarray = self.mkplane(roi,c)
            print 'imarray shape:',imarray.shape
            
            print("Writing channel:  ", c+1)
            plane = imarray[0,:,:]
                
            image_data[c,:,:] = plane    
#        tif_image = TIFFimage(image_data,description=description)
#        tif_image.write_file(self.filename,compression='lzw') 
#        del tif_image  
        tif_image.write_image(image_data, compression='lzw')
        tif_image.close()
                        
    def mkplane(self,roi,channel):
        return self.data.get_data_in_channel(self.scale,channel,roi)

class Dataset(ElementBase): pass            
class Group(ElementBase): pass
class Experimenter(ElementBase): pass
class Instrument(ElementBase): pass
class Image(ElementBase): pass

class OMEBase:
    """ Base class for OME-XML writers.
    """

    _subelement_classes = [Dataset, Experimenter, Group, Instrument, Image]

    prefix = ''
    def __init__(self):
        self.tif_images = {}
#        self.cwd = os.path.abspath(os.getcwd())
#        self.output_prefix = os.path.join(self.cwd, self.prefix)
#        if not os.path.exists (self.output_prefix):
#            os.makedirs(self.output_prefix)
#        self.file_prefix = os.path.join(self.output_prefix,'')

    def process(self, options=None, validate=default_validate):
        template_xml = list(self.make_xml())
        tif_gen = TiffImageGenerator(self.instrument,self.tif_filename,self.slide,self.scalefact)
        self.tif_images[self.instrument,self.tif_filename,self.tif_uuid,self.PhysSize] = tif_gen

        s = None
        for (detector, fn, uuid, res), tif_gen in self.tif_images.items():
            xml= ome_elem.OME(ATTR('xsi','schemaLocation',"%s %s/ome.xsd" % ((namespace_map['ome'],)*2)),
                          UUID = uuid)
            for item in template_xml:

                if item.tag.endswith('Image') and item.get('ID')!='Image:%s' % (detector):
                    continue
                xml.append(item)
                
            if s is None and validate:
                s = etree.tostring(xml, encoding='UTF-8', xml_declaration=True)
                validate_xml(xml)
            else:
                s = etree.tostring(xml, encoding='UTF-8', xml_declaration=True)
            
            if (self.sizeX < 4096) or (self.sizeY < 4096):
                tif_gen.create_plane(self.roi,self.sizeX,self.sizeY,self.sizeC,s)
            else:
                tc = tif_gen.create_tiles(self.roi,self.sizeX, self.sizeY, self.sizeZ, self.sizeC, self.sizeT, self.tile_width, self.tile_height, s)
                print 'tile count=',tc
            print 'SUCCESS!'

        return s

    def _mk_uuid(self):
        return 'urn:uuid:%s' % (uuid())

    def make_xml(self):
        self.temp_uuid = self._mk_uuid()
        xml = ome_elem.OME(ATTR('xsi','schemaLocation',"%s %s/ome.xsd" % ((namespace_map['ome'],)*2)),
                       UUID = self.temp_uuid)
        for element_cls in self._subelement_classes:
            element_cls(self, xml) # element_cls should append elements to root
        return xml   

    def get_AcquiredDate(self):
        return None

    @staticmethod
    def dtype2PixelIType(dtype):
        return dict (int8='int8',int16='int16',int32='int32',
                     uint8='uint8',uint16='uint16',uint32='uint32',
                     complex128='double-complex', complex64='complex',
                     float64='double', float32='float',
                     ).get(dtype.name, dtype.name)
