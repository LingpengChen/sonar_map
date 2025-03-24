import numpy as np
import math
from marine_acoustic_msgs.msg import ProjectedSonarImage

class AzimuthRangeIndices:
    def __init__(self, a, r):
        self._azimuthIdx = a
        self._rangeIdx = r
        
    def azimuth(self):
        return self._azimuthIdx
        
    def range_idx(self):
        return self._rangeIdx

class AbstractSonarInterface:
    def __init__(self):
        self._rangeBounds = None
        self._azimuthBounds = None
        self._minAzimuthTan = None
        self._maxAzimuthTan = None
        self._maxRangeSquared = None
    
    def data_type(self):
        raise NotImplementedError
    
    def azimuths(self):
        raise NotImplementedError
    
    def ranges(self):
        raise NotImplementedError
    
    def nBearings(self):
        return len(self.azimuths())
    
    def nAzimuths(self):
        return len(self.azimuths())
    
    def nRanges(self):
        return len(self.ranges())
    
    def azimuthBounds(self):
        if self._azimuthBounds is None:
            azimuths = self.azimuths()
            if azimuths:
                self._azimuthBounds = (min(azimuths), max(azimuths))
                self._minAzimuthTan = math.tan(self._azimuthBounds[0])
                self._maxAzimuthTan = math.tan(self._azimuthBounds[1])
            else:
                self._azimuthBounds = (0, 0)
                self._minAzimuthTan = 0
                self._maxAzimuthTan = 0
        return self._azimuthBounds
    
    def rangeBounds(self):
        if self._rangeBounds is None:
            ranges = self.ranges()
            if ranges:
                self._rangeBounds = (min(ranges), max(ranges))
                self._maxRangeSquared = self._rangeBounds[1] ** 2
            else:
                self._rangeBounds = (0, 0)
                self._maxRangeSquared = 0
        return self._rangeBounds
    
    def minAzimuth(self):
        return self.azimuthBounds()[0]
    
    def maxAzimuth(self):
        return self.azimuthBounds()[1]
    
    def minRange(self):
        return self.rangeBounds()[0]
    
    def maxRange(self):
        return self.rangeBounds()[1]
    
    def intensity_float(self, idx):
        raise NotImplementedError
    
    def intensity_uint8(self, idx):
        return int(255 * self.intensity_float(idx))
    
    def intensity_uint16(self, idx):
        return int(65535 * self.intensity_float(idx))
    
    def intensity_uint32(self, idx):
        return int(4294967295 * self.intensity_float(idx))


class SonarImageMsgInterface(AbstractSonarInterface):
    TYPE_NONE = 0
    TYPE_UINT8 = 1
    TYPE_UINT16 = 2
    TYPE_UINT32 = 3
    TYPE_FLOAT32 = 4
    
    def __init__(self, ping: ProjectedSonarImage):
        super().__init__()
        self._ping = ping
        
        # Assuming ping has an attribute tx_beamwidths and beam_directions
        self._verticalTanSquared = math.pow(math.tan(ping.ping_info.tx_beamwidths[0] / 2.0), 2)
        
        self._ping_azimuths = []
        for pt in ping.beam_directions:
            az = math.atan2(-1 * pt.y, pt.z)
            self._ping_azimuths.append(az)
        
        self.do_log_scale_ = False
        self.min_db_ = 0
        self.max_db_ = 0
        self.range_db_ = 0
    
    def do_log_scale(self, min_db, max_db):
        self.do_log_scale_ = True
        self.min_db_ = min_db
        self.max_db_ = max_db
        self.range_db_ = max_db - min_db
    
    def data_type(self):
        if self._ping.image.dtype == self._ping.image.DTYPE_UINT8:
            return self.TYPE_UINT8
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT16:
            return self.TYPE_UINT16
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT32:
            return self.TYPE_UINT32
        return self.TYPE_NONE
    
    def ranges(self):
        return self._ping.ranges
    
    def azimuths(self):
        return self._ping_azimuths
    
    def verticalTanSquared(self):
        return self._verticalTanSquared
    
    def index(self, idx:AzimuthRangeIndices):
        data_size = 0
        if self._ping.image.dtype == self._ping.image.DTYPE_UINT8:
            data_size = 1
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT16:
            data_size = 2
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT32:
            data_size = 4
        else:
            assert False
        
        return data_size * ((idx.range_idx() * self.nBearings()) + idx.azimuth())
    
    def read_uint8(self, idx):
        assert self._ping.image.dtype == self._ping.image.DTYPE_UINT8
        i = self.index(idx)
        return self._ping.image.data[i]
    
    def read_uint16(self, idx):
        assert self._ping.image.dtype == self._ping.image.DTYPE_UINT16
        i = self.index(idx)
        return (self._ping.image.data[i] | 
                (self._ping.image.data[i + 1] << 8))
    
    def read_uint32(self, idx):
        assert self._ping.image.dtype == self._ping.image.DTYPE_UINT32
        i = self.index(idx)
        return (self._ping.image.data[i] | 
                (self._ping.image.data[i + 1] << 8) |
                (self._ping.image.data[i + 2] << 16) |
                (self._ping.image.data[i + 3] << 24))
    
    def intensity_float_log(self, idx):
        intensity = self.read_uint32(idx)
        intensity = max(1, intensity)
        v = math.log(float(intensity) / 0xFFFFFFFF) * 10  # dbm
        
        min_db = self.min_db_ if self.min_db_ != 0 else math.log(1.0 / 0xFFFFFFFF) * 10
        
        return min(1.0, max(0.0, (v - min_db) / self.range_db_))
    
    def intensity_uint8(self, idx):
        if self.do_log_scale_ and (self._ping.image.dtype == self._ping.image.DTYPE_UINT32):
            return int(self.intensity_float_log(idx) * 0xFF)
        
        if self._ping.image.dtype == self._ping.image.DTYPE_UINT8:
            return self.read_uint8(idx)
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT16:
            return self.read_uint16(idx) >> 8
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT32:
            return self.read_uint32(idx) >> 24
        
        return 0
    
    def intensity_uint16(self, idx):
        if self.do_log_scale_ and (self._ping.image.dtype == self._ping.image.DTYPE_UINT32):
            return int(self.intensity_float_log(idx) * 0xFFFF)
        
        if self._ping.image.dtype == self._ping.image.DTYPE_UINT8:
            return self.read_uint8(idx) << 8
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT16:
            return self.read_uint16(idx)
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT32:
            return self.read_uint32(idx) >> 16
        
        return 0
    
    def intensity_uint32(self, idx):
        if self.do_log_scale_ and (self._ping.image.dtype == self._ping.image.DTYPE_UINT32):
            return int(self.intensity_float_log(idx) * 0xFFFFFFFF)
        
        if self._ping.image.dtype == self._ping.image.DTYPE_UINT8:
            return self.read_uint8(idx) << 24
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT16:
            return self.read_uint16(idx) << 16
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT32:
            return self.read_uint32(idx)
        
        return 0
    
    def intensity_float(self, idx):
        if self.do_log_scale_ and (self._ping.image.dtype == self._ping.image.DTYPE_UINT32):
            return self.intensity_float_log(idx)
        
        if self._ping.image.dtype == self._ping.image.DTYPE_UINT8:
            return float(self.read_uint8(idx)) / 0xFF # normalize to 0~1
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT16:
            return float(self.read_uint16(idx)) / 0xFFFF
        elif self._ping.image.dtype == self._ping.image.DTYPE_UINT32:
            return float(self.read_uint32(idx)) / 0xFFFFFFFF
        
        return 0.0

# Usage example:
# interface = SonarImageMsgInterface(sonar_msg)