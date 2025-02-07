import datetime
import io
import json
from typing import Union
import base64
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sig
from scipy.constants import mu_0


class MTSimpleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            temp = io.BytesIO()
            np.savez(temp, arr=obj)
            v_bytes = temp.getvalue()
            temp.close()
            return base64.b64encode(v_bytes).decode('ascii')
        elif isinstance(obj, complex):
            return 'complex:'+str(obj)
        elif isinstance(obj, datetime.datetime):
            return 'datetime:'+obj.isoformat()
        return super().default(obj)

class MTSimpleDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict):
            new_obj = {}
            for key, val in obj.items():
                if isinstance(val, str):
                    if val.startswith("ndarray:"):
                        dat = val.split(":")[1]
                        dat = base64.b64decode(dat.encode('ascii'))
                        temp = io.BytesIO(dat)
                        val = np.load(temp)['arr']
                        temp.close()
                    elif val.startswith("complex:"):
                        val = complex(val[8:])
                    elif val.startswith("datetime:"):
                        val = datetime.datetime.fromisoformat(val[9:])
                new_obj[key] = val
            obj = new_obj
        return obj

class SimpleBase:
    __slots__ = ['data', 'azimuth', 'zpk_filter', 'gain', 'start_time', 'sample_rate', 'time_delay']

    def __init__(self, data, azimuth, zpk_filter, gain, start_time: datetime.datetime, sample_rate : float, time_delay=0):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[None, :]
        self.data = data
        self.azimuth = azimuth
        self.zpk_filter = zpk_filter
        self.gain = gain
        self.start_time = start_time
        self.sample_rate = sample_rate
        self.time_delay = time_delay

    def shallow_copy(self):
        attrs = {key:getattr(self, key) for key in self.__slots__}
        return type(self)(**attrs)

    def update(self, **kwargs):
        new = self.shallow_copy()
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new

    def assert_equal_configuration(self, other):
        my_attrs = self.__slots__
        other_attrs = other.__slots__

        same_attrs = my_attrs == other_attrs
        if same_attrs:
            for attr in my_attrs:
                v1 = getattr(self, attr)
                v2 = getattr(other, attr)
                if attr == 'data':
                    if v1.shape != v2.shape:
                        return False
                elif v1 != v2:
                    return False
            return True
        return False

    @property
    def data_shape(self):
        return self.data.shape

    def to_dict(self):
        self_dict = {}
        for attr in self.__slots__:
            self_dict[attr] = getattr(self, attr)
        return self_dict

    def to_json(self, file_name=None):
        self_dict = self.to_dict()
        if file_name is None:
            return json.dumps(self_dict, cls=MTSimpleEncoder)
        with open(file_name, 'w') as f:
            json.dump(self_dict, f, cls=MTSimpleEncoder)

    @classmethod
    def from_json_string(cls, str):
        vals = json.loads(str, cls=MTSimpleDecoder)
        return cls(**vals)

def geographic_orient(x1 : SimpleBase, x2: SimpleBase):
    # if not x1.assert_equal_configuration(x2):
    #     raise ValueError("x1 and x2 must have the same configuration.")
    data_shape = x1.data.shape
    v_in = np.c_[x1.data.reshape(-1), x2.data.reshape(-1)]

    th = np.pi / 2 - x1.azimuth
    u1 = np.r_[np.cos(th), np.sin(th)]

    th = np.pi / 2 - x2.azimuth
    u2 = np.r_[np.cos(th), np.sin(th)]

    U = np.c_[u1, u2]

    r_east_north = np.linalg.solve(U, v_in[..., None])[..., 0]
    dat1 = r_east_north[..., 0].reshape(data_shape)
    dat2 = r_east_north[..., 1].reshape(data_shape)

    e = x1.update(data=dat1)
    n = x2.update(data=dat2)

    return e, n


class MTChannel(SimpleBase):
    def detrend(self):
        data = self.data

        it = np.arange(data.shape[-1])
        p1, p0 = np.polyfit(it, data.T, 1)
        data = data - (p1[:, None] * it + p1[:, None])
        return self.update(data=data)

    def window(self):
        n_window = self.data.shape[-1]
        window = sig.windows.hamming(n_window)
        return self.update(data=self.data * window)

    def plot(self):
        plt.plot(self.data.T)


class ElectricChannel(MTChannel):

    def frequency_spectrum(self):
        f_data = np.fft.rfft(self.data)[...,1:]
        freqs = np.fft.rfftfreq(self.data.shape[-1], self.sample_rate)[1:]
        f_data *= np.exp(1j * 2 * np.pi * freqs * self.time_delay)
        spec = ElectricalSpectrum(f_data, self.azimuth, self.zpk_filter, self.gain, self.start_time, self.sample_rate, time_delay=0)
        return spec


class MagneticChannel(MTChannel):

    def frequency_spectrum(self):
        f_data = np.fft.rfft(self.data)[...,1:]
        freqs = np.fft.rfftfreq(self.data.shape[-1], self.sample_rate)[1:]
        f_data *= np.exp(1j * 2 * np.pi * freqs * self.time_delay)
        spec = MagneticSpectrum(f_data, self.azimuth, self.zpk_filter, self.gain, self.start_time, self.sample_rate, time_delay=0)
        return spec


class MTSpectrum(SimpleBase):

    @property
    def frequencies(self):
        return np.fft.rfftfreq(self.data.shape[-1]*2+1, self.sample_rate)[1:]

    @property
    def omegas(self):
        return 2 * np.pi * self.frequencies

    def calibrate(self):
        f_data = self.data.copy()

        f_data /= np.prod(self.gain) # flexibly handle gain as a list of gains

        zpks = self.zpk_filter
        if not isinstance(self.zpk_filter, (list, tuple, set)):
            zpks = [zpks]

        oms = self.omegas
        for zpk in zpks:
            _, fc = sig.freqs_zpk(**zpk, worN=oms)
            f_data /= fc

        return self.update(data=f_data, gain=[], zpk_filter=[])

    def plot(self):
        plt.loglog(self.frequencies, np.abs(self.data.T))


class ElectricalSpectrum(MTSpectrum):
    pass


class MagneticSpectrum(MTSpectrum):
    pass


class MTTimeChannelCollection():

    def __init__(self, ex : MTChannel, ey : MTChannel, hx : MTChannel, hy : MTChannel):
        channels = [ex, ey, hx, hy]
        for channel in channels:
            if channel.start_time != ex.start_time:
                raise TypeError("Channels must have the same start time.")
            elif channel.sample_rate != ex.sample_rate:
                raise TypeError("Channels must have the same sample rate.")
            elif channel.data.shape != ex.data.shape:
                raise TypeError("Channels must have the number of data.")
        self.channels = channels

    @property
    def start_time(self):
        return self.channels[0].start_time

    @property
    def sample_rate(self):
        return self.channels[0].sample_rate

    @property
    def data_shape(self):
        return self.channels[0].data.shape

    def to_json(self, file_name=None):
        out = []
        for i_c, channel in enumerate(self.channels):
            out.append((type(channel).__name__, channel.to_json()))
        if file_name is None:
            return json.dumps(out)
        with open(file_name, 'w') as f:
            json.dump(out, f, cls=MTSimpleEncoder)

    @classmethod
    def from_json(cls, file_name):
        with open(file_name, 'r') as f:
            channel_list = json.load(f)
        ch_list = []
        for channel_dat in channel_list:
            cls = CLASS_NAME_TO_CLASS[channel_dat[0]]
            ch_list.append(cls.from_json_string(channel_dat[1]))

        return cls(*ch_list)

    def frequency_spectrum(self):
        fs = []
        for channel in self.channels:
            fs.append(channel.frequency_spectrum())
        return MTFrequencySpectrumCollection(*fs)

    def detrend(self):
        fs = []
        for channel in self.channels:
            fs.append(channel.detrend())
        return type(self)(*fs)

    def window(self):
        fs = []
        for channel in self.channels:
            fs.append(channel.window())
        return type(self)(*fs)


class MTFrequencySpectrumCollection():

    def __init__(self, ex : MTChannel, ey : MTChannel, hx : MTChannel, hy : MTChannel):
        channels = [ex, ey, hx, hy]
        for channel in channels:
            if channel.start_time != ex.start_time:
                raise TypeError("Channels must have the same start time.")
            elif channel.sample_rate != ex.sample_rate:
                raise TypeError("Channels must have the same sample rate.")
            elif channel.data.shape != ex.data.shape:
                raise TypeError("Channels must have the number of data.")
        self.channels = channels

    def calibrate(self):
        channels = []
        for channel in self.channels:
            channels.append(channel.calibrate())
        return type(self)(*channels)

def _band_sum(band, where):
    band = np.broadcast_to(band[..., None], (*band.shape, where.shape[-1]))
    return np.add.reduce(band, axis=-2, where=where)

def calculate_Z(site, remote=None, bands=None):
    if remote is None:
        remote = site
    Rxc = remote.channels[2].data.conjugate()
    Ryc = remote.channels[3].data.conjugate()

    Ex, Ey, Hx, Hy = site.channels
    Ex = Ex.data
    Ey = Ey.data
    Hx = Hx.data
    Hy = Hy.data

    f = site.channels[0].frequencies

    if bands is not None:
        f = _band_sum(f, bands)
        func = lambda x : np.sum(_band_sum(x, where=bands), axis=0)
    else:
        func = lambda x : np.sum(x, axis=0)

    bot = func(Hx * Rxc) * func(Hy * Ryc) - func(Hx * Ryc) * func(Hy * Rxc)

    top_xx = func(Ex * Rxc) * func(Hy * Ryc) - func(Ex * Ryc) * func(Hy * Rxc)
    top_xy = func(Ex * Ryc) * func(Hx * Rxc) - func(Ex * Rxc) * func(Hx * Ryc)

    top_yx = func(Ey * Rxc) * func(Hy * Ryc) - func(Ey * Ryc) * func(Hy * Rxc)
    top_yy = func(Ey * Ryc) * func(Hx * Rxc) - func(Ey * Rxc) * func(Hx * Ryc)

    zx = np.stack([top_xx, top_xy], axis=-1)
    zy = np.stack([top_yx, top_yy], axis=-1)

    Z = np.stack([zx, zy], axis=1) / bot[:, None, None]

    return f, Z

def calc_rho_a(f, Z):
    
    om = 2 * np.pi * f
    rho_a = 1/(mu_0 * om[:, None, None]) * np.abs(Z)**2
    return rho_a

CLASS_NAME_TO_CLASS = {
    "SimpleBase":SimpleBase,
    "MTChannel":MTChannel,
    "ElectricChannel":ElectricChannel,
    "MagneticChannel":MagneticChannel,
    "MTSpectrum":MTSpectrum,
    "ElectricalSpectrum":ElectricalSpectrum,
    "MagneticSpectrum":MagneticSpectrum,
}


def get_overlapping_series(
        x1: Union[MTChannel, MTTimeChannelCollection],
        x2: Union[MTChannel, MTTimeChannelCollection]
):
    if x1.sample_rate != x2.sample_rate:
        raise ValueError("x1 and x2 must have the same sample rate.")
    if x1.data.shape[:-1] != x2.data.shape[:-1]:
        raise ValueError("x1 and x2 must have the same shape up to the last dimension.")
    if isinstance(x1, MTChannel) and not isinstance(x2, MTChannel):
        raise ValueError("x1 and x2 must be both MTChannel.")
    if isinstance(x1, MTTimeChannelCollection) and not isinstance(x2, MTTimeChannelCollection):
        raise ValueError("x1 and x2 must be MTTimeChannelCollection.")

    start = max(x1.start_time, x2.start_time)
    d1 = start - x1.start_time
    d2 = start - x2.start_time
    x1_start_ind = int(d1.total_seconds() * x1.sample_rate)
    x2_start_ind = int(d2.total_seconds() * x2.sample_rate)

    n1 = x1.data.shape[-1] - x1_start_ind
    n2 = x2.data.shape[-1] - x2_start_ind
    n_new = min(n1, n2)
    if isinstance(x1, MTChannel):
        channels1 = [x1]
    else:
        channels1 = x1.channels

    if isinstance(x2, MTChannel):
        channels2 = [x2]
    else:
        channels2 = x2.channels

    new_c1s = []
    new_c2s = []
    for c1, c2 in zip(channels1, channels2):
        dat1 = c1.data[x1_start_ind:x1_start_ind + n_new]
        dat2 = c1.data[x2_start_ind:x2_start_ind + n_new]

        new_c1s.append(c1.update(data=dat1, start_time=start))
        new_c2s.append(c2.update(data=dat2, start_time=start))
    if isinstance(x1, MTChannel):
        return new_c1s[0], new_c2s[0]
    else:
        new_collec1 = MTTimeChannelCollection(new_c1s)
        new_collec2 = MTTimeChannelCollection(new_c2s)
        return new_collec1, new_collec2