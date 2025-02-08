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
            return {'__class__':"numpy.ndarray", "data":base64.b64encode(v_bytes).decode('ascii')}
        elif isinstance(obj, complex):
            return {'__class__': "complex", "real": obj.real, "imag": obj.imag}
        elif isinstance(obj, datetime.datetime):
            return {'__class__': "datetime.datetime", "isoformat": obj.isoformat()}
        return super().default(obj)


def _custom_decode(val):
    if isinstance(val, str):
        if val.startswith("ndarray:"):
            print("decoding ndarray")
            dat = val.split(":")[1]
            dat = base64.b64decode(dat.encode('ascii'))
            temp = io.BytesIO(dat)
            val = np.load(temp)['arr']
            temp.close()
        elif val.startswith("complex:"):
            print("decoding complex")
            val = complex(val.split(":")[1])
        elif val.startswith("datetime:"):
            print("decoding datetime")
            val = datetime.datetime.fromisoformat(val.split(":")[1])
        else:
            print("doing nothing with", val)
    return val

class MTSimpleDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.from_dict, *args, **kwargs)

    @staticmethod
    def from_dict(d):
        if d.get("__class__") == "numpy.ndarray":
            dat = d["data"]
            dat = base64.b64decode(dat.encode('ascii'))
            temp = io.BytesIO(dat)
            arr = np.load(temp)['arr']
            temp.close()
            return arr
        elif d.get("__class__") == "complex":
            return d["real"] + 1j * d["imag"]
        elif d.get("__class__") == "datetime.datetime":
            return datetime.datetime.fromisoformat(d["isoformat"])
        return d

class SimpleBase:
    __slots__ = ['data', 'azimuth', 'zpk_filter', 'gain', 'start_time', 'sample_rate', 'time_delay']

    def __init__(
            self, data, azimuth, start_time: Union[str, datetime.datetime], sample_rate : float, zpk_filter=None, gain=None, time_delay=0.0
    ):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[None, :]
        self.data = data
        self.azimuth = azimuth
        if zpk_filter is None:
            zpk_filter = []
        self.zpk_filter = zpk_filter
        if gain is None:
            gain = []
        self.gain = gain
        if isinstance(start_time, str):
            start_time = datetime.datetime.fromisoformat(start_time)
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

    def validate_equal_configuration(self, other, ignore=None):
        my_attrs = self.__slots__

        if ignore is None:
            ignore = []

        for attr in my_attrs:
            if attr in ignore:
                continue
            v1 = getattr(self, attr)
            v2 = getattr(other, attr)
            if attr == 'data':
                if v1.shape != v2.shape:
                    raise ValueError("Data shapes not equal")
            elif v1 != v2:
                raise ValueError(f"{attr} values are not equal")
        return True

    @property
    def shape(self):
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

def geographic_orient(x1 : SimpleBase, x2: SimpleBase):
    if not (isinstance(x1, type(x2)) or isinstance(x2, type(x1))):
        raise TypeError("x1 and x2 must be of same type")
    x1.validate_equal_configuration(x2, ignore=["azimuth", "time_delay"])

    in_shape = x1.shape
    v_in = np.c_[x1.data.reshape(-1), x2.data.reshape(-1)]

    th = np.pi / 2 - x1.azimuth
    u1 = np.r_[np.cos(th), np.sin(th)]

    th = np.pi / 2 - x2.azimuth
    u2 = np.r_[np.cos(th), np.sin(th)]

    U = np.c_[u1, u2]

    r_east_north = np.linalg.solve(U, v_in[..., None])[..., 0]
    dat1 = r_east_north[..., 0].reshape(in_shape)
    dat2 = r_east_north[..., 1].reshape(in_shape)

    time_delay = 0.5 * (x1.time_delay + x2.time_delay)

    e = x1.update(data=dat1, azimuth=np.pi/2, time_delay=time_delay)
    n = x2.update(data=dat2, azimuth=0, time_delay=time_delay)

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

    def split(self, n_divisions):
        n_samples = self.data.size
        end = n_samples  - (n_samples % n_divisions)
        data = self.data.reshape(-1)[:end].reshape(n_divisions, -1)
        return self.update(data=data)

    def frequency_spectrum(self):
        data = self.data
        n_d = data.shape[-1]
        f_data = np.fft.rfft(data)[...,1:]
        freqs = np.fft.rfftfreq(n_d, self.sample_rate)[1:]
        f_data *= np.exp(-2j * np.pi * freqs * self.time_delay)

        # adjust columns of fft for phases based off of start times:
        n_wins = data.shape[0]
        dt_wins = n_wins / self.sample_rate
        window_starts = dt_wins * np.arange(n_wins)
        f_data *= np.exp(-2j * np.pi * freqs[None, :] * window_starts[:, None])

        pair_class = SPECTRUM_PAIRS[type(self)]
        spec = pair_class(
            f_data, self.azimuth, start_time=0, gain=self.gain,
            zpk_filter=self.zpk_filter, sample_rate=self.sample_rate,
        )
        return spec

class ElectricChannel(MTChannel):
    pass


class MagneticChannel(MTChannel):
    pass


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
        self.ex = ex
        self.ey = ey
        self.hx = hx
        self.hy = hy

    @property
    def channels(self):
        return [self.ex, self.ey, self.hx, self.hy]

    @property
    def start_time(self):
        return self.channels[0].start_time

    @property
    def sample_rate(self):
        return self.channels[0].sample_rate

    @property
    def shape(self):
        return (4, *self.channels[0].shape)

    @property
    def data_shape(self):
        return self.channels[0].shape

    def to_dict(self):
        return {
            "ex": self.channels[0].to_dict(),
            "ey": self.channels[1].to_dict(),
            "hx": self.channels[2].to_dict(),
            "hy": self.channels[3].to_dict(),
        }

    def to_json(self, file_name=None):
        self_dict = self.to_dict()
        if file_name is None:
            return json.dumps(self_dict, cls=MTSimpleEncoder)
        else:
            with open(file_name, "w") as f:
                json.dump(self_dict, f, cls=MTSimpleEncoder)

    @classmethod
    def from_json(cls, file_name):
        with open(file_name, 'r') as f:
            in_dict = json.load(f, cls=MTSimpleDecoder)

        ex = ElectricChannel(**in_dict["ex"])
        ey = ElectricChannel(**in_dict["ey"])
        hx = MagneticChannel(**in_dict["hx"])
        hy = MagneticChannel(**in_dict["hy"])

        return cls(ex=ex, ey=ey, hx=hx, hy=hy)

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

    def split(self, n_split):
        fs = []
        for channel in self.channels:
            fs.append(channel.split(n_split))
        return type(self)(*fs)

    def orient(self):
        ex, ey = geographic_orient(self.ex, self.ey)
        hx, hy = geographic_orient(self.hx, self.hy)
        return type(self)(ex, ey, hx, hy)

    def plot(self):
        for i, channel in enumerate(self.channels):
            plt.subplot(4, 1, i+1)
            channel.plot()



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

    def calculate_transfer_funcs(self, remote=None, period_bands=None):
        return calculate_Z(self, remote=remote, period_bands=period_bands)

    def orient(self):
        ex, ey = geographic_orient(*self.channels[:2])
        hx, hy = geographic_orient(*self.channels[2:])
        return type(self)(ex, ey, hx, hy)


def _band_sum(band, where):
    band = np.broadcast_to(band[..., None], (*band.shape, where.shape[-1]))
    return np.add.reduce(band, axis=-2, where=where)

def calculate_Z(site, remote=None, period_bands=None):
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

    if period_bands is not None:
        freq_bands = 1.0 / period_bands[::-1]
        lowers = freq_bands[:-1]
        uppers = freq_bands[1:]

        bands = (f[:, None] >= lowers) & (f[:, None] < uppers)

        f = _band_sum(f, bands)/np.sum(bands, axis=0)
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

SPECTRUM_PAIRS = {
    ElectricChannel:ElectricalSpectrum,
    MagneticChannel:MagneticSpectrum,
    ElectricalSpectrum:ElectricChannel,
    MagneticSpectrum:MagneticChannel,
}


def get_overlapping_series(
        x1: Union[MTChannel, MTTimeChannelCollection],
        x2: Union[MTChannel, MTTimeChannelCollection]
):
    if x1.sample_rate != x2.sample_rate:
        raise ValueError("x1 and x2 must have the same sample rate.")
    if x1.data_shape[:-1] != x2.data_shape[:-1]:
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

    n1 = x1.data_shape[-1] - x1_start_ind
    n2 = x2.data_shape[-1] - x2_start_ind
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
        dat1 = c1.data[..., x1_start_ind:x1_start_ind + n_new]
        dat2 = c2.data[..., x2_start_ind:x2_start_ind + n_new]

        new_c1s.append(c1.update(data=dat1, start_time=start))
        new_c2s.append(c2.update(data=dat2, start_time=start))
    if isinstance(x1, MTChannel):
        return new_c1s[0], new_c2s[0]
    else:
        new_collec1 = MTTimeChannelCollection(*new_c1s)
        new_collec2 = MTTimeChannelCollection(*new_c2s)
        return new_collec1, new_collec2