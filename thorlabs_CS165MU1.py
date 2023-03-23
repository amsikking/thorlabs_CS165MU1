import time
import ctypes as C
import os
import numpy as np

class Camera:
    '''
    Basic device adaptor for Thorlabs CS165MU1 Zelux 1.6 MP Monochrome
    CMOS Camera. Many more commands are available and have not been
    implemented.
    '''
    def __init__(self, name='CS165MU1', verbose=True, very_verbose=False):
        self.name = name
        self.verbose = verbose
        self.very_verbose = very_verbose
        if self.verbose:
            print("%s: opening..."%self.name)
        # find cameras:
        dll.open_sdk()
        _STRING_MAX = 4096
        serial_number_bytes = C.c_char_p(_STRING_MAX * b' ')
        dll.find_cameras(serial_number_bytes, _STRING_MAX)
        self.list = serial_number_bytes.value.decode('ascii').split()
        self.handles = []
        for camera in self.list:
            handle = C.c_void_p(0)
            dll.open_camera(bytes(camera, 'ascii'), handle)
            self.handles.append(handle.value)
        self.channels = tuple(range(len(self.handles)))
        if self.very_verbose:
            print("%s: camera list     = %s"%(self.name, self.list))
            print("%s: camera handles  = %s"%(self.name, self.handles))
            print("%s: camera channels = %s"%(self.name, self.channels))
        # get camera attributes and set defaults:
        self.exposure_min_us =          [None] * len(self.channels)
        self.exposure_max_us =          [None] * len(self.channels)
        self.exposure_us =              [None] * len(self.channels)
        self.read_time_us =             [None] * len(self.channels)
        self.frame_time_us =            [None] * len(self.channels)
        self.roi_range_px =             [None] * len(self.channels)
        self.roi_px =                   [None] * len(self.channels)
        self.height_px =                [None] * len(self.channels)
        self.width_px =                 [None] * len(self.channels)
        self.poll_timeout_ms =          [None] * len(self.channels)
        self.frames_per_trigger_min =   [None] * len(self.channels)
        self.frames_per_trigger_max =   [None] * len(self.channels)
        self.frames_per_trigger =       [None] * len(self.channels)
        self.trigger_mode =             [None] * len(self.channels)
        self.num_images =               [None] * len(self.channels)
        self._frame_count =             [None] * len(self.channels)
        self._num_buffers = []
        self._armed = []
        self.latency_us = 10000 # what is the pc/connection latency? ~ 10ms?
        for ch in self.channels:
            self._get_exposure_time_range_us(ch)
            self._get_exposure_time_us(ch)
            self._get_roi_range(ch)
            self._get_roi(ch)
            self._get_poll_timeout_ms(ch)
            self._get_frames_per_trigger_range(ch)
            self._set_frames_per_trigger(ch, 1)     # default 1 frame
            self._set_trigger_mode(ch, 'software')  # default software
            self._num_buffers.append(2)             # default 2 buffers (min)
            self._armed.append(False)               # default disarmed
            self._disarm(ch)                        # apply disarm
        if self.verbose:
            print("%s: -> open and ready."%self.name)

    def _get_exposure_time_range_us(self, ch):
        if self.very_verbose:
            print("%s(%i): getting exposure time range (us)"%(self.name, ch))
        exposure_min_us = C.c_longlong()
        exposure_max_us = C.c_longlong()
        dll.get_exposure_time_range_us(
            self.handles[ch], exposure_min_us, exposure_max_us)
        self.exposure_min_us[ch] = exposure_min_us.value
        self.exposure_max_us[ch] = exposure_max_us.value
        if self.very_verbose:
            print("%s(%i): -> exposure_min_us = %09i"%(
                        self.name, ch, self.exposure_min_us[ch]))
            print("%s(%i): -> exposure_max_us = %09i"%(
                        self.name, ch, self.exposure_max_us[ch]))
        return self.exposure_min_us[ch], self.exposure_max_us[ch]

    def _get_exposure_time_us(self, ch):
        if self.very_verbose:
            print("%s(%i): getting exposure time (us)"%(self.name, ch))
        exposure_us = C.c_longlong()
        dll.get_exposure_time_us(self.handles[ch], exposure_us)
        self.exposure_us[ch] = exposure_us.value
        self._get_frame_time_us(ch) # exposure_us changes the frame_time_us!
        if self.very_verbose:
            print("%s(%i): -> exposure_us = %09i"%(
                self.name, ch, self.exposure_us[ch]))
        return self.exposure_us[ch]

    def _set_exposure_time_us(self, ch, exposure_us): # min->max in ~25us steps
        if self.very_verbose:
            print("%s(%i): setting exposure time (us) = %08i"%(
                self.name, ch, exposure_us))
        assert type(exposure_us) is int
        assert exposure_us >= self.exposure_min_us[ch], (
            'exposure %i out of range'%exposure_us)
        assert exposure_us <= self.exposure_max_us[ch], (
            'exposure %i out of range'%exposure_us)
        dll.set_exposure_time_us(self.handles[ch], exposure_us)
        self._get_exposure_time_us(ch)
        assert exposure_us - 25 <= self.exposure_us[ch] <= exposure_us + 25
        if self.very_verbose:
            print("%s(%i): -> done setting exposure time."%(self.name, ch))
        return None

    def _get_read_time_us(self, ch):
        if self.very_verbose:
            print("%s(%i): getting read time (us)"%(self.name, ch))
        read_time_ns = C.c_int()
        dll.get_read_time_ns(self.handles[ch], read_time_ns)
        self.read_time_us[ch] = int(round(1e-3 * read_time_ns.value))
        if self.very_verbose:
            print("%s(%i): -> read_time_us = %09i"%(
                self.name, ch, self.read_time_us[ch]))
        return self.read_time_us[ch]

    def _get_frame_time_us(self, ch):
        if self.very_verbose:
            print("%s(%i): getting frame time (us)"%(self.name, ch))
        frame_time_us = C.c_int()
        dll.get_frame_time_us(self.handles[ch], frame_time_us)
        self.frame_time_us[ch] = frame_time_us.value
        if self.very_verbose:
            print("%s(%i): -> frame_time_us = %09i"%(
                self.name, ch, self.frame_time_us[ch]))
        return self.frame_time_us[ch]

    def _get_roi_range(self, ch):
        if self.very_verbose:
            print("%s(%i): getting roi range"%(self.name, ch))
        l_px_min, l_px_max = C.c_int(), C.c_int()
        r_px_min, r_px_max = C.c_int(), C.c_int()
        u_px_min, u_px_max = C.c_int(), C.c_int()
        d_px_min, d_px_max = C.c_int(), C.c_int()
        dll.get_roi_range(self.handles[ch],
                          l_px_min, u_px_min, r_px_min, d_px_min,
                          l_px_max, u_px_max, r_px_max, d_px_max)
        self.roi_range_px[ch] = {'left' :(l_px_min.value, l_px_max.value),
                                 'right':(r_px_min.value, r_px_max.value),
                                 'up'   :(u_px_min.value, u_px_max.value),
                                 'down' :(d_px_min.value, d_px_max.value)}
        if self.very_verbose:
            print("%s(%i): -> roi_range ="%(self.name, ch))
            print("%s(%i): -> left: %s"%(
                self.name, ch, self.roi_range_px[ch]['left']))
            print("%s(%i): -> right: %s"%(
                self.name, ch, self.roi_range_px[ch]['right']))
            print("%s(%i): -> up: %s"%(
                self.name, ch, self.roi_range_px[ch]['up']))
            print("%s(%i): -> down: %s"%(
                self.name, ch, self.roi_range_px[ch]['down']))
        return self.roi_range_px[ch]

    def _get_roi(self, ch):
        if self.very_verbose:
            print("%s(%i): getting roi"%(self.name, ch))
        l_px, r_px, u_px, d_px = C.c_int(), C.c_int(), C.c_int(), C.c_int()
        dll.get_roi(self.handles[ch], l_px, u_px, r_px, d_px)
        self.roi_px[ch] = {'left' :l_px.value,
                           'right':r_px.value,
                           'up'   :u_px.value,
                           'down' :d_px.value}
        self.height_px[ch] = (
            self.roi_px[ch]['down'] - self.roi_px[ch]['up'] + 1)
        self.width_px[ch] =  (
            self.roi_px[ch]['right'] - self.roi_px[ch]['left'] + 1)
        self._get_read_time_us(ch)  # ROI changes the read_time_us!
        self._get_frame_time_us(ch) # ROI changes the frame_time_us!
        if self.very_verbose:
            print("%s(%i): -> roi = %s"%(self.name, ch, self.roi_px[ch]))
        return self.roi_px[ch]

    def _set_roi(self, ch, roi_px):
        if self.very_verbose:
            print("%s(%i): setting roi = %s"%(self.name, ch, roi_px))
        l_px, r_px = roi_px['left'], roi_px['right']
        u_px, d_px = roi_px['up'],   roi_px['down']
        dll.set_roi(self.handles[ch], l_px, u_px, r_px, d_px)
        assert self._get_roi(ch) == roi_px
        if self.very_verbose:
            print("%s(%i): -> done setting roi."%(self.name, ch))
        return None

    # This seems unsupported
    def _get_poll_timeout_ms(self, ch):
        if self.very_verbose:
            print("%s(%i): getting poll timeout (ms)"%(self.name, ch))
        poll_timeout_ms = C.c_int()
        dll.get_poll_timeout_ms(self.handles[ch], poll_timeout_ms)
        self.poll_timeout_ms[ch] = poll_timeout_ms.value
        if self.very_verbose:
            print("%s(%i): -> poll_timeout_ms = %09i"%(
                self.name, ch, self.poll_timeout_ms[ch]))
        return self.poll_timeout_ms[ch]

    # This seems unsupported
    def _set_poll_timeout_ms(self, ch, poll_timeout_ms):
        if self.very_verbose:
            print("%s(%i): setting poll timeout (ms) = %08i"%(
                self.name, ch, poll_timeout_ms))
        poll_timeout_ms = C.c_int()
        dll.set_poll_timeout_ms(self.handles[ch], poll_timeout_ms)
        assert self._get_poll_timeout_ms(ch) == poll_timeout_ms
        if self.very_verbose:
            print("%s(%i): -> done setting poll timeout."%(self.name, ch))
        return None

    def _get_frames_per_trigger_range(self, ch):
        if self.very_verbose:
            print("%s(%i): getting frames per trigger range"%(self.name, ch))
        frames_per_trigger_min = C.c_uint()
        frames_per_trigger_max = C.c_uint()
        dll.get_frames_per_trigger_range(
            self.handles[ch], frames_per_trigger_min, frames_per_trigger_max)
        self.frames_per_trigger_min[ch] = frames_per_trigger_min.value
        self.frames_per_trigger_max[ch] = frames_per_trigger_max.value
        if self.very_verbose:
            print("%s(%i): -> frames_per_trigger_min = %012i"%(
                        self.name, ch, self.frames_per_trigger_min[ch]))
            print("%s(%i): -> frames_per_trigger_max = %012i"%(
                        self.name, ch, self.frames_per_trigger_max[ch]))
        return self.frames_per_trigger_min[ch], self.frames_per_trigger_max[ch]

    def _get_frames_per_trigger(self, ch):
        if self.very_verbose:
            print("%s(%i): getting frames per trigger"%(self.name, ch))
        frames_per_trigger = C.c_uint()
        dll.get_frames_per_trigger(self.handles[ch], frames_per_trigger)
        self.frames_per_trigger[ch] = frames_per_trigger.value
        if self.very_verbose:
            print("%s(%i): -> frames_per_trigger = %012i"%(
                        self.name, ch, self.frames_per_trigger[ch]))
        return self.frames_per_trigger[ch]

    def _set_frames_per_trigger(self, ch, frames_per_trigger): # 0 = unlimited
        if self.very_verbose:
            print("%s(%i): setting frames per trigger = %012i"%(
                self.name, ch, frames_per_trigger))
        assert type(frames_per_trigger) is int
        assert frames_per_trigger >= self.frames_per_trigger_min[ch], (
            'frames_per_trigger %i out of range'%frames_per_trigger)
        assert frames_per_trigger <= self.frames_per_trigger_max[ch], (
            'frames_per_trigger %i out of range'%frames_per_trigger)
        dll.set_frames_per_trigger(self.handles[ch], frames_per_trigger)
        assert self._get_frames_per_trigger(ch) == frames_per_trigger
        if self.very_verbose:
            print("%s(%i): -> done setting frames per trigger."%(self.name, ch))
        return None

    def _get_trigger_mode(self, ch):
        if self.very_verbose:
            print("%s(%i): getting trigger mode"%(self.name, ch))
        number_to_mode = {0: "software", 1:"external", 2:"external_exposure"}
        trigger_mode = C.c_int(777)  # 777 not valid -> should change
        dll.get_trigger_mode(self.handles[ch], trigger_mode)
        self.trigger_mode[ch] = number_to_mode[trigger_mode.value]
        if self.very_verbose:
            print("%s(%i): -> trigger_mode = %s"%(
                        self.name, ch, self.trigger_mode[ch]))
        return self.trigger_mode[ch]

    def _set_trigger_mode(self, ch, trigger_mode):
        """
        Modes:
        - 'software': camera starts exposing after recieving a software
        trigger from self._send_software_trigger(ch). It will then either run
        continuously (frames_per_trigger=0) or for 'n' frames
        (frames_per_trigger=n) at the max frame rate based on the exposure, roi
        and binning.
        - 'external': an exposure is started ~12-15.5us after an external
        trigger is recieved at 'TRIGGER IN'. This is a 3.3V TTL signal with a
        min duration of 100us which can be configured for the RISING or
        FALLING edge. ***DO NOT EXCEED THE -0.7V to +5.0V RANGE!***
        - 'external_exposure': same as 'external' but here the exposure time is
        governed by the length of the TLL HIGH (or LOW) signal at 'TRIGGER IN'.
        There is also some delay AFTER the FALLING (or RISING) edge at the end
        of the exposure which is inherent to the camera. Thorlabs refers to
        this mode as 'BULB' or 'PDX' in the manual.
        NOTE: for all modes Thorlabs suggests inspecting the 'STROBE OUT' signal
        directly (e.g. using an oscilloscope) to determine exact timing of
        the exposure and readout etc (see manual).
        """
        if self.very_verbose:
            print("%s(%i): setting trigger mode = %s"%(
                self.name, ch, trigger_mode))
        mode_to_number = {"software":0, "external":1, "external_exposure":2}
        assert trigger_mode in mode_to_number, "mode '%s' not allowed"%mode
        dll.set_trigger_mode(self.handles[ch], mode_to_number[trigger_mode])
        assert self._get_trigger_mode(ch) == trigger_mode
        if self.very_verbose:
            print("%s(%i): -> done setting trigger mode."%(self.name, ch))
        return None

    def _send_software_trigger(self, ch):
        assert self.trigger_mode[ch] == 'software'
        dll.send_software_trigger(self.handles[ch])
        return None

    def _arm(self, ch, num_buffers):
        if self.very_verbose:
            print("%s(%i): arming..."%(self.name, ch), end='')
        assert 2 <= num_buffers <= 686, (
            'num_buffers %i out of range'%num_buffers)
        dll.arm_camera(self.handles[ch], num_buffers)
        self._num_buffers[ch] = num_buffers
        self._armed[ch] = True
        self._frame_count[ch] = None
        if self.very_verbose:
            print(" done.")
        return None

    def _disarm(self, ch):
        if self.very_verbose:
            print("%s(%i): disarming..."%(self.name, ch), end='')
        dll.disarm_camera(self.handles[ch])
        self._armed[ch] = False
        if self.very_verbose:
            print(" done.")
        return None

    def apply_settings(
        self,
        ch,                         # which camera? (=0 if only 1 connected)
        num_images=None,            # number of images to record, type(int)
        exposure_us=None,           # 40 <= type(int) <= 26,843,432
        height_px=None,             # 'min', 'max' or type(int) in legal range
        width_px=None,              # 'min', 'max' or type(int) in legal range
        trigger=None,               # "software"/"external"/"external_exposure"
        num_buffers=None,           # 2 <= type(int) <= 686
        ):
        if self.verbose:
            print("%s(%i): applying settings..."%(self.name, ch))
        if self._armed[ch]: self._disarm(ch)
        if num_images is not None:
            assert type(num_images) is int
            self.num_images[ch] = num_images
            if self.trigger_mode[ch] == 'software':
                self._set_frames_per_trigger(ch, num_images)
        if exposure_us is not None:
            self._set_exposure_time_us(ch, exposure_us)
        if height_px is not None or width_px is not None:
            if height_px is None: height_px = self.height_px[ch]
            if width_px  is None: width_px  = self.width_px[ch]
            roi_px = legalize_image_size(
                height_px, width_px, name=self.name, verbose=self.verbose)[2]
            self._set_roi(ch, roi_px)
        if trigger is not None:
            self._set_trigger_mode(ch, trigger)
            if trigger == 'software':
                if self.frames_per_trigger[ch] != self.num_images[ch]:
                    self._set_frames_per_trigger(ch, self.num_images[ch])
            else:
                if self.frames_per_trigger[ch] != 1:
                    self._set_frames_per_trigger(ch, 1) # 1 per external trigger
        if num_buffers is not None:
            self._num_buffers[ch] = num_buffers
        self._arm(ch, self._num_buffers[ch])
        if self.verbose:
            print("%s(%i): -> done applying settings."%(self.name, ch))
        return None

    def record_to_memory(
        self,
        ch,                     # which camera? (=0 if only 1 connected)
        allocated_memory=None,  # optionally pass numpy array for images
        software_trigger=True,  # False -> external trigger needed
        ):
        if self.verbose:
            print("%s(%i): recording to memory..."%(self.name, ch))
        assert self._armed[ch], 'camera not armed -> call .apply_settings()'
        h_px, w_px = self.height_px[ch], self.width_px[ch]
        if allocated_memory is None: # make numpy array if none given
            allocated_memory = np.zeros(
                (self.num_images[ch], h_px, w_px), 'uint16')
            output = allocated_memory # no memory provided so return some images
        else: # images placed in provided array
            assert isinstance(allocated_memory, np.ndarray)
            assert allocated_memory.dtype == np.uint16
            assert allocated_memory.shape == (self.num_images[ch], h_px, w_px)
            output = None # avoid returning potentially large array
        image_buffer = C.POINTER(C.c_ushort)()
        frame_count = C.c_int()
        metadata_pointer = C.POINTER(C.c_char)()
        metadata_size_in_bytes = C.c_int()
        timeout_us = self.frame_time_us[ch] + self.latency_us
        if software_trigger:
            self._send_software_trigger(ch) # mean time ~1281us (1e3 calls)
        for i in range(self.num_images[ch]):
            image_count = i + 1
            t0 = time.perf_counter()
            while True: # poll camera for next frame, not the most efficient...
                t_us = 1e6 * (time.perf_counter() - t0)
                dll.get_frame(self.handles[ch], # mean time ~1281us (1e3 calls)
                              image_buffer,
                              frame_count,
                              metadata_pointer,
                              metadata_size_in_bytes)
                current_frame = frame_count.value
                if self._frame_count[ch] is not None: # re-zero frame_count
                    current_frame -= self._frame_count[ch]
                if current_frame == image_count:# correct image
                    break
                if current_frame > image_count: # we lost a frame
                    print("%s(%i): ***LOST FRAME!***"%(self.name, ch))
                    print("%s(%i): current_frame (%i) > image_count (%i)"%(
                        self.name, ch, current_frame, image_count))
                    raise
                if t_us >  timeout_us:          # timeout
                    print("%s(%i): ***LOST FRAME TIMEOUT!***"%(self.name, ch))
                    print("%s(%i): t_us (%i) > timeout_us (%i)"%(
                        self.name, ch, t_us, timeout_us))
                    print("%s(%i): (current_frame = %i, image_count = %i)"%(
                        self.name, ch, current_frame, image_count))
                    raise
            frame = np.ctypeslib.as_array(image_buffer, shape=(h_px, w_px))
            allocated_memory[i, :, :] = frame   # get image
        self._frame_count[ch] = frame_count.value
        if self.verbose:
            print("%s(%i): -> done recording to memory."%(self.name, ch))
        return output

    def close(self):
        if self.verbose: print("%s: closing..."%self.name, end='')
        for handle in self.handles:
            dll.close_camera(handle)
        dll.close_sdk()
        if self.verbose: print(" done.")
        return None

def legalize_image_size(
    height_px='max', width_px='max', name='CS165MU1', verbose=True):
    """returns a nearby legal image size at the *center* of the camera chip"""
    height_step, width_step = 4, 16
    min_height, min_width, max_height, max_width = 16, 1200, 1080, 1440
    # Actually min_height=4, min_width=80. However, with the vertical cropping
    # too low the PC/adaptor may struggle to keep up and frames can be dropped
    # (since vertical cropping increases FPS and reduces read time).
    # For some unknown reason, if the width_px < 1200 the camera also drops
    # frames for ROI with more than ~2^14 = 16384 pixels. It took quite a while
    # to figure this out, and so instead of encoding it in the legalizer it
    # seems preferable to simply not allow narrow images. There has no obviuos
    # performance penalty and the 'extra' pixels can be ignored.
    ud_center = (max_height / 2)
    lr_center = (max_width  / 2)
    if verbose:
        print("%s: requested image size (pixels)"%name)
        print("%s:  = %s x %s (height x width)"%(name, height_px, width_px))
    if height_px == 'min': height_px = min_height
    if height_px == 'max': height_px = max_height        
    if width_px  == 'min': width_px  = min_width
    if width_px  == 'max': width_px  = max_width
    assert type(height_px) is int and type(width_px) is int
    assert min_height <= height_px <= max_height    
    assert min_width  <= width_px  <= max_width
    num_height_steps = (height_px // height_step)
    num_width_steps  = (width_px  // width_step)
    if num_height_steps % 2 != 0: num_height_steps += 1 # must be even for chip 
    height_px = height_step * num_height_steps # now legalized
    width_px  = width_step  * num_width_steps  # now legalized
    l_px = int(lr_center - (width_px  / 2))
    r_px = int(lr_center + (width_px  / 2)) - 1
    u_px = int(ud_center - (height_px / 2))
    d_px = int(ud_center + (height_px / 2)) - 1 
    roi_px = {'left':l_px, 'right':r_px, 'up':u_px, 'down':d_px}
    if verbose:
        print("%s: legal image size (pixels)"%name)
        print("%s:  = %i x %i (height x width)"%(name, height_px, width_px))
        print("%s: roi = %s"%(name, roi_px))
    return height_px, width_px, roi_px

### Tidy and store DLL calls away from main program:

os.add_dll_directory(os.getcwd())
# needs "thorlabs_tsi_camera_sdk.dll" in directory
# must equate to 'tl_camera_sdk_dll_initialize'
dll = C.cdll.LoadLibrary("thorlabs_tsi_camera_sdk")

dll.get_error_message = dll.tl_camera_get_last_error
dll.get_error_message.argtypes = None
dll.get_error_message.restype = C.c_char_p

def check_error(error_code):
    if error_code != 0:
        print("Error message from Thorlabs CS165MU1: ", end='')
        error_message = dll.get_error_message()
        print(error_message.decode('ascii'))
        raise UserWarning(
            "Thorlabs CS165MU1: %i; see above for details."%(error_code))
    return error_code

dll.open_sdk = dll.tl_camera_open_sdk
dll.open_sdk.argtypes = None
dll.open_sdk.restype = check_error

dll.find_cameras = dll.tl_camera_discover_available_cameras
dll.find_cameras.argtypes = [
    C.c_char_p,                         # serial_numbers
    C.c_int]                            # str_length
dll.find_cameras.restype = check_error

dll.open_camera = dll.tl_camera_open_camera
dll.open_camera.argtypes = [
    C.c_char_p,                         # camera_serial_number
    C.POINTER(C.c_void_p)]              # tl_camera_handle
dll.open_camera.restype = check_error

dll.get_exposure_time_range_us = dll.tl_camera_get_exposure_time_range
dll.get_exposure_time_range_us.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_longlong),            # exposure_time_us_min
    C.POINTER(C.c_longlong)]            # exposure_time_us_max
dll.get_exposure_time_range_us.restype = check_error

dll.get_exposure_time_us = dll.tl_camera_get_exposure_time
dll.get_exposure_time_us.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_longlong)]            # exposure_time_us
dll.get_exposure_time_us.restype = check_error

dll.set_exposure_time_us = dll.tl_camera_set_exposure_time
dll.set_exposure_time_us.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.c_longlong]                       # exposure_time_us
dll.set_exposure_time_us.restype = check_error

dll.get_read_time_ns = dll.tl_camera_get_sensor_readout_time
dll.get_read_time_ns.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_int)]                 # sensor_readout_time_ns
dll.get_read_time_ns.restype = check_error

dll.get_frame_time_us = dll.tl_camera_get_frame_time
dll.get_frame_time_us.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_int)]                 # frame_time_us
dll.get_frame_time_us.restype = check_error

dll.get_roi_range = dll.tl_camera_get_roi_range
dll.get_roi_range.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_int),                 # upper_left_x_pixels_min
    C.POINTER(C.c_int),                 # upper_left_y_pixels_min
    C.POINTER(C.c_int),                 # lower_right_x_pixels_min
    C.POINTER(C.c_int),                 # lower_right_y_pixels_min
    C.POINTER(C.c_int),                 # upper_left_x_pixels_max
    C.POINTER(C.c_int),                 # upper_left_y_pixels_max
    C.POINTER(C.c_int),                 # lower_right_x_pixels_max
    C.POINTER(C.c_int)]                 # lower_right_y_pixels_max
dll.get_roi_range.restype = check_error

dll.get_roi = dll.tl_camera_get_roi
dll.get_roi.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_int),                 # upper_left_x_pixels
    C.POINTER(C.c_int),                 # upper_left_y_pixels
    C.POINTER(C.c_int),                 # lower_right_x_pixels
    C.POINTER(C.c_int)]                 # lower_right_y_pixels
dll.get_roi.restype = check_error

dll.set_roi = dll.tl_camera_set_roi
dll.set_roi.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.c_int,                            # upper_left_x_pixels
    C.c_int,                            # upper_left_y_pixels
    C.c_int,                            # lower_right_x_pixels
    C.c_int]                            # lower_right_y_pixels
dll.set_roi.restype = check_error

dll.get_poll_timeout_ms = dll.tl_camera_get_image_poll_timeout
dll.get_poll_timeout_ms.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_int)]                 # timeout_ms
dll.get_poll_timeout_ms.restype = check_error

dll.set_poll_timeout_ms = dll.tl_camera_set_image_poll_timeout
dll.set_poll_timeout_ms.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.c_int]                            # timeout_ms
dll.set_poll_timeout_ms.restype = check_error

dll.get_frames_per_trigger_range = dll.tl_camera_get_frames_per_trigger_range
dll.get_frames_per_trigger_range.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_uint),                # number_of_frames_per_trigger_min
    C.POINTER(C.c_uint)]                # number_of_frames_per_trigger_max
dll.get_frames_per_trigger_range.restype = check_error

dll.get_frames_per_trigger = (
    dll.tl_camera_get_frames_per_trigger_zero_for_unlimited)
dll.get_frames_per_trigger.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_uint)]                # number_of_fpt_or_zero_for_unlimited
dll.get_frames_per_trigger.restype = check_error

dll.set_frames_per_trigger = (
    dll.tl_camera_set_frames_per_trigger_zero_for_unlimited)
dll.set_frames_per_trigger.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.c_uint]                           # number_of_fpt_or_zero_for_unlimited
dll.set_frames_per_trigger.restype = check_error

dll.get_trigger_mode = dll.tl_camera_get_operation_mode
dll.get_trigger_mode.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.c_int)]                 # operation_mode
dll.get_trigger_mode.restype = check_error

dll.set_trigger_mode = dll.tl_camera_set_operation_mode
dll.set_trigger_mode.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.c_int]                            # operation_mode
dll.set_trigger_mode.restype = check_error

dll.arm_camera = dll.tl_camera_arm
dll.arm_camera.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.c_int]                            # number_of_frames_to_buffer
dll.arm_camera.restype = check_error

dll.send_software_trigger = dll.tl_camera_issue_software_trigger
dll.send_software_trigger.argtypes = [
    C.c_void_p]                         # tl_camera_handle
dll.send_software_trigger.restype = check_error

dll.get_frame = dll.tl_camera_get_pending_frame_or_null
dll.get_frame.argtypes = [
    C.c_void_p,                         # tl_camera_handle
    C.POINTER(C.POINTER(C.c_ushort)),   # image_buffer
    C.POINTER(C.c_int),                 # frame_count
    C.POINTER(C.POINTER(C.c_char)),     # metadata
    C.POINTER(C.c_int)]                 # metadata_size_in_bytes
dll.get_frame.restype = check_error

dll.disarm_camera = dll.tl_camera_disarm
dll.disarm_camera.argtypes = [
    C.c_void_p]                         # tl_camera_handle
dll.disarm_camera.restype = check_error

dll.close_camera = dll.tl_camera_close_camera
dll.close_camera.argtypes = [
    C.c_void_p]                         # tl_camera_handle
dll.close_camera.restype = check_error

dll.close_sdk = dll.tl_camera_close_sdk
dll.close_sdk.argtypes = None
dll.close_sdk.restype = check_error

if __name__ == '__main__':
    from tifffile import imread, imwrite
    camera = Camera(verbose=True, very_verbose=True)
    ch = 0 # change to test different cameras

    # take some pictures:
    camera.apply_settings(ch,
                          num_images=10,
                          exposure_us=1000,
                          height_px='min',
                          width_px=1200)
    images = camera.record_to_memory(ch)
    imwrite('test0.tif', images, imagej=True)

    # max fps test:
    frames = 1000
    camera.apply_settings(ch,
                          num_images=frames,
                          exposure_us=camera.exposure_min_us[ch],
                          height_px='min',
                          width_px='max',
                          trigger='software')
    images = np.zeros((camera.num_images[ch],
                       camera.height_px[ch],
                       camera.width_px[ch]),
                      'uint16')
    t0 = time.perf_counter()
    camera.record_to_memory(ch, allocated_memory=images)
    time_s = time.perf_counter() - t0
    print("\nMax fps = %0.2f\n"%(frames/time_s)) # ~ 35 -> 650 typical
    imwrite('test1.tif', images, imagej=True)

    # max fps test -> multiple recordings:
    iterations = 10
    frames = 100
    camera.apply_settings(ch,
                          num_images=frames,
                          exposure_us=camera.exposure_min_us[ch],
                          height_px='min',
                          width_px='max',
                          trigger='software')
    images = np.zeros((camera.num_images[ch],
                       camera.height_px[ch],
                       camera.width_px[ch]),
                      'uint16')
    t0 = time.perf_counter()
    for i in range(iterations):
        camera.record_to_memory(ch, allocated_memory=images)
    time_s = time.perf_counter() - t0
    total_frames = iterations * frames
    print("\nMax fps = %0.2f\n"%(total_frames/time_s)) # ~ 35 -> 650 typical
    imwrite('test2.tif', images, imagej=True)

    # random input testing: successful with 1000 iterations and 2 cameras
    iterations = 10
    min_h_px, min_w_px = legalize_image_size('min','min')[:2]
    max_h_px, max_w_px = legalize_image_size('max','max')[:2]
    camera.verbose, camera.very_verbose = False, False
    blank_frames, total_latency_ms = 0, 0
    for i in range(iterations):
        print('\nRandom input test: %06i (ch%i)'%(i, ch))
        ch      = np.random.randint(0, len(camera.channels))
        frames  = np.random.randint(1, 10)
        exp_us  = np.random.randint(camera.exposure_min_us[ch], 100000)
        h_px    = np.random.randint(min_h_px, max_h_px)
        w_px    = np.random.randint(min_w_px, max_w_px)
        buffers = np.random.randint(2, 686)
        camera.apply_settings(ch=ch,
                              num_images=frames,
                              exposure_us=exp_us,
                              height_px=h_px,
                              width_px=w_px,
                              trigger='software',
                              num_buffers=buffers)
        images = np.zeros((camera.num_images[ch],
                           camera.height_px[ch],
                           camera.width_px[ch]),
                          'uint16')
        t0 = time.perf_counter()
        camera.record_to_memory(ch, allocated_memory=images)
        t1 = time.perf_counter()
        time_per_image_ms = 1e3 * (t1 - t0) / frames
        latency_ms = time_per_image_ms - 1e-3 * camera.exposure_us[ch]
        total_latency_ms += latency_ms
        print("latency (ms) = %0.6f"%latency_ms)
        print("shape of images:", images.shape)
        if i == 0: imwrite('test3.tif', images, imagej=True)
    average_latency_ms = total_latency_ms / iterations
    print(" -> average latency (ms) = %0.6f"%average_latency_ms) # ~3ms typical

    camera.close()
