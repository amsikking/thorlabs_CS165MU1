import numpy as np
from tifffile import imread, imwrite
import thorlabs_CS165MU1
import ni_PCIe_6738

'''Test the camera's ability to follow external triggers'''
ao = ni_PCIe_6738.DAQ(num_channels=1, rate=1e6, verbose=False)

camera = thorlabs_CS165MU1.Camera(verbose=True)
ch = 0 # change to test different cameras

frames = 10                                 # ~35 -> 650 fps achieved
camera.apply_settings(ch=ch,
                      num_images=frames,
                      exposure_us=40,       # 'min' -> 'max' seems stable
                      height_px='min',      # 'min' -> 'max' seems stable
                      width_px='max',
                      trigger='external')

jitter_time_us = 16 # how much slop is needed between triggers? 16us?
jitter_px = max(ao.s2p(1e-6 * jitter_time_us), 1)
exposure_px = ao.s2p(1e-6 * camera.exposure_us[ch])
read_px = ao.s2p(1e-6 * camera.read_time_us[ch])
period_px = jitter_px + exposure_px + read_px

voltage_series = []
for i in range(frames):
    volt_period = np.zeros((period_px, ao.num_channels), 'float64')
    volt_period[ # rising edge light on, falling edge light off
        jitter_px:jitter_px + exposure_px, 0] = 3.3 
    voltage_series.append(volt_period)
voltages = np.concatenate(voltage_series, axis=0)

# can the camera keep up?
ao._write_voltages(voltages)                # write volts first to avoid delay
images = np.zeros((camera.num_images[ch],   # allocate memory to pass in
                   camera.height_px[ch],
                   camera.width_px[ch]),'uint16')

iteration = 10                      # stable to 10000 iterations
camera.verbose = False              # avoid printing to help with race!
for i in range(iteration):
    print('\nIteration %i'%i)
    ao.play_voltages(block=False)   # race condition!
    camera.record_to_memory(        # -> waits for trigger
        ch, allocated_memory=images, software_trigger=False)
imwrite('test_external_trigger.tif', images, imagej=True)
camera.verbose = True

time_s = ao.p2s(voltages.shape[0])
fps = frames /  time_s
print('fps = %02f'%fps)             # (forced by ao play)

camera.close()
ao.close()
