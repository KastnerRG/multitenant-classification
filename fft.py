import os
import glob
import numpy as np
import time
from tuned_trace import CombinedTrace
from scipy import signal
import multiprocessing
import random
from scipy.fft import fft

def multi_proc_fft(app, seg_len, ffts_per_trace, detrend, half_trace, data_dir_train, data_dir_test, data_dir, transition):
  print(app)
  count = 0 
  num_traces = 100
  num_train = int(num_traces*ffts_per_trace*4/5) # Reserve 20% for testing
  data_dir = glob.glob(data_dir)[0] + '/'
  for np_file in os.listdir(data_dir):
    filename = data_dir + np_file
    print(filename)
    data = np.load(filename)
    if half_trace:
      data = data[:len(data) // 2]
      
    trace = CombinedTrace(data)
    neg_trace = trace.neg.pop
    pos_trace = trace.pos.pop

    # Detrend
    if detrend: 
      neg_trace = signal.detrend(neg_trace)
      pos_trace = signal.detrend(pos_trace)
       
    for j in range(ffts_per_trace):
      # TODO: Make sure this isn't a segment we've seen already?
      # Splice a random subtrace of length seg_len
      rand_seg_start = random.randint(0, len(pos_trace)-seg_len)

      rand_neg_seg = neg_trace[rand_seg_start:rand_seg_start+seg_len]
      rand_pos_seg = pos_trace[rand_seg_start:rand_seg_start+seg_len]

      # FFT
      # 4 channels: each real and complex freq has positive and negative
      neg_freq = np.abs(fft(rand_neg_seg))
      pos_freq = np.abs(fft(rand_pos_seg))

      # Normalize fft across all 4 channels?

      # Splice first half
      neg_freq = neg_freq[:len(neg_freq)//2]
      pos_freq = pos_freq[:len(pos_freq)//2]

      # Encode pos and neg edges in two dimensions
      # fft_res = np.stack((neg_freq, pos_freq), axis=-1) 
      # fft_res = neg_freq
      if transition == 'pos':
          fft_res = pos_freq
      elif transition == 'neg':
          fft_res = neg_freq
      # fft_res = np.concatenate((neg_freq, pos_freq), axis=None) # For half trace

      if count < num_train:
        np.save(data_dir_train + app + "." + str(count), fft_res)
      else:
        np.save(data_dir_test + app + "." + str(count), fft_res)
      count+=1


def generate_FFT(data_sets, seg_len, ffts_per_trace, detrend, half_trace, train_dir, test_dir, transition):
    starttime = time.time()
    processes = []
    for run in data_sets: 
      p = multiprocessing.Process(target=multi_proc_fft, args=(run, seg_len, ffts_per_trace, detrend, half_trace, train_dir, test_dir, data_sets[run], transition))
      processes.append(p)
      p.start()

    for process in processes:
        process.join()

    print('That took {} seconds.'.format(time.time() - starttime))


if __name__ == "__main__":

  # Neg edge MAX phi and MAX theta (no bg subtraction)
  # boards = [
  #   '60578572-0904-11ec-b64d-00056b00b6c5',
  #   '90e5659c-0904-11ec-a4ba-00183e0248fe',
  #   '6f1e8076-094d-11ec-a2d2-00056b00f8ca',
  #   '74ef25f4-094e-11ec-bcb9-00056b0100fd',
  #   'ad63c212-0973-11ec-ab31-00183e02912e'
  # ]

  # Neg edge MAX phi and MAX theta with background subtraction
  # boards = [
  #   '03bcc8ae-0ae9-11ec-a87e-00183e0248fe',
  #   '269427fa-0ae9-11ec-8607-00183e02912e',
  #   '15184b50-0ae9-11ec-896c-00056b00f8ca',
  #   '103e5f2a-0ae9-11ec-ae28-00056b00b6c5',
  #   '1912b3e4-0ae9-11ec-b551-00056b0100fd'
  # ]

  # Neg edge MIN phi and MAX theta with background subtraction
  # boards = [
  #   'b3ed5b48-0d4c-11ec-b4de-00183e0248fe',
  #   '8e84acd0-0d4c-11ec-b88d-00183e02912e',
  #   '7d491686-0d4c-11ec-b799-00056b00b6c5',
  #   '791be624-0d4c-11ec-ad11-00056b00f8ca',
  #   '6e408fca-0d4c-11ec-843c-00056b0100fd'
  # ]

  # Pos edge MIN phi and MIN theta (no bg subtraction)
  # boards = [
  #   '218e2bce-0f95-11ec-9973-00056b00f8ca',
  #   '39aa0ba6-0f95-11ec-99b8-00056b00b6c5',
  #   '43cb4000-0f95-11ec-bc1d-00183e02912e',
  #   '4c1b5ac4-0f95-11ec-987a-00183e0248fe',
  #   '8c43f620-0f94-11ec-ab10-00056b0100fd'
  # ]

  # Neg edge MIN phi and MAX theta (no bg sub)
  # boards = [
  #   '1ab5997c-10ee-11ec-a440-00183e0248fe',
  #   '6037d892-10ef-11ec-b6bc-00183e02912e',
  #   '6395d75a-10ef-11ec-a0ef-00056b00b6c5',
  #   '81c0fa5c-10ef-11ec-ae62-00056b00f8ca',
  #   '91ae1314-10ef-11ec-8ad0-00056b0100fd'
  # ]

  # Pos edge MAX phi and MAX theta with bg sub
  #boards = [
  #  'd1d73bec-126f-11ec-bffa-00183e02912e',
  #  'e5fb3dda-126f-11ec-af7d-00056b00f8ca',
  #  'f0d751da-126f-11ec-bb74-00056b00b6c5',
  #  'fa510490-126f-11ec-ad4b-00056b0100fd',
  #  'df9b8668-153d-11ec-bb14-00183e0248fe'
  #]

  # ALL
  boards = [
          'neg_edge_max_phi_max_theta/60578572-0904-11ec-b64d-00056b00b6c5',
          'neg_edge_max_phi_max_theta/90e5659c-0904-11ec-a4ba-00183e0248fe',
          'neg_edge_max_phi_max_theta/6f1e8076-094d-11ec-a2d2-00056b00f8ca',
          'neg_edge_max_phi_max_theta/74ef25f4-094e-11ec-bcb9-00056b0100fd',
          'neg_edge_max_phi_max_theta/ad63c212-0973-11ec-ab31-00183e02912e',
          'neg_edge_max_phi_max_theta_bg_sub/03bcc8ae-0ae9-11ec-a87e-00183e0248fe',
          'neg_edge_max_phi_max_theta_bg_sub/269427fa-0ae9-11ec-8607-00183e02912e',
          'neg_edge_max_phi_max_theta_bg_sub/15184b50-0ae9-11ec-896c-00056b00f8ca',
          'neg_edge_max_phi_max_theta_bg_sub/103e5f2a-0ae9-11ec-ae28-00056b00b6c5',
          'neg_edge_max_phi_max_theta_bg_sub/1912b3e4-0ae9-11ec-b551-00056b0100fd',
          'neg_edge_min_phi_max_theta_bg_sub/b3ed5b48-0d4c-11ec-b4de-00183e0248fe',
          'neg_edge_min_phi_max_theta_bg_sub/8e84acd0-0d4c-11ec-b88d-00183e02912e',
          'neg_edge_min_phi_max_theta_bg_sub/7d491686-0d4c-11ec-b799-00056b00b6c5',
          'neg_edge_min_phi_max_theta_bg_sub/791be624-0d4c-11ec-ad11-00056b00f8ca',
          'neg_edge_min_phi_max_theta_bg_sub/6e408fca-0d4c-11ec-843c-00056b0100fd',
          'pos_edge_min_phi_min_theta/218e2bce-0f95-11ec-9973-00056b00f8ca',
          'pos_edge_min_phi_min_theta/39aa0ba6-0f95-11ec-99b8-00056b00b6c5',
          'pos_edge_min_phi_min_theta/43cb4000-0f95-11ec-bc1d-00183e02912e',
          'pos_edge_min_phi_min_theta/4c1b5ac4-0f95-11ec-987a-00183e0248fe',
          'pos_edge_min_phi_min_theta/8c43f620-0f94-11ec-ab10-00056b0100fd',
          'neg_edge_min_phi_max_theta/1ab5997c-10ee-11ec-a440-00183e0248fe',
          'neg_edge_min_phi_max_theta/6037d892-10ef-11ec-b6bc-00183e02912e',
          'neg_edge_min_phi_max_theta/6395d75a-10ef-11ec-a0ef-00056b00b6c5',
          'neg_edge_min_phi_max_theta/81c0fa5c-10ef-11ec-ae62-00056b00f8ca',
          'neg_edge_min_phi_max_theta/91ae1314-10ef-11ec-8ad0-00056b0100fd',
          'pos_edge_max_phi_max_theta_bg_sub/d1d73bec-126f-11ec-bffa-00183e02912e',
          'pos_edge_max_phi_max_theta_bg_sub/e5fb3dda-126f-11ec-af7d-00056b00f8ca',
          'pos_edge_max_phi_max_theta_bg_sub/f0d751da-126f-11ec-bb74-00056b00b6c5',
          'pos_edge_max_phi_max_theta_bg_sub/fa510490-126f-11ec-ad4b-00056b0100fd',
          'pos_edge_max_phi_max_theta_bg_sub/df9b8668-153d-11ec-bb14-00183e0248fe'
  ]

  for board in boards:

      data_set = {
        'base'          : f'./power_traces/{board}/base',
        'aes'           : f'./power_traces/{board}/aes',
        'ro'            : f'./power_traces/{board}/ro',
        'orca-aes'      : f'./power_traces/{board}/orca_aes',
        'orca-present'  : f'./power_traces/{board}/orca_present',
        'mb-aes'        : f'./power_traces/{board}/microblaze_aes',
        'mb-present'    : f'./power_traces/{board}/microblaze_present',
        'pico-aes'      : f'./power_traces/{board}/pico_aes',
        'pico-present'  : f'./power_traces/{board}/pico_present',
        'present-hls'   : f'./power_traces/{board}/present',
        'dsp'           : f'./power_traces/{board}/dsp',
        'cortex-aes'    : f'./power_traces/{board}/cortex_aes',
        'cortex-present': f'./power_traces/{board}/cortex_present'
      }

      data_dir_train = f"./segments/fft/{board}/Train_Segments_Detrend_8K/"
      data_dir_test = f"./segments/fft/{board}/Test_Segments_Detrend_8K/"

      # Create directories if they don't already exist
      if not os.path.exists(data_dir_train):
        os.makedirs(data_dir_train)
      if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)
      
      half_trace = False
      seg_len = 8192
      ffts_per_trace = 10
      detrend = True

      transition = board.split('_')[0]
      print(transition)

      generate_FFT(data_set, seg_len, ffts_per_trace, detrend, half_trace, data_dir_train, data_dir_test, transition)
