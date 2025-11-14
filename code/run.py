import csv
import datetime
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import numpy.ma as ma
import pandas as pd
import torch

from model import DecBlock, Decoder, EncBlock, Encoder
from torch.autograd import Variable

def EnKF_para(swh_f_ens, swh_f_mean, swh_o, H, n, m, N, sigma_o, obs_dist, r_local):
    swh_o_member = np.zeros(m)
    swh_f_member = np.zeros(n)
    swh_a_member = np.zeros(n)
    swh_a_ens    = np.zeros((n, N))
    K          = np.zeros((n, m))
    
    # Foracast covariance
    P_f      = np.cov(swh_f_ens)

    # Covariance localization
    for i in range(n):
        for j in range(n):
            cor = localize(i, j, n, r_local)
            P_f[i, j] = cor*P_f[i, j]

    # Kalman gain
    K = P_f@(H.T)@np.linalg.inv((sigma_o**2)*np.eye(m) + H@P_f@(H.T))

    # Analysis ensemble
    for k in range(N):
        swh_o_member[0:m] = swh_o[0:m] + sigma_o*np.random.randn(m)
        swh_f_member[0:n] = swh_f_ens[0:n, k]
        swh_a_member = swh_f_member + K@(swh_o_member - H@swh_f_member)
        swh_a_ens[0:n, k] = swh_a_member[0:n]
        
    return swh_a_ens

def localize(i, j, n, r_local):
    if r_local < 0:        
        cor = 1.0
    else:
        n_local = 2*r_local
        n_dist  = i - j

        if (i > n_local and i < n - n_local) or (j > n_local and j < n - n_local):
            r = abs(n_dist)
        else:
            r = n
            for k in range(-1, 2):
                r = min(abs(n_dist + k*n), r)
                
        if r_local > 0:
            x = float(r)/float(r_local)
        else:
            x = 0.0

        if   r <= r_local:
            cor = -1.0/4.0*x**5 + 1.0/2.0*x**4 + 5.0/8.0*x**3 - 5.0/3.0*x**2 + 1.0
        elif r <= n_local:
            cor = 1.0/12.0*x**5 - 1.0/2.0*x**4 + 5.0/8.0*x**3 + 5.0/3.0*x**2 - 5.0*x + 4.0 - 2.0/3.0/x
        else:
            cor = 0.0

    return cor

def inflation_rev(swh_f_mean, P_f, swh_o, H, sigma_o, n, m, rho_min, rho_max, kappa, rho, ratio):
    d    = swh_o - H @ swh_f_mean
    work = (np.dot(d, d) - float(m) * sigma_o**2) / np.trace(H @ P_f @ (H.T))

    if work < rho_min:
        work = rho_min
    elif work > rho_max:
        work = rho_max
        
    rho   = (rho + ratio*work)/(1.0 + ratio)
    ratio = kappa*ratio/(1.0 + ratio)
    
    return rho, ratio

def Compute_RMSE(x, y):
    result = []
    
    for t in range(len(x)):
        a = ma.masked_invalid(x[t])
        b = ma.masked_invalid(y[t])
        mask = (~a.mask & ~b.mask)
        
        result.append(np.sqrt(((a[mask] - b[mask]) ** 2).mean(axis=0)))

    return np.array(result)

path = [
    '../../data/insitu/DH.csv',
    '../../data/insitu/WJ.csv',
    '../../data/insitu/SC.csv',
    '../../data/insitu/PH.csv',
    '../../data/insitu/UL.csv',
    '../../data/insitu/ULNE.csv',
    '../../data/insitu/ULNW.csv',
]

if __name__ == '__main__' :
    date_split = ['2015010100', '2016123123']

    insitu = []

    for h in range(len(path)):
        data = pd.read_csv(path[h], encoding='cp949')
        data.index = data['date']
        
        x_data = data.loc[pd.IndexSlice[date_split[0]:date_split[1]], :]
        x_data = np.stack([x_data.loc[:, 'date'], x_data.loc[:, 'swh'], x_data.loc[:, 'pp1d']], axis=1)
        insitu.append(np.array(x_data)[:, 1:])
        
        # x_insitu[h] = (x_insitu[h] - insitu_mean[h]) / insitu_std[h]
        # y_insitu[h] = (y_insitu[h] - insitu_mean[h]) / insitu_std[h]
        
    insitu = np.array(insitu, dtype=np.float32)[:, :, 0]
    insitu = np.transpose(insitu, (1, 0))

    ############################################

    wave_mean = np.load('../../data/wave_mean.npy')
    wave_std  = np.load('../../data/wave_std.npy')

    swh_mean = np.reshape(wave_mean, (1, 2, 34, 30))[:, 0]
    swh_std  = np.reshape(wave_std, (1, 2, 34, 30))[:, 0]
    mwp_mean = np.reshape(wave_mean, (1, 2, 34, 30))[:, 1]
    mwp_std  = np.reshape(wave_std, (1, 2, 34, 30))[:, 1]
    
    wind_mean = np.load('../../data/wind_mean.npy')
    wind_std  = np.load('../../data/wind_std.npy')

    u_wind_mean = np.reshape(wind_mean, (1, 2, 68, 60))[:, 0]
    u_wind_std  = np.reshape(wind_std, (1, 2, 68, 60))[:, 0]
    v_wind_mean = np.reshape(wind_mean, (1, 2, 68, 60))[:, 1]
    v_wind_std  = np.reshape(wind_std, (1, 2, 68, 60))[:, 1]

    x_wave = np.load('../../data/x_wave.npy')
    y_wave = np.load('../../data/y_wave.npy')

    x_wind  = np.load('../../data/x_wind.npy')
    y_wind  = np.load('../../data/y_wind.npy')

    x_wave = (x_wave - wave_mean) / wave_std
    y_wave = (y_wave - wave_mean) / wave_std
    insitu[0] = (insitu[0] - wave_mean[0, -10, 5]) / wave_std[0, -10, 5]
    insitu[1] = (insitu[1] - wave_mean[0,  -9, 4]) / wave_std[0,  -9, 4]
    insitu[2] = (insitu[2] - wave_mean[0, -10, 4]) / wave_std[0, -10, 4]
    insitu[3] = (insitu[3] - wave_mean[0,  -8, 5]) / wave_std[0,  -8, 5]
    insitu[4] = (insitu[4] - wave_mean[0, -10, 7]) / wave_std[0, -10, 7]
    insitu[5] = (insitu[5] - wave_mean[0, -11, 8]) / wave_std[0, -11, 8]
    insitu[6] = (insitu[6] - wave_mean[0, -10, 6]) / wave_std[0, -10, 6]

    x_wind = (x_wind - wind_mean) / wind_std
    y_wind = (y_wind - wind_mean) / wind_std

    # 이름      변수            기간                    모양(batch_size, channel, height, width)
    # x_wave    swh, mwp        2013.01.01 ~ 2014.12.31 (17520, 2, 34, 30)
    # y_wave    swh, mwp        2015.01.01 ~ 2016.12.31 (17544, 2, 34, 30)
    # x_wind    u-wind, v-wind  2013.01.01 ~ 2014.12.31 (17520, 2, 68, 60)
    # y_wind    u-wind, v-wind  2015.01.01 ~ 2016.12.31 (17544, 2, 68, 60)

    ###############################################

    time_step = [24, 48]
    dt_da     = 3
    time      = 17497

    # Para. definition ####################################################
    sigma_o = 0.2   # Standard deviation of observation error; #0.2 (m, Ho)
    sigma_f = 0.4   # Standard deviation of forecast error at initial time

    r_local = 4     # Covariance localization radius  # 1,000km
    rho_min = 0.9   # Lower bound for adaptive covariance inflation; # ?
    rho_max = 5.0   # Upper bound for adaptive covariance inflation; # ?

    obs_dist = 4    # 1,000km spatial correlation length
    r_extr   = 2    # input radius; # ? 2D? 1D? 

    # Parameters of adaptive covariance inflation at initial time
    kappa = 1.1
    ratio = 1.0
    rho   = 1.0

    mean  = 0
    sd2   = 1.0e-2       # added noise (variance)
    sd1   = np.sqrt(sd2) # added noise (standard deviation)

    m = 7
    N = 30
    n = dt_da*34*30
    ######################################################################

    H = np.zeros((m, 1, 34, 30), dtype='int')
    H[0, 0, -10, 5] = 1
    H[1, 0,  -9, 4] = 1
    H[2, 0, -10, 4] = 1
    H[3, 0,  -8, 5] = 1
    H[4, 0, -10, 7] = 1
    H[5, 0, -11, 8] = 1
    H[6, 0, -10, 6] = 1
    H = np.repeat(H, dt_da, axis=1)
    H = np.reshape(H, (m, -1))

    E = Encoder(EncBlock).cuda()
    D = Decoder(DecBlock).cuda()

    E.load_state_dict(torch.load('encoder_weights.pt'), strict=False)
    D.load_state_dict(torch.load('decoder_weights.pt'), strict=False)
    
    E.eval()
    D.eval()

    inferA = [[], [], [], []]
    inferB = [[], [], [], []]
    inferC = [[], [], [], []]

    mask = np.ones_like(y_wave[:, 0:1])[:1]
    mask[np.isnan(mask)] = 0
    mask = Variable(torch.from_numpy(mask)).type(torch.FloatTensor).cuda().unsqueeze(0)

    with torch.no_grad():
        wind = Variable(torch.from_numpy(np.array([x_wind[-time_step[0]:]]))).type(torch.FloatTensor).cuda()
        wave = Variable(torch.from_numpy(np.array([x_wave[-time_step[0]:]]))).type(torch.FloatTensor).cuda()

        swh = wave[:, :, 0:1]
        mwp = wave[:, :, 1:2]
        u_wind = wind[:, :, 0:1]
        v_wind = wind[:, :, 1:2]

        date = [datetime.datetime(int(date_split[0][:4]), int(date_split[0][4:6]), int(date_split[0][6:8]), int(date_split[0][8:10])), None, None]
        date[1] = date[0] + datetime.timedelta(hours=time_step[1] - 1)

        recent_index = 0
        recent_date  = date[0]
        label        = [[], [], [], []]
        RMSE         = [[], [], [], []]

        swh_a_ens = []

        for t in range(0, len(y_wave) - time_step[1] + 1, dt_da):
            if t >= time:
                break

            # cold start 마다 swh에 noise를 추가하여 N개의 emsemble (swh_a_ens) 생성
            if len(swh_a_ens) == 0:
                for k in range(N):
                    error_f = sigma_f * (torch.randn_like(swh[:, -dt_da:]) * sd1 + mean)
                    swh_a_ens.append(swh[:, -dt_da:] + error_f)

                swh_a_ens = torch.cat(swh_a_ens)
            else:
                swh_a_ens = torch.from_numpy(swh_a_ens).type(torch.FloatTensor).cuda()
                swh_a_ens = swh_a_ens.transpose(0, 1).view(N, dt_da, 1, 34, 30)

            _swh      = []
            _mwp      = []
            _u_wind   = []
            _v_wind   = []

            swh_f_ens = []

            # forecast ensemble
            for k in range(N):
                _s = []
                _m = []
                _u = []
                _v = []

                for i in range(time_step[1]):
                    lead_time = Variable(torch.from_numpy(np.array([np.eye(time_step[1])[i]]))).type(torch.FloatTensor).cuda()
                    output = D(E(torch.cat([swh[:, dt_da:], swh_a_ens[k:k + 1]], dim=1), mwp, u_wind, v_wind), lead_time, 1)

                    _s.append(output[0])
                    _m.append(output[1])
                    _u.append(output[2])
                    _v.append(output[3])

                _s = torch.stack(_s, dim=1)
                _m = torch.stack(_m, dim=1)
                _u = torch.stack(_u, dim=1)
                _v = torch.stack(_v, dim=1)

                _swh.append(_s)
                _mwp.append(_m)
                _u_wind.append(_u)
                _v_wind.append(_v)

                swh_f_ens.append(_s[:, :dt_da])

            _swh = torch.cat(_swh).mean(0, keepdim=True)
            _mwp = torch.cat(_mwp).mean(0, keepdim=True)
            _u_wind = torch.cat(_u_wind).mean(0, keepdim=True)
            _v_wind = torch.cat(_v_wind).mean(0, keepdim=True)

            # assimilation ################################################################
            swh_f_ens = torch.cat(swh_f_ens).view(N, -1).transpose(0, 1).cpu().data.numpy()

            # Mean and covariance of forecast
            swh_f_mean = np.average(swh_f_ens, axis=1)
            P_f        = np.cov(swh_f_ens)

            _insitu = np.array(insitu[t:t + dt_da])
            _insitu[np.isnan(_insitu)] = 0
            _mask = np.ones_like(_insitu)
            _mask[np.isnan(insitu[t:t + dt_da])] = 0
            
            swht = _swh[:, :dt_da].cpu().data.numpy()
            swht[0, :, 0, -10, 5] = _mask[:, 0] * _insitu[:, 0] + (1 - _mask[:, 0]) * swht[0, :, 0, -10, 5]
            swht[0, :, 0,  -9, 4] = _mask[:, 1] * _insitu[:, 1] + (1 - _mask[:, 1]) * swht[0, :, 0,  -9, 4]
            swht[0, :, 0, -10, 4] = _mask[:, 2] * _insitu[:, 2] + (1 - _mask[:, 2]) * swht[0, :, 0, -10, 4]
            swht[0, :, 0,  -8, 5] = _mask[:, 3] * _insitu[:, 3] + (1 - _mask[:, 3]) * swht[0, :, 0,  -8, 5]
            swht[0, :, 0, -10, 7] = _mask[:, 4] * _insitu[:, 4] + (1 - _mask[:, 4]) * swht[0, :, 0, -10, 7]
            swht[0, :, 0, -11, 8] = _mask[:, 5] * _insitu[:, 5] + (1 - _mask[:, 5]) * swht[0, :, 0, -11, 8]
            swht[0, :, 0, -10, 6] = _mask[:, 6] * _insitu[:, 6] + (1 - _mask[:, 6]) * swht[0, :, 0, -10, 6]
            swht  = np.reshape(swht, (-1))
            swh_o = H @ swht

            # Covariance inflation
            rho, ratio = inflation_rev(swh_f_mean, P_f, swh_o, H, sigma_o, n, m, rho_min, rho_max, kappa, rho, ratio)
            
            for k in range(N):
                swh_f_ens[:, k] = swh_f_mean + np.sqrt(rho) * (swh_f_ens[:, k] - swh_f_mean)
                
            if m == 0:
                swh_a_ens = np.copy(swh_f_ens)
            else:
                swh_a_ens = EnKF_para(swh_f_ens, swh_f_mean, swh_o, H, n, m, N, sigma_o, obs_dist, r_local)
                
            # Mean and covarina of analysis
            swh_a_mean = np.average(swh_a_ens, axis=1)
            P_a        = np.cov(swh_a_ens)
            ################################################################################

            inferA[0].append(_swh[0, :, 0].cpu().data.numpy() * swh_std + swh_mean)
            inferA[1].append(_mwp[0, :, 0].cpu().data.numpy() * mwp_std + mwp_mean)
            inferA[2].append(_u_wind[0, :, 0].cpu().data.numpy() * u_wind_std + u_wind_mean)
            inferA[3].append(_v_wind[0, :, 0].cpu().data.numpy() * v_wind_std + v_wind_mean)

            label[0].append(y_wave[t:t + time_step[1], 0] * swh_std + swh_mean)
            label[1].append(y_wave[t:t + time_step[1], 1] * mwp_std + mwp_mean)
            label[2].append(y_wind[t:t + time_step[1], 0] * u_wind_std + u_wind_mean)
            label[3].append(y_wind[t:t + time_step[1], 1] * v_wind_std + v_wind_mean)

            path = 'output/AR+ENKF/[{}-{:02d}-{:02d}-{:02d}]-({}-{:02d}-{:02d}-{:02d})'.format(date[0].year, date[0].month, date[0].day, date[0].hour, date[1].year, date[1].month, date[1].day, date[1].hour)

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(path + '/swh.npy', inferA[0][-1])
            np.save(path + '/mwp.npy', inferA[1][-1])
            np.save(path + '/u_wind.npy', inferA[2][-1])
            np.save(path + '/v_wind.npy', inferA[3][-1])

            np.save(path + '/swh_gt.npy', label[0][-1])
            np.save(path + '/mwp_gt.npy', label[1][-1])
            np.save(path + '/u_wind_gt.npy', label[2][-1])
            np.save(path + '/v_wind_gt.npy', label[3][-1])

            RMSE[0].append(Compute_RMSE(inferA[0][-1], label[0][-1]))
            RMSE[1].append(Compute_RMSE(inferA[1][-1], label[1][-1]))
            RMSE[2].append(Compute_RMSE(inferA[2][-1], label[2][-1]))
            RMSE[3].append(Compute_RMSE(inferA[3][-1], label[3][-1]))

            with open(path + '/RMSE.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['lead time', 'swh', 'mwp', 'u_wind', 'v_wind'])

                for i in range(time_step[1]):
                    writer.writerow(['{}'.format(i + 1), RMSE[0][-1][i], RMSE[1][-1][i], RMSE[2][-1][i], RMSE[3][-1][i]])

            _swh = torch.from_numpy(swh_a_mean).type(torch.FloatTensor).cuda()
            _swh = _swh.view(1, dt_da, 1, 34, 30)

            swh = torch.cat([swh[:, dt_da:], _swh], dim=1) * mask
            mwp = torch.cat([mwp[:, dt_da:], _mwp[:, :dt_da]], dim=1) * mask
            u_wind = torch.cat([u_wind[:, dt_da:], _u_wind[:, :dt_da]], dim=1)
            v_wind = torch.cat([v_wind[:, dt_da:], _v_wind[:, :dt_da]], dim=1)

            date[2] = date[0] + datetime.timedelta(hours=dt_da)

            # 약 10일마다 cold start
            if (t + dt_da) % 48 == 0:
                # (date[0].month != date[2].month) or (date[0].day == 10 and date[2].day == 11) or (date[0].day == 20 and date[2].day == 21):
                with open('output/AR+ENKF/RMSE_[{}-{:02d}-{:02d}-{:02d}]-[{}-{:02d}-{:02d}-{:02d}].csv'.format(recent_date.year, recent_date.month, recent_date.day, recent_date.hour, date[0].year, date[0].month, date[0].day, date[0].hour), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(['lead time', 'swh', 'mwp', 'u_wind', 'v_wind'])

                    _RMSE = np.array(RMSE)
                    _RMSE = np.mean(_RMSE[:, recent_index:], axis=1)

                    for i in range(time_step[1]):
                        writer.writerow(['{}'.format(i + 1), _RMSE[0, i], _RMSE[1, i], _RMSE[2, i], _RMSE[3, i]])

                recent_index = len(RMSE[0])
                recent_date  = date[2]

                wind = Variable(torch.from_numpy(np.array([y_wind[t + dt_da - time_step[0]:t + dt_da]]))).type(torch.FloatTensor).cuda()
                wave = Variable(torch.from_numpy(np.array([y_wave[t + dt_da - time_step[0]:t + dt_da]]))).type(torch.FloatTensor).cuda()

                swh = wave[:, :, 0:1]
                mwp = wave[:, :, 1:2]
                u_wind = wind[:, :, 0:1]
                v_wind = wind[:, :, 1:2]

                swh_a_ens = []

            date[0] += datetime.timedelta(hours=dt_da)
            date[1] += datetime.timedelta(hours=dt_da)

            print('AR+ENKF', t + 1, '/', time)
