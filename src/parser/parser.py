import os
import numpy as np
import pandas as pd
import tiffile as tif
# from ..nanobio_core.epic_cardio.defs import WELL_NAMES, flatten
import argparse
import pywt
import json

WELL_IDS = [['C1', 'C2', 'C3', 'C4'], ['B1', 'B2', 'B3', 'B4'], ['A1', 'A2', 'A3', 'A4']] # Tükrözés korrigálva

def flatten(t):
    return [item for sublist in t for item in sublist]

WELL_NAMES = sorted(flatten(WELL_IDS))

def create_dirs_ifne(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def filter_dataset(dataset):
    value = dataset.copy()
    filtered = np.zeros((value.shape[0], value.shape[1] - 1))
    for i in range(value.shape[0]):
        if np.sum(value[i, 1:]) == 0:
            filtered[i, :] = value[i, 1:]
        else:
            sig = np.array(value[i, 1:])
            w = pywt.Wavelet('sym17')
            maxlev = pywt.dwt_max_level(sig.shape[0], w.dec_len)
            threshold = 0.5

            coeffs = pywt.wavedec(sig, w, level=maxlev)

            for j in range(1, len(coeffs)):
                coeffs[j] = pywt.threshold(coeffs[j], threshold*max(coeffs[j]))
            filtered[i, :] = pywt.waverec(coeffs, w)[:sig.shape[0]]
    filtered = np.subtract(filtered, np.expand_dims(filtered[:, 0], -1))
    return filtered

def filter_3D_data(dataset):
    value = dataset.copy()
    filtered = np.zeros((dataset.shape[1] * dataset.shape[2], value.shape[0] - 1))
    value = value.reshape(value.shape[0], -1).T
    for i in range(value.shape[0]):
        if np.sum(value[i, 1:]) == 0:
            filtered[i, :] = value[i, 1:]
        else:
            sig = np.array(value[i, 1:])
            w = pywt.Wavelet('sym17')
            maxlev = pywt.dwt_max_level(sig.shape[0], w.dec_len)
            threshold = 0.5

            coeffs = pywt.wavedec(sig, w, level=maxlev)

            for j in range(1, len(coeffs)):
                coeffs[j] = pywt.threshold(coeffs[j], threshold*max(coeffs[j]))

            filtered[i, :] = pywt.waverec(coeffs, w)[:sig.shape[0]]
    filtered = np.subtract(filtered, np.expand_dims(filtered[:, 0], -1))
    filtered = filtered.T.reshape((dataset.shape[0] - 1, dataset.shape[1], dataset.shape[2]))

    return filtered

def evaluate_single(path, inf, res, cell_types):
    print(path, inf)
    src_path = path
    platemap = pd.read_excel(os.path.join(inf, 'platemap.xlsx'), header=None).values.tolist()
    platemap = [(w, p.split(' ')[-1].lower()) for w, p in zip(WELL_NAMES, flatten(platemap))]
    platemap = [(w,p) for w, p in platemap if p in cell_types and 
                os.path.exists(os.path.join(path, f'{w}_seg.npz'))]
    stats_path = [ [os.path.join(path, f'.metadata/{w}_stats.json'), p] for w, p in platemap if p in cell_types and 
                os.path.exists(os.path.join(path, f'{w}_seg.npz'))]

    if len(platemap) == 0:
        return

    for (pth, p) in stats_path:
        if not os.path.exists(os.path.join(res, p)):
            os.makedirs(os.path.join(res, p))

        with open(pth, 'r') as fp:
            obj = json.load(fp)
            op_obj = {key: [] for key in obj.keys()}
            if os.path.exists(os.path.join(res, p, f'{p}_stats.json')):
                with open(os.path.join(res, p, f'{p}_stats.json'), 'r') as op:
                    op_obj = json.load(op)

            for key, value in obj.items():
                op_obj[key].append(value)

            with open(os.path.join(res, p, f'{p}_stats.json'), 'w') as op:
                json.dump(op_obj, op)

    signals = []
    for w, p in platemap:  
        src_max_path = os.path.join(src_path, f'{w}_max_signals.csv')
        signals.append(np.asarray(pd.read_csv(src_max_path).iloc[:, 1:]))
    signals = np.vstack(signals)
    diff = np.diff(np.mean(signals, axis=0))
    err = np.argwhere(np.logical_or(diff > 5, diff < -5))
    err = err.reshape((-1,))
    err.sort()
    peaks = np.argwhere(np.diff(err) == 1)
    peaks = peaks.reshape((-1, ))
    err = np.unique(np.concatenate([err[peaks], err[peaks + 1]]))
    data_range = np.arange(signals.shape[1])
    org = data_range.copy()
    data_range = np.delete(data_range, err, axis=0)

    for w, p in platemap:
        print(f'Parsing {w} from {path}')
        interpolate = False
        res_path = os.path.join(res, p)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
            
        src_areas_path = os.path.join(src_path, f'{w}_areas.csv')  
        res_areas_path = os.path.join(res_path, f'{p}_areas.csv')
        data = pd.read_csv(src_areas_path).iloc[:, 1:]
        data.to_csv(res_areas_path, mode='a', header=None, index=None)
        del data
        
        src_max_path = os.path.join(src_path, f'{w}_max_signals.csv')
        res_max_path = os.path.join(res_path, f'{p}_max_signals.csv')
        data = pd.read_csv(src_max_path).iloc[:, 1:]
        time = np.round(np.array(data.iloc[0, :]), 0).astype(int)
        data = np.asarray(data.iloc[1:, :])
        time_avg_max = np.round(np.mean(np.diff(time)), 0)
        
        new_data = np.zeros(data.shape)
        data = np.delete(data, err, axis=1)
        for i in range(new_data.shape[0]):
            new_data[i] = np.interp(org, data_range, data[i])
        data = new_data

        if time_avg_max == 12:
            interpolate = True
            new_time = np.round(np.linspace(0,max(time), int(max(time) / 3)),0).astype(int)
            new_time_red = np.interp(new_time, time, time)[:None:3]

        if '20210407_LCLC_H838_nonc' in path:
            interpolate = True
            new_time_red = np.round(np.linspace(0, max(time), 600),0).astype(int)
            
        if interpolate:
            print(f'Interpolating {path}')
            new_data = np.zeros((data.shape[0], new_time_red.shape[0]))
            for i in range(data.shape[0]):
                new_data[i] = np.interp(new_time_red, time, data[i])
            data = new_data

        pd.DataFrame(data).to_csv(res_max_path, mode='a', header=None, index=None)
        
        src_int_path = os.path.join(src_path, f'{w}_int_signals.csv')  
        res_int_path = os.path.join(res_path, f'{p}_int_signals.csv')
        data = np.asarray(pd.read_csv(src_int_path).iloc[1:, 1:])
        
        new_data = np.zeros(data.shape)
        data = np.delete(data, err, axis=1)
        for i in range(new_data.shape[0]):
            new_data[i] = np.interp(org, data_range, data[i])
        data = new_data

        if interpolate:
            new_data = np.zeros((data.shape[0], new_time_red.shape[0]))
            for i in range(data.shape[0]):
                new_data[i] = np.interp(new_time_red, time, data[i])
            data = new_data

        pd.DataFrame(data).to_csv(res_int_path, mode='a', header=None, index=None)
    
        seg = np.load(os.path.join(src_path, f'{w}_seg.npz'))
        # time, im_cardio, im_watershed, im_cover, im_mic, im_marker, im_marker_sig, im_mic_sig = seg['time'], seg['cardio'], seg['cardio_watershed'], seg['cardio_cover'], seg['mic'].astype('uint8'), seg['marker'].astype('uint16'), seg['marker_singular'].astype('uint16'), seg['mic_singular'].astype('uint8')
        time, im_cardio, im_watershed, im_cover, im_pred = seg['time'], seg['cardio'], seg['cardio_watershed'], seg['cardio_cover'], seg['cardio_pred']
        cardio_path = os.path.join(res_path, 'im_cardio')
        watershed_path = os.path.join(res_path, 'im_watershed')
        pred_path = os.path.join(res_path, 'im_pred')
        cover_path = os.path.join(res_path, 'im_cover')
        # mic_path = os.path.join(res_path, 'im_mic')
        # mic_sig_path = os.path.join(res_path, 'im_mic_sig')
        # marker_path = os.path.join(res_path, 'im_marker')
        # marker_sig_path = os.path.join(res_path, 'im_marker_sig')
        

        # create_dirs_ifne([cardio_path, watershed_path, cover_path, mic_path, 
        #                   mic_sig_path, marker_path, marker_sig_path])
        create_dirs_ifne([cardio_path, watershed_path, cover_path, pred_path])

        ln = len(os.listdir(cardio_path))
        for i in range(im_cardio.shape[0]):
            if interpolate:
                new_cardio = np.zeros((new_time_red.shape[0], im_cardio.shape[2], im_cardio.shape[3]))
                new_cover = np.zeros((new_time_red.shape[0], im_cover.shape[2], im_cover.shape[3]))
                for j in range(im_cardio.shape[2]):
                    for k in range(im_cardio.shape[3]):
                        if err.shape[0] > 0:
                            tmp = np.delete(im_cardio[i, :, j, k], err, axis=0)
                            new_cardio[:, j, k] = np.interp(new_time_red, time, np.interp(org, data_range, tmp))
                            tmp = np.delete(im_cover[i, :, j, k], err, axis=0)
                            new_cover[:, j, k] = np.interp(new_time_red, time, np.interp(org, data_range, tmp))
                        else:
                            new_cardio[:, j, k] = np.interp(new_time_red, time, im_cardio[i, :, j, k])
                            new_cover[:, j, k] = np.interp(new_time_red, time, im_cover[i, :, j, k])
                tif.imsave(os.path.join(cardio_path, f'image_{ln}.tiff'), new_cardio)
                tif.imsave(os.path.join(cover_path, f'image_{ln}.tiff'), new_cover)
            else:
                new_cardio = im_cardio[i].copy()
                new_cover = im_cover[i].copy()
                if err.shape[0] > 0:
                    for j in range(im_cardio.shape[2]):
                        for k in range(im_cardio.shape[3]):
                            tmp = np.delete(im_cardio[i, :, j, k], err, axis=0)
                            new_cardio[:, j, k] = np.interp(org, data_range, tmp)
                            tmp = np.delete(im_cover[i, :, j, k], err, axis=0)
                            new_cover[:, j, k] = np.interp(org, data_range, tmp)
                tif.imsave(os.path.join(cardio_path, f'image_{ln}.tiff'), new_cardio)
                tif.imsave(os.path.join(cover_path, f'image_{ln}.tiff'), new_cover)
            # tif.imsave(os.path.join(mic_path, f'image_{ln}.tiff'), im_mic[i])
            # tif.imsave(os.path.join(mic_sig_path, f'image_{ln}.tiff'), im_mic_sig[i])
            # tif.imsave(os.path.join(marker_path, f'image_{ln}.tiff'), im_marker[i])
            # tif.imsave(os.path.join(marker_sig_path, f'image_{ln}.tiff'), im_marker_sig[i])
            ln += 1

        ln = len(os.listdir(watershed_path))
        for i in range(im_watershed.shape[0]):
            if interpolate:
                new_watershed = np.zeros((new_time_red.shape[0], im_watershed.shape[2], im_watershed.shape[3]))
                for j in range(im_watershed.shape[2]):
                    for k in range(im_watershed.shape[3]):
                        if err.shape[0] > 0:
                            tmp = np.delete(im_watershed[i, :, j, k], err, axis=0)
                            new_watershed[:, j, k] = np.interp(new_time_red, time, np.interp(org, data_range, tmp))
                        else:
                            new_watershed[:, j, k] = np.interp(new_time_red, time, im_watershed[i, :, j, k])
                tif.imsave(os.path.join(watershed_path, f'image_{ln}.tiff'), new_watershed)
            else:
                new_watershed = im_watershed[i].copy()
                if err.shape[0] > 0:
                    for j in range(im_watershed.shape[2]):
                        for k in range(im_watershed.shape[3]):
                            tmp = np.delete(im_watershed[i, :, j, k], err, axis=0)
                            new_watershed[:, j, k] = np.interp(org, data_range, tmp)
                tif.imsave(os.path.join(watershed_path, f'image_{ln}.tiff'), new_watershed)
            ln += 1

        ln = len(os.listdir(pred_path))
        for i in range(im_pred.shape[0]):
            if interpolate:
                new_pred = np.zeros((new_time_red.shape[0], im_pred.shape[2], im_pred.shape[3]))
                for j in range(im_pred.shape[2]):
                    for k in range(im_pred.shape[3]):
                        if err.shape[0] > 0:
                            tmp = np.delete(im_pred[i, :, j, k], err, axis=0)
                            new_pred[:, j, k] = np.interp(new_time_red, time, np.interp(org, data_range, tmp))
                        else:
                            new_pred[:, j, k] = np.interp(new_time_red, time, im_pred[i, :, j, k])
                tif.imsave(os.path.join(pred_path, f'image_{ln}.tiff'), new_pred)
            else:
                new_pred = im_pred[i].copy()
                if err.shape[0] > 0:
                    for j in range(im_pred.shape[2]):
                        for k in range(im_pred.shape[3]):
                            tmp = np.delete(im_pred[i, :, j, k], err, axis=0)
                            new_pred[:, j, k] = np.interp(org, data_range, tmp)
                tif.imsave(os.path.join(pred_path, f'image_{ln}.tiff'), new_pred)
            ln += 1
    

def create(src_path, info_path, res_path, cell_types):
    dirs = []
    src_dirs = sorted(os.listdir(src_path))
    for d in src_dirs:
        pth = os.path.join(src_path, d)
        inf = os.path.join(info_path, d)
        if os.path.exists(pth) and os.path.exists(inf):
            dirs.append((os.path.join(src_path, d), os.path.join(info_path, d, 'info')))
    dirs = sorted(dirs)    
    for (d, i)in dirs:
        evaluate_single(d, i, res_path, cell_types)