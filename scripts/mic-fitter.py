#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from src.nanobio_core.epic_cardio.measurement_load import load_measurement, wl_map_to_wells
from src.nanobio_core.epic_cardio.data_correction import correct_well
from src.nanobio_core.image_fitting.single_cell_evaluator import CardioMicSingleCellEvaluator
import cv2
from datetime import datetime


WIDTH = 80
WELL_IDS = [['C1', 'C2', 'C3', 'C4'], ['B1', 'B2', 'B3', 'B4'], ['A1', 'A2', 'A3', 'A4']] # Tükrözés korrigálva
def flatten(t):
    return [item for sublist in t for item in sublist]
WELL_NAMES = flatten(WELL_IDS)

EPIC_PATHS = ['../../data/classification_datasets/fibronectin/', '../../data/classification_datasets/noncoated/']
MIC_PATHS = ['../../data/classification_datasets/lognorm-images/fibronectin/annotated/','../../data/classification_datasets/lognorm-images/noncoated/annotated/']
PRED_PATHS = ['/home/balint/projects/lognorm-images/fibronectin/','/home/balint/projects/lognorm-images/noncoated/']
RESULT_PATH = '../../data/classification_datasets/total-results'

# # Well mappa elérési útvonala
# CARDIO_PATH = '../../data/classification_datasets/fibronectin/20210526_LCLC_fn//epic_data/'
# MIC_PATH = '../../data/classification_datasets/lognorm-images/fibronectin/annotated/20210526_LCLC_fn/'

from multiprocessing import Pool, cpu_count


def evaluate_well(well_id, time, well_data, CARDIO_PATH, MIC_PATH, PRED_PATH, RESULT_PATH, adjacent=False):
    start = datetime.now()
    mic_path = f'{MIC_PATH}/{well_id}.jpeg'
    annot_path = f'{MIC_PATH}/{well_id}_seg.npy'
    pred_path = f'{PRED_PATH}/{well_id}_pred_seg.npy'
    params_path = f'{CARDIO_PATH}/result/metadata/{well_id}_map_params.json'
    
    if not os.path.exists(params_path) or not os.path.exists(mic_path) or not os.path.exists(annot_path):
        print(not os.path.exists(params_path), not os.path.exists(mic_path), not os.path.exists(annot_path))
        return

    img = cv2.imread(mic_path)
    well = well_data[well_id]

    mask = np.load(annot_path, allow_pickle=True).item()['masks'][0]
    mask_pred = np.load(pred_path, allow_pickle=True).item()['masks-tuned']

    filter_params = {
        'area': (0, np.Inf),
        'max_value': (100, np.Inf),
        'adjacent': adjacent,
    }

    print(well_id)
    evaluator = CardioMicSingleCellEvaluator(time, well, img, mask, mask_pred, params_path, load_selection=None, filter_params=filter_params)

    evaluator.select_all()

    evaluator.save(RESULT_PATH, well_id, 4)
    print(f"{CARDIO_PATH} Well {well_id} {datetime.now() - start}")

def evaluate(CARDIO_PATH, MIC_PATH, PRED_PATH, RESULT_PATH, adjacent=False):
    
    print(CARDIO_PATH)

    # Betölti a 3x4-es well képet a projekt mappából.
    WL_map, time = load_measurement(CARDIO_PATH)
    # Itt szétválasztásra kerülnek a wellek. Betöltéskor egy 240x320-as képen található a 3x4 elrendezésű 12 well.
    wells = wl_map_to_wells(WL_map, flip=True)
    phases = list(np.where((np.diff(time)) > 60)[0] + 1)
    [(n+1, p) for n, p in enumerate(phases)]

    # Sejt szűrés a wellekből.
    well_data = {}
    cut_point = 1
    signal_start = 0 if cut_point == 0 or cut_point > len(phases) else phases[cut_point - 1]
    signal_start += 5
    time = time[signal_start:]
    time -= time[0]

    for name in WELL_NAMES:
        # print("Parsing", name)
        well_tmp = wells[name]
        well_tmp = well_tmp[signal_start:]
        well_corr = correct_well(well_tmp)
        well_data[name] = well_corr
    # print("Parsing finished!")

    pool = Pool(processes=(cpu_count() - 1))

    for well_id in WELL_NAMES:
        pool.apply_async(evaluate_well, args=(well_id, time, well_data, CARDIO_PATH, MIC_PATH, PRED_PATH, RESULT_PATH, adjacent))
        # evaluate_well(well_id, time, well_data, CARDIO_PATH, MIC_PATH, PRED_PATH, RESULT_PATH, adjacent)

    pool.close()
    pool.join()
        
epic_paths, mic_paths, pred_paths = [], [], []
for path in os.listdir(EPIC_PATHS[0]):
    if '_fn' in path:
        epic_path = os.path.join(EPIC_PATHS[0], path, 'epic_data')
        if os.path.exists(epic_path):
            epic_paths.append(epic_path)
            mic_paths.append(os.path.join(MIC_PATHS[0], path))
            pred_paths.append(os.path.join(PRED_PATHS[0], path))
            
for path in os.listdir(EPIC_PATHS[1]):
    if '_nonc' in path:
        epic_path = os.path.join(EPIC_PATHS[1], path, 'epic_data')
        if os.path.exists(epic_path):
            epic_paths.append(epic_path)
            mic_paths.append(os.path.join(MIC_PATHS[1], path))
            pred_paths.append(os.path.join(PRED_PATHS[1], path))

for cardio, mic, pred in zip(epic_paths, mic_paths, pred_paths):
    result_path = os.path.join(RESULT_PATH, cardio.split('/')[-3], cardio.split('/')[-2])
    
    print(result_path)
    # RESULT_PATH = os.path.join(cardio, 'result')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    

    evaluate(cardio, mic, pred, result_path, adjacent=False)