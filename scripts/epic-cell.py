import numpy as np
import argparse
import os
from src.nanobio_core.epic_cardio.measurement_load import load_measurement
from src.nanobio_core.epic_cardio.cell_selector import WellArrayLineSelector
from src.nanobio_core.epic_cardio.data_correction import correct_well
from src.nanobio_core.epic_cardio.math_ops import calculate_cell_maximas

# parser = argparse.ArgumentParser(description='Epic Cardio Segmentation')
# parser.add_argument('--path', type=str, help='measurement folder path', required=True)
# args = parser.parse_args()

WIDTH = 80
WELL_IDS = [['C1', 'C2', 'C3', 'C4'], ['B1', 'B2', 'B3', 'B4'], ['A1', 'A2', 'A3', 'A4']] # Tükrözés korrigálva
def flatten(t):
    return [item for sublist in t for item in sublist]
WELL_NAMES = flatten(WELL_IDS)

def clear_line():
    import sys
    sys.stdout.write('\x1b[2K')

def run(args):
    DIR_PATH = args.path
    DEST_PATH = args.dest
    print(f"Loading measurement from {DIR_PATH}.")
    # Betölti a 3x4-es well képet a projekt mappából.
    WL_map, time = load_measurement(DIR_PATH)
    # Itt szétválasztásra kerülnek a wellek. Betöltéskor egy 240x320-as képen található a 3x4 elrendezésű 12 well.
    wells = {name: WL_map[:, i : i+WIDTH, j:j+WIDTH]  for names, i in zip(WELL_IDS, range(0, 240, WIDTH)) for name, j in zip(names, range(0, 320, WIDTH))}
    # print("Measurement loaded!")
    phases = list(np.where((np.diff(time)) > 60)[0] + 3)
    
    print(time.shape)

    # Sejt szűrés a wellekből.
    well_data = {}

    # = 0 legyen ha nem kell vágni, amúgy a másik
    signal_start = 0
    time_red = time.copy()

    print(phases)

    if len(phases) > 0:
        signal_start = phases[-1]
        signal_start += 5
        print(f"Cutting measurement at {signal_start}")
        time_red = time[signal_start:].copy()
        time_red -= time_red[0]

    for name in WELL_NAMES:
        print("Parsing", name, end='\r')
        well_tmp = wells[name][signal_start:, :, :]
        well_corr = correct_well(well_tmp)
        ptss, lines = calculate_cell_maximas(well_corr, min_threshold=.1*1000, max_threshold=3*1000)
        well_data[name] = (well_corr, ptss, lines, time_red)
    clear_line()
    print("Parsing finished!")

    # Sejtválogatás. A felugró ablakban lehet(nem biztos, hogy előtérbe kerül).
    # A kiválasztás végén automatikusan bezáródik.
    if(args.selector):
        selector = WellArrayLineSelector(well_data, time_red, phases)

    for name in WELL_NAMES:
        print("Saving", name, end='\r')
        np.savez(os.path.join(DEST_PATH, name), well=well_data[name][0], coords=well_data[name][1], lines=well_data[name][2], time=well_data[name][3], )
        np.savetxt(os.path.join(DEST_PATH, name) + '.csv', np.vstack(well_data[name][3],well_data[name][2]), delimiter=',')
    clear_line()
    print("Saving finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Epic Cardio Single Cell Segmentation')
    parser.add_argument('-p','--path', type=str, help='measurement folder path', required=True)
    parser.add_argument('-d','--dest', type=str, help='measurement folder path', required=True)
    parser.add_argument('-s', '--selector', action='store_true')
#     parser.add_argument('-t', '--output-type')
    args = parser.parse_args()
    run(args)
