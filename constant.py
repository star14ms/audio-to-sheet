import numpy as np


NOTE_FREQUENCIES = {
    'A0': 27.50, 'A#0': 29.14, 'B0': 30.87,
    'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89,
    'E1': 41.20, 'F1': 43.65, 'F#1': 46.25, 'G1': 49.00,
    'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78,
    'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00,
    'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56,
    'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00,
    'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25,
    'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99,
    'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51,
    'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98, 'G6': 1567.98,
    'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53,
    'C7': 2093.00, 'C#7': 2217.46, 'D7': 2349.32, 'D#7': 2489.02,
    'E7': 2637.02, 'F7': 2793.83, 'F#7': 2959.96, 'G7': 3135.96,
    'G#7': 3322.44, 'A7': 3520.00, 'A#7': 3729.31, 'B7': 3951.07,
    'C8': 4186.01,
}

NOTE_TO_KEY = {
    'A0': 21, 'A#0': 22, 'B0': 23,
    'C1': 24, 'C#1': 25, 'D1': 26, 'D#1': 27,
    'E1': 28, 'F1': 29, 'F#1': 30, 'G1': 31,
    'G#1': 32, 'A1': 33, 'A#1': 34, 'B1': 35,
    'C2': 36, 'C#2': 37, 'D2': 38, 'D#2': 39,
    'E2': 40, 'F2': 41, 'F#2': 42, 'G2': 43,
    'G#2': 44, 'A2': 45, 'A#2': 46, 'B2': 47,
    'C3': 48, 'C#3': 49, 'D3': 50, 'D#3': 51,
    'E3': 52, 'F3': 53, 'F#3': 54, 'G3': 55,
    'G#3': 56, 'A3': 57, 'A#3': 58, 'B3': 59,
    'C4': 60, 'C#4': 61, 'D4': 62, 'D#4': 63,
    'E4': 64, 'F4': 65, 'F#4': 66, 'G4': 67,
    'G#4': 68, 'A4': 69, 'A#4': 70, 'B4': 71,
    'C5': 72, 'C#5': 73, 'D5': 74, 'D#5': 75,
    'E5': 76, 'F5': 77, 'F#5': 78, 'G5': 79,
    'G#5': 80, 'A5': 81, 'A#5': 82, 'B5': 83,
    'C6': 84, 'C#6': 85, 'D6': 86, 'D#6': 87,
    'E6': 88, 'F6': 89, 'F#6': 90, 'G6': 91,
    'G#6': 92, 'A6': 93, 'A#6': 94, 'B6': 95,
    'C7': 96, 'C#7': 97, 'D7': 98, 'D#7': 99,
    'E7': 100, 'F7': 101, 'F#7': 102, 'G7': 103,
    'G#7': 104, 'A7': 105, 'A#7': 106, 'B7': 107,
    'C8': 108,
}


NOTES = [
    'A0', 'A#0', 'B0',
    'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1',
    'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
    'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
    'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
    'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
    'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
    'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7',
    'C8',
]

KEY_TO_NOTE = {v: k for k, v in NOTE_TO_KEY.items()}


# SCALES
C_MAJOR_SCALE = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
CS_MAJOR_SCALE = ['C#', 'D#', 'F', 'F#', 'G#', 'A#', 'C']
D_MAJOR_SCALE = ['D', 'E', 'F#', 'G', 'A', 'B', 'C#']
DS_MAJOR_SCALE = ['D#', 'F', 'G', 'G#', 'A#', 'C', 'D']
E_MAJOR_SCALE = ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#']
F_MAJOR_SCALE = ['F', 'G', 'A', 'A#', 'C', 'D', 'E']
FS_MAJOR_SCALE = ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'F']
G_MAJOR_SCALE = ['G', 'A', 'B', 'C', 'D', 'E', 'F#']
GS_MAJOR_SCALE = ['G#', 'A#', 'C', 'C#', 'D#', 'F', 'G']
A_MAJOR_SCALE = ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#']
AS_MAJOR_SCALE = ['A#', 'C', 'D', 'D#', 'F', 'G', 'A']
B_MAJOR_SCALE = ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#']

C_MINOR_SCALE = ['C', 'D', 'D#', 'F', 'G', 'G#', 'A#']
CS_MINOR_SCALE = ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B']
D_MINOR_SCALE = ['D', 'E', 'F', 'G', 'A', 'A#', 'C']
DS_MINOR_SCALE = ['D#', 'F', 'F#', 'G#', 'A#', 'B', 'C#']
E_MINOR_SCALE = ['E', 'F#', 'G', 'A', 'B', 'C', 'D']
F_MINOR_SCALE = ['F', 'G', 'G#', 'A#', 'C', 'C#', 'D#']
FS_MINOR_SCALE = ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E']
G_MINOR_SCALE = ['G', 'A', 'A#', 'C', 'D', 'D#', 'F']
GS_MINOR_SCALE = ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#']
A_MINOR_SCALE = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
AS_MINOR_SCALE = ['A#', 'C', 'C#', 'D#', 'F', 'F#', 'G#']
B_MINOR_SCALE = ['B', 'C#', 'D', 'E', 'F#', 'G', 'A']

SCALES = { 
    'C': C_MAJOR_SCALE, 'C#': CS_MAJOR_SCALE, 'D': D_MAJOR_SCALE, 'D#': DS_MAJOR_SCALE, 'E': E_MAJOR_SCALE, 'F': F_MAJOR_SCALE, 'F#': FS_MAJOR_SCALE, 'G': G_MAJOR_SCALE, 'G#': GS_MAJOR_SCALE, 'A': A_MAJOR_SCALE, 'A#': AS_MAJOR_SCALE, 'B': B_MAJOR_SCALE, 
    'Cm': C_MINOR_SCALE, 'C#m': CS_MINOR_SCALE, 'Dm': D_MINOR_SCALE, 'D#m': DS_MINOR_SCALE, 'Em': E_MINOR_SCALE, 'Fm': F_MINOR_SCALE, 'F#m': FS_MINOR_SCALE, 'Gm': G_MINOR_SCALE, 'G#m': GS_MINOR_SCALE, 'Am': A_MINOR_SCALE, 'A#m': AS_MINOR_SCALE, 'Bm': B_MINOR_SCALE,
}


# 88 keys on piano
C_KEYS = np.array([4, 16, 28, 40, 52, 64, 76, 88]) - 1
CS_KEYS = np.array([5, 17, 29, 41, 53, 65, 77]) - 1
D_KEYS = np.array([6, 18, 30, 42, 54, 66, 78]) - 1
DS_KEYS = np.array([7, 19, 31, 43, 55, 67, 79]) - 1
E_KEYS = np.array([8, 20, 32, 44, 56, 68, 80]) - 1
F_KEYS = np.array([9, 21, 33, 45, 57, 69, 81]) - 1
FS_KEYS = np.array([10, 22, 34, 46, 58, 70, 82]) - 1
G_KEYS = np.array([11, 23, 35, 47, 59, 71, 83]) - 1
GS_KEYS = np.array([12, 24, 36, 48, 60, 72, 84]) - 1
A_KEYS = np.array([1, 13, 25, 37, 49, 61, 73, 85]) - 1
AS_KEYS = np.array([2, 14, 26, 38, 50, 62, 74, 86]) - 1
B_KEYS = np.array([3, 15, 27, 39, 51, 63, 75, 87]) - 1

KYE_DICT_ON_PIANO = {
    'C': C_KEYS, 'C#': CS_KEYS, 'D': D_KEYS, 'D#': DS_KEYS, 'E': E_KEYS, 'F': F_KEYS, 'F#': FS_KEYS, 'G': G_KEYS, 'G#': GS_KEYS, 'A': A_KEYS, 'A#': AS_KEYS, 'B': B_KEYS,
}

