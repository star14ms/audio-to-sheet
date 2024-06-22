import matplotlib.pyplot as plt
import librosa.display
import numpy as np 
import sys
import torch
from modules.constants import NOTES


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\u001b[38;5;208m'

    GRAY = '\033[90m'
    BRIGHT = '\033[97m'

    c = [HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD, UNDERLINE, ORANGE]

    def test():
        for i in range(91, 100):
            print(i, f'\033[{i}mWarning: No active frommets remain. Continue?{bcolors.ENDC}', end='\n')
        for color in bcolors.c:
            print(color + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)

    def according_to_score(x):
        if x < 1:
            return bcolors.ENDC
        elif 1 <= x < 20:
            return bcolors.FAIL
        elif 20 <= x < 40:
            return bcolors.ORANGE
        elif 40 <= x < 60:
            return bcolors.WARNING
        elif 60 <= x < 80:
            return bcolors.OKGREEN
        elif 80 <= x < 95:
            return bcolors.OKBLUE
        elif 95 <= x < 99:
            return bcolors.HEADER
        else:
            return bcolors.BOLD
    
    def according_to_chance(x):
        if x < 0.5:
            return bcolors.ENDC
        elif 0.5 <= x < 5:
            return bcolors.FAIL
        elif 5 <= x < 20:
            return bcolors.ORANGE
        elif 20 <= x < 50:
            return bcolors.WARNING
        elif 50 <= x < 80:
            return bcolors.OKGREEN
        elif 80 <= x < 95:
            return bcolors.OKBLUE
        elif 95 <= x < 99:
            return bcolors.HEADER
        else:
            return bcolors.BOLD

    def ANSI_codes():
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[38;5;" + code + "m" + code.ljust(4))
            print(u"\u001b[0m")


def plot_spectrogram(spectrogram, sr):
    # Display the spectrogram
    plt.figure(figsize=(15, 9))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def plot_spectrogram_simplified(spectrogram, sr, hop_length):
    plt.figure(figsize=(15, 9))
    plt.pcolormesh(
        librosa.frames_to_time(np.arange(spectrogram.shape[1]+1), sr=sr, hop_length=hop_length), 
        np.arange(spectrogram.shape[0]+1), 
        spectrogram,
        cmap='magma',
        shading='flat', 
    )
    plt.colorbar(format='%+2.0f dB')
    # plt.xticks(np.arange(spectrogram.shape[1]+1, 30), np.arange(spectrogram.shape[1]+1, 30))
    plt.yticks(np.arange(3, spectrogram.shape[0]+1, 12), ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
    plt.title('Spectrogram Simplified')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def plot_spectrogram_hightlighting_pressing_notes(spectrogram, pressed_notes, sr, hop_length):
    _, ax = plt.subplots(figsize=(15, 9))
    img = ax.pcolormesh(
        librosa.frames_to_time(np.arange(spectrogram.shape[1]+1), sr=sr, hop_length=hop_length), 
        np.arange(spectrogram.shape[0]+1), 
        spectrogram,
        cmap='magma',
    )

    unit_x_on_canvas = ax.dataLim.x1 / spectrogram.shape[1]
    unit_y_on_canvas = ax.dataLim.y1 / spectrogram.shape[0]
    for i, pressed_notes_moment in enumerate(pressed_notes):
        for j, pressed in enumerate(pressed_notes_moment):
            if pressed:
                ax.scatter(unit_x_on_canvas*j, unit_y_on_canvas*i+unit_y_on_canvas/2, color='red', s=10)

    plt.colorbar(img, format='%+2.0f dB', ax=ax)
    ax.set_yticks(np.arange(3, spectrogram.shape[0]+1, 12), ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
    ax.set_title('Spectrogram Simplified')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.show()


def print_matching_highlight(list1, list2, prefix1='Correct: ', prefix2='Predict: ', header='         A B C' + ' D EF G A B C' * 7):
    str1 = prefix1
    str2 = prefix2
    for i, (item_t, item_y) in enumerate(zip(list1, list2)):
        if i % 12 == 3:
            str1 += ' '
            str2 += ' '
        if item_t == 0:
            str1 += f"\033[90m{item_t}\033[0m"
        else:
            str1 += f"\033[97m{item_t}\033[0m"
        if item_t == 0 and item_y == 0:
            str2 += f"\033[90m{item_y}\033[0m"
        elif item_t != 0 and item_y != 0:
            str2 += f"\033[92m{item_y}\033[0m"
        else:
            str2 += f"\033[91m{item_y}\033[0m"

    return print(str1, str2, header, sep='\n')


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def print_values_colored_by_min_max_normalizing(values):
    scores = min_max_normalize(torch.tensor(values))
    torch.set_printoptions(sci_mode=False)
    
    text = ''
    for i, score in enumerate(scores):
        if i % 12 == 3:
            text += ' '
        char = '#' if '#' in NOTES[i] else NOTES[i][0]
        text += f'{bcolors.according_to_score(score*100)}{char}{bcolors.ENDC}'
    print(text)


if __name__ == '__main__':
    bcolors.test()
    