import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def wav_to_spectrogram(
    wav_path: str, 
    sr: int = 22050, 
    frame_length: float = 20.0, 
    frame_shift: float = 10.0, 
    n_mels: int = 80,
):
    '''
    Parameters
    ----------
        `wav_path`: `str`
            음성 파일 (.wav) 경로
        `sr`: `int`
            sampling rate - 1초 당 뽑을 오디오 시계열 데이터 수 (단위: Hz) [참고](https://youtu.be/OglqDo44zpQ?t=300)
        `frame_length`: `float`
            단일 발음 지속 시간 (단위: ms) [참고](https://youtu.be/OglqDo44zpQ?t=365)
        `frame_shift`: `float`
            number of samples between successive frames 
            연속적인 프레임 사이의 오디오 샘플 수 (단위: ms)
        `n_mels`: `int`
            number of Mel bands to generate 
            Mel 밴드 수 (단위: 개)\\
            Mel Scale: 저주파에서 상대적으로 예민한 사람의 청각기관을 반영한 Scale [참고](https://youtu.be/OglqDo44zpQ?t=402)
    '''
    signal, sample_rate = librosa.load(wav_path, sr=sr)
    
    n_fft = int(round(sample_rate * 0.001 * frame_length)) # default=441, default_origin=2048 (0.001: s -> ms)
    hop_length = int(round(sample_rate * 0.001 * frame_shift)) # default=220, default_origin=512
    
    # melspectrogram()에 들어갈 인자를 더 바꿔볼 여지 있음
    spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    return spectrogram


def visualize_wav_to_spectrogram(
    wav_path: str, 
    sr: int = 22050,
    frame_length: float = 20.0, 
    frame_shift: float = 10.0, 
    n_mels: int = 80,
):
    '''
    Parameters
    ----------
        `wav_path`: `str`
            음성 파일 (.wav) 경로
        `sr`: `int`
            sampling rate - 1초 당 뽑을 오디오 시계열 데이터 수 (단위: Hz) [참고](https://youtu.be/OglqDo44zpQ?t=300)
        `frame_length`: `float`
            단일 발음 지속 시간 (단위: ms) [참고](https://youtu.be/OglqDo44zpQ?t=365)
        `frame_shift`: `float`
            number of samples between successive frames 
            연속적인 프레임 사이의 오디오 샘플 수 (단위: ms)
        `n_mels`: `int`
            number of Mel bands to generate 
            Mel 밴드 수 (단위: 개)\\
            Mel Scale: 저주파에서 상대적으로 예민한 사람의 청각기관을 반영한 Scale [참고](https://youtu.be/OglqDo44zpQ?t=402)
    '''
    signal, sample_rate = librosa.load(wav_path, sr=sr)
    
    n_fft = int(round(sample_rate * 0.001 * frame_length))
    hop_length = int(round(sample_rate * 0.001 * frame_shift))
    
    spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    plt.plot(signal)
    plt.title('{} - Signal'.format(wav_path.split('/')[-1]))
    plt.show()
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.title('{} - Spectrogram'.format(wav_path.split('/')[-1]))
    plt.ylabel('n mels')
    plt.show()    
    
    # Passing through arguments to the Mel filters - melspectrogram() Examples 예제 시각화 코드
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128,
                                           fmax=8000)
    print(S.shape)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


if __name__ == '__main__':

    wav_path = 'data/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm' ### 수정하세요
    # wav_path = "data/Everything's Alright- Laura Shigihara- lyrics [nP-AAlZlCkM].m4a"

    visualize_wav_to_spectrogram(wav_path, n_mels=1)

