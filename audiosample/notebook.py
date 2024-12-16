import io
from pathlib import Path
import base64
from audiosample import AudioSample
from uuid import uuid4
import html as HTML
import numpy as np

try:
    import ipywidgets as widgets
    import IPython.display as ipd
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import io
    NOTEBOOK_DISPLAY_SUPPORTED = True
except ImportError:
    #warnings.warn("AudioSample unable to support audio display, please install jupyter notebook")
    NOTEBOOK_DISPLAY_SUPPORTED = False


def display(self, autoplay=False):
    """
    Display the audio sample in a jupyter notebook.
    """
    if not NOTEBOOK_DISPLAY_SUPPORTED:
        raise NotImplementedError("display() requires IPython.display and jupyter notebook")
    return ipd.Audio(self.as_wav_data(), autoplay=autoplay)

AudioSample.register_plugin('display', display)

def _mel_img(self, axis=False):
    # Generate the mel spectrogram
    S = librosa.feature.melspectrogram(y=self.to_mono().as_numpy(), sr=self.sample_rate, n_mels=128)

    # Convert to log scale (dB). Use the peak power as reference.
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Create a plot
    width = min(max(10, self.duration), 200)
    fig = plt.figure(figsize=(width, 4))
    fig.tight_layout()
    if not axis:
        plt.axis('off')
        plt.margins(0)
        ax = plt.gca()
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        librosa.display.specshow(S_dB, sr=self.sample_rate, cmap='viridis', ax=ax)
    else:
        #display time axis and log frequencies:
        librosa.display.specshow(S_dB, sr=self.sample_rate, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')

    # Save the plot to a buffer
    buffered = io.BytesIO()
    plt.savefig(buffered, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()

    return buffered.getvalue()


def _wave_img(self, axis=False):
    width = min(max(10, self.duration), 200)
    fig = plt.figure(figsize=(width, 4))
    fig.tight_layout()
    if not axis:
        plt.axis('off')
        plt.margins(0)
        ax = plt.gca()
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        librosa.display.waveshow(self.to_mono().as_numpy(), sr=self.sample_rate, ax=ax)
    else:
        librosa.display.waveshow(self.as_numpy(), sr=self.sample_rate)
    buffered = io.BytesIO()
    plt.savefig(buffered, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()

    return buffered.getvalue()

def _player_html(self, image_buffer: io.BytesIO = b"", player: bool = True, axis=True):
    player_id = str(uuid4())

    html = "<html>"
    html += """
            <head>
            <style>
            .sp-viewer {
                position: relative;
                background-repeat: no-repeat;
            }

            .sp-axis {
                position: absolute;
            }

            .sp-timeBar {
                width: 3px;
                height: 100%;
                position: absolute;
                left: 25%;
                background-color: #555;
            }
            .sp-image { 
                visibility: hidden;
                position: float;
                height: 0;
                width: 0;
            }
            .sp-vis-image {
                height: 200px;
                width: 100%;
            }
            .sp-audio {
                width: 100%;
            }
            </style></head>
            """
    html += "<body>"
    html += f"""<div class="spectrogram-player" data-id="{player_id}" data-height="200" data-freq-min="0" data-freq-max="{self.sample_rate}" {"data-axis-width=0" if not axis else ""}>"""
    visible = not player
    html += f"""<img class="{"sp-vis-image" if visible else "sp-image"}" data-id="{player_id}" src="data:image/jpeg;base64,{base64.b64encode(image_buffer).decode()}" />"""

    data = self.as_wav_data()
    if player:
        html += f"""<audio class="sp-audio" data-id="{player_id}" controls src="data:audio/wav;base64,{base64.b64encode(data).decode()}" />"""
    html += """</div>"""
    if player:
        html += "<script>"+ open(f"{Path(__file__).parent}/notebook_js/spectrogram-player.js").read() + "</script>"
        html += f"""<script>
                    window.addEventListener('load', function() {{ window.spectrogram_player.init("{player_id}") }})
            </script>"""
    html += "</body>"
    html += "</html>"

    iframe = f"""<div width="100%"><iframe style='width: 100%; height: 275px; border: none;' srcdoc="{HTML.escape(html)}" /></div>"""
    return iframe


def mel(self, player: bool = True):
    """
    Display the audio sample as a mel spectrogram in a jupyter notebook.
    The mel spectrogram is displayed as an image and is clickable to seek to a specific time.
    Parameters
    ----------
    player : bool
        If True, display the audio player with the mel spectrogram. 
        If False, display only the mel spectrogram without the player.
    """    
    if not NOTEBOOK_DISPLAY_SUPPORTED:
        raise NotImplementedError("display() requires IPython.display and jupyter notebook")

    html = _player_html(self, _mel_img(self, axis=False), player, axis=True)

    return widgets.HTML(html)

AudioSample.register_plugin('mel', mel)

def mel_display(self, player: bool = True):
    """
    Display the audio sample as a mel spectrogram in a jupyter notebook.
    The mel spectrogram is displayed as an image and is clickable to seek to a specific time.
    Parameters
    ----------
    player : bool
        If True, display the audio player with the mel spectrogram. 
        If False, display only the mel spectrogram without the player.
    """    
    return ipd.display(mel(self, player))

AudioSample.register_plugin('mel_display', mel_display)

def wave(self, player=True):
    """
    Display the audio sample as a waveform in a jupyter notebook.
    The waveform is displayed as an image and is clickable to seek to a specific time.
    Parameters
    ----------
    player : bool
        If True, display the audio player with the waveform. 
        If False, display only the waveform without the player.
    """
    if not NOTEBOOK_DISPLAY_SUPPORTED:
        raise NotImplementedError("display() requires IPython.display and jupyter notebook")

    html = _player_html(self, _wave_img(self, axis=False), player, axis=False)

    return widgets.HTML(html)


AudioSample.register_plugin('wave', wave)


def wave_display(self, player=True):
    """
    Display the audio sample as a waveform in a jupyter notebook.
    The waveform is displayed as an image and is clickable to seek to a specific time.
    Parameters
    ----------
    player : bool
        If True, display the audio player with the waveform. 
        If False, display only the waveform without the player.
    """
    return ipd.display(wave(self, player))
AudioSample.register_plugin('wave_display', wave_display)


def _mel_show(self):
    return mel_display(self, player=False)

AudioSample.register_plugin('mel_show', _mel_show)

def _wave_show(self):
    return wave_display(self, player=False)

AudioSample.register_plugin('_wave_show', _wave_show)

