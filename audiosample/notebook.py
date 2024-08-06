import io
from pathlib import Path
import base64
from audiosample import AudioSample
from uuid import uuid4

import numpy as np

try:
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

def _mel_html(self, axis=False, player_id="", visible=True):
    # Generate the mel spectrogram
    S = librosa.feature.melspectrogram(y=self.to_mono().as_numpy(), sr=self.sample_rate, n_mels=128)

    # Convert to log scale (dB). Use the peak power as reference.
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Create a plot
    fig = plt.figure(figsize=(10, 4))
    fig.tight_layout()
    if not axis:
        ax = plt.gca()
        ax.set_axis_off()
        librosa.display.specshow(S_dB, sr=self.sample_rate, cmap='viridis', ax=ax)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        #display time axis and log frequencies:
        librosa.display.specshow(S_dB, sr=self.sample_rate, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')

    # Save the plot to a buffer
    buffered = io.BytesIO()
    plt.savefig(buffered, format='jpeg')
    plt.close()

    return f"""<img data-id="{player_id}" style="{"visibility: hidden; position: absolute;" if not visible else ""}" src="data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}" />"""


def _wave_html(self, axis=False, player_id="", visible=True):
    fig = plt.figure(figsize=(10, 4))
    fig.tight_layout()
    if not axis:
        ax = plt.gca()
        ax.set_axis_off()
        librosa.display.waveplot(self.to_mono().as_numpy(), sr=self.sample_rate, ax=ax)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        librosa.display.waveplot(self.as_numpy(), sr=self.sample_rate)
    buffered = io.BytesIO()
    plt.savefig(buffered, format='jpeg')
    plt.close()

    return f"""<img data-id="{player_id}" style="{"visibility: hidden; position: absolute;" if not visible else ""}" src="data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}" />"""

def _mel_show(self):
    if not NOTEBOOK_DISPLAY_SUPPORTED:
        raise NotImplementedError("display() requires IPython.display and jupyter notebook")
    return ipd.display(ipd.HTML(_mel_html(self, axis=True)))

AudioSample.register_plugin('mel_show', _mel_show)

def _wave_show(self):
    if not NOTEBOOK_DISPLAY_SUPPORTED:
        raise NotImplementedError("display() requires IPython.display and jupyter notebook")
    return ipd.display(ipd.HTML(_wave_html(self, axis=True)))

AudioSample.register_plugin('_wave_show', _wave_show)


def _player_html(self):
    should_init = getattr(AudioSample, 'INITIALIZE_PLAYER_SCRIPT', True)
    html = ""
    if True: #should_init:
        AudioSample.INITIALIZE_PLAYER_SCRIPT = False
        html += """
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
                left: 50%;
                background-color: #555;
                }
                </style>
                """
        html += "<script>"+ open(f"{Path(__file__).parent}/notebook_js/spectrogram-player.js").read() + "</script>"
    return html

def _player_html_init(player_id):
    return f"""<script>if (document.readyState == 'complete') {{ 
            window.spectrogram_player.init("{player_id}") 
        }} else {{
            document.addEventListener('load', function() {{
                window.spectrogram_player.init("{player_id}")
                }})
        }}
        </script>"""

def mel_display(self, player=True):
    """
    Display the audio sample as a mel spectrogram in a jupyter notebook.
    The mel spectrogram is displayed as an image and is clickable to seek to a specific time.
    Parameters
    ----------
    player : bool
        If True, display the audio player with the mel spectrogram. 
        If False, display only the mel spectrogram without the player.
    """
    if not player:
        _mel_show(self)
        return
    data = self.as_wav_data()
    player_id = str(uuid4())

    html = _player_html(self)
    html += f"""<div class="spectrogram-player" data-id="{player_id}" data-width="600" data-height="200" data-freq-min="0" data-freq-max="20">"""
    html += _mel_html(self, player_id=player_id, visible=False)
    html += f"""<audio data-id="{player_id}" controls src="data:audio/wav;base64,{base64.b64encode(data).decode()}" />"""
    html += """</div>"""
    html += _player_html_init(player_id)

    return ipd.display(ipd.HTML(html))


AudioSample.register_plugin('mel_display', mel_display)

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
    if not player:
        _wave_show(self)
        return
    data = self.as_wav_data()
    player_id = str(uuid4())

    html = _player_html(self)
    html += f"""<div class="spectrogram-player" data-id="{player_id}" data-width="600" data-height="200" data-axis-width="0" data-freq-min="0" data-freq-max="20">"""
    html += _wave_html(self, player_id=player_id, visible=False)
    html += f"""<audio data-id="{player_id}" controls src="data:audio/wav;base64,{base64.b64encode(data).decode()}" />"""
    html += """</div>"""
    html += _player_html_init(player_id)
    
    return ipd.display(ipd.HTML(html))


AudioSample.register_plugin('wave_display', wave_display)

