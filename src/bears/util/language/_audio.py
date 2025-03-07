import numpy as np

# Define a hidden audio class to suppress the audio player output


def beep(
    freq1: int = 300,
    freq2: int = 400,
    duration: int = 0.25,  # Duration in seconds
    reps: int = 3,
):
    from IPython.display import Audio, display

    class HiddenAudio(Audio):
        def _repr_html_(self):
            html = super()._repr_html_()
            return f'<div style="display:none;">{html}</div>'

    # Sound parameters
    fs = 44100  # Sampling rate in Hz
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Create an exponential decay envelope
    envelope = np.exp(-4 * t)

    # Define two frequencies for a pleasant chord (in Hz)

    # Generate the sound: combine two sine waves and apply the envelope
    single_beep = ((np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)) * envelope).tolist()
    sound = []
    for i in range(reps):
        sound.extend(single_beep)
    sound = np.array(sound)
    # Play the sound without displaying the audio controls
    display(HiddenAudio(sound, rate=fs, autoplay=True))
