"""
Turn a spoken question into text, for the microphone buttons of the app.

The recording goes to OpenAI's speech-to-text model, which works out by itself
whether the speaker used Arabic or English. The text then goes through answer()
(or the company search) like typed text, so the answers and their checks do not change.

No word list is sent with the recording: with a short or quiet recording the model
wrote the list back instead of what was said.

    text = transcribe(wav_bytes)    -> "ما هي حقوق صاحب البيانات؟"
"""
import io

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-4o-transcribe"


def transcribe(audio_bytes, filename="question.wav"):
    """The words in the recording, or "" if nothing was said. Raises RuntimeError
    if OpenAI cannot be reached."""
    load_dotenv()
    audio = io.BytesIO(audio_bytes)
    audio.name = filename  # OpenAI reads the format from the file name
    try:
        result = OpenAI().audio.transcriptions.create(model=MODEL, file=audio)
    except Exception as error:
        raise RuntimeError(f"Speech to text failed: {error}") from error
    return arabic_marks(result.text.strip())


def arabic_marks(text):
    """The model writes Arabic with English punctuation ("?" for "؟"): use the Arabic marks
    when most letters are Arabic."""
    from src.policies.language import text_language
    if text_language(text) != "ar":
        return text
    return text.translate(str.maketrans({"?": "؟", ",": "،", ";": "؛"}))
