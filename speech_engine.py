# import whisper
import sounddevice as sd
import numpy as np
# import torch
# import queue
import pyttsx3
import re
import speech_recognition as sr
from queue import Queue
from datetime import datetime, timedelta
# import time
import threading

class SpeechEngine:
    def __init__(self):
        # Initialize TTS engine
        self.engine = pyttsx3.init()

        self.recorder = sr.Recognizer()
        # self.recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        self.mic = sr.Microphone()


    """Convert text to speech using pyttsx3"""
    def speak(self, text, blocking=True):
        if blocking:
            self.engine.say(text)
            self.engine.save_to_file(text, "system-voice/user_confirm_tts.wav")
            self.engine.runAndWait()
        else:
            # Run the TTS operation in a separate thread TODO: doesn't work well
            threading.Thread(target=self._speak_non_blocking, args=(text,)).start()

    def _speak_non_blocking(self, text):
        """Helper method to handle non-blocking TTS"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_audio(self, print_status=True):
        # r = sr.Recognizer()
        if print_status: print("Listening...")
        with self.mic as source:
            audio = self.recorder.listen(source)
            said = ""

            try:
                said = self.recorder.recognize_google(audio)
                print(said)
            except Exception as e:
                print("Exception: " + str(e))

        return said.lower()
    
    """Real-time listening using speech recognition"""
    def live_listening(self):
        pattern = r"ems (.*)" # wake up word is "EMS"

        while True:
            user_prompt = self.get_audio() # get_audio()
            # print(f"live mic: {user_prompt}")

            match = re.search(pattern, user_prompt, re.DOTALL)
            # print(match)

            if "exit" in user_prompt:
                self.speak("Goodbye!")
                break
            elif match:
                matches = re.findall(pattern, user_prompt,  re.DOTALL)
                # print(matches)
                return matches[0]
            elif "hello" in user_prompt:
                self.speak("Hello! How can I assist you?")

    def mp3_to_text(self, audio_file_name):
        audio_file = sr.AudioFile(audio_file_name)

        with audio_file as source:
            audio = self.recorder.record(source)
        
        try:
            text = self.recorder.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        
        return text
    
if __name__ == "__main__":
    # # main()
    # print("Speech engine...")
    # listener = SpeechEngine()
    # # listener.listening()
    # print(f"prompt is: {listener.live_listening()}")
    text2speech = SpeechEngine()
    text2speech.speak("I'm going to move your right foot")
