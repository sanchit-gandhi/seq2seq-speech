#!/usr/bin/env python3
import whisper

model = whisper.load_model("base.en")
result = model.transcribe("audio.mp3")
print(result["text"])
