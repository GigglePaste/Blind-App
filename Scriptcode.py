import cv2
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import pyttsx3
import time
import pytesseract
import numpy as np
import spacy
import language_tool_python
from spellchecker import SpellChecker

engine = pyttsx3.init()
video = cv2.VideoCapture(0)
engine.setProperty('rate', 122)

def scene():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    engine.startLoop(False)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Failed to capture frame, ending.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(3)
        inputs = processor(frame, return_tensors="pt")

        generated_ids = model.generate(**inputs, max_length=100, min_length=25, do_sample=True,
                                       repetition_penalty=2.0, temperature=0.2, top_k=0, top_p=0.9)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        print(generated_text)

        engine.say("Scene Description " + generated_text)
        engine.iterate()
        
    video.release()
    engine.stop()

def ocr():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    engine.startLoop(False)
    tool = language_tool_python.LanguageTool('en-US')
    nlp = spacy.load("en_core_web_sm")
    
    def preprocess_frame_for_ocr(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        return frame

    def word_correction(text):
        spell = SpellChecker()
        corrected_text = " ".join([spell.correction(word) if spell.correction(word) is not None else word for word in text.split()])
        return corrected_text

    def sentence_reconstruction(ocr_text):
        def correct_grammar(text):
            matches = tool.check(text)
            corrected_text = language_tool_python.utils.correct(text, matches)
            return corrected_text

        def reconstruct_text(text):
            doc = nlp(text)
            reconstructed_sentences = []
            for sent in doc.sents:
                tokens = [token.text for token in sent]
                reconstructed_sentence = " ".join(tokens)
                reconstructed_sentence = correct_grammar(reconstructed_sentence)
                reconstructed_sentences.append(reconstructed_sentence)
            return " ".join(reconstructed_sentences)

        corrected_text = correct_grammar(ocr_text)
        final_text = reconstruct_text(corrected_text)
        return final_text

    i = 0 
    while video.isOpened() and i < 50:
        ret, frame = video.read()
        if not ret:
            print("Frame was not captured, ending...!")
            break
        
        frame = preprocess_frame_for_ocr(frame)
        ocr_text = pytesseract.image_to_string(frame)
        corrected_words = word_correction(ocr_text)
        corrected_sentence = sentence_reconstruction(corrected_words)
        print(corrected_sentence)

        engine.say(corrected_sentence)
        engine.iterate()
        
        i += 1
        
    video.release()
    engine.endLoop()

print("What do you want to do?")
choice = int(input("Enter 1 for Scene Description, 2 for OCR Processing: "))

if choice == 1:
    scene()
elif choice == 2:
    ocr()
else:
    print("Invalid choice.")
