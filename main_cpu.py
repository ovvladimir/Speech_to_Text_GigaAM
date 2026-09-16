# virtualenv ENV
# ENV\Scripts\activate
# pip install -r requirements.txt

# БЕЗ CUDA, ТОЛЬКО CPU

''' - ОСНОВНОЙ КОД - '''

import sys
import subprocess
import numpy as np
import torch
import pyautogui
import pyperclip  
import time
import keyboard
from transformers import AutoModel, AutoProcessor
# Отключаем информационные логи (предупреждения)
import warnings
import logging
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_LEVEL"] = "-1"
# Отключаем предупреждения Python и transformers
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- НАСТРОЙКА ПУТЕЙ ДЛЯ PYINSTALLER ---
if getattr(sys, 'frozen', False):
    # Если запущен скомпилированный .exe
    EXE_DIR = os.path.dirname(sys.executable)   # Папка снаружи, где лежит .exe
    INTERNAL_DIR = sys._MEIPASS                 # Папка изнутри .exe (временная)
else:
    # Если запуск как обычный .py скрипт
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    INTERNAL_DIR = EXE_DIR

# Прямые пути к локальным файлам БЕЗ изменения PATH
FFMPEG_BINARY = os.path.normpath(os.path.join(INTERNAL_DIR, "ffmpeg", "ffmpeg.exe"))

# МИКРОФОН
# ищем командой - ffmpeg -list_devices true -f dshow -i dummy
# device_name = '"audio=Микрофон (2- USB PnP Sound Device)"'
# или
dev = subprocess.run(
    f'{FFMPEG_BINARY} -hide_banner -list_devices true -f dshow -i dummy',
    stderr=subprocess.PIPE, encoding='utf-8').stderr
index = dev[:dev.find('(audio)')]
device_name = index[index.rfind(' "'):].strip(' " ')
device_name = f'"audio={device_name}"'
print('-'*55)
print(f'Микрофон: {device_name}')
print('-'*55)

# Модель
model_id = "waveletdeboshir/gigaam-ctc"
LOCAL_MODEL_DIR = os.path.normpath(os.path.join(EXE_DIR, "model"))
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
device = "cpu"

# --- Конфигурация диктовки ---
TARGET_SAMPLE_RATE = 16000  # Родная частота GigaAM
VAD_THRESHOLD = 0.02        # Чувствительность микрофона (0.04 если шумно)
SILENCE_DURATION = 1.0      # Пауза тишины в секундах перед распознаванием
CHUNK_DURATION = 0.1        # Шаг обработки (100 мс)

# Количество байт в одном чанке: 100мс * 16000Гц * 4 байта (float32)
CHUNK_SIZE = int(CHUNK_DURATION * TARGET_SAMPLE_RATE * 4)

# Печать текста
def type_text_via_clipboard(text: str):
    if not text or not text.strip():
        return
    if isinstance(text, list):
        text = " ".join(text)
    
    text = text.strip() + " " 
    
    # Просто копируем и сразу вставляем, ничего не возвращая назад
    pyperclip.copy(text)
    time.sleep(0.1)  
    
    if sys.platform == 'darwin':  
        pyautogui.hotkey('command', 'v')
    else:
        keyboard.press_and_release('ctrl+v')
        
    time.sleep(0.1)

def draw_volume_bar(rms, threshold, is_speaking):
    bar_length = 30
    filled_length = int(min(rms * 100, bar_length))
    threshold_pos = int(min(threshold * 100, bar_length))
    
    bar = ""
    for i in range(bar_length):
        if i == threshold_pos:
            bar += "|"  
        elif i < filled_length:
            bar += "█" if is_speaking else "░"
        else:
            bar += " "
            
    status = "[Голос]" if is_speaking else "[Тишина]"
    sys.stdout.write(f"\rГромкость: [{bar}] {rms:.4f} {status}   ")
    sys.stdout.flush()

def main():
    # Загрузка нейросети GigaAM

    try:
        # 3. Загрузка процессора и модели (может занять время при первом запуске)
        # Передаем оригинальное имя репозитория, но жестко фиксируем cache_dir.
        # Так библиотека создаст правильную структуру кэша внутри вашей папки проекта.
        print(f"\n[1/2] Загрузка GigaAM на {device.upper()}... Пожалуйста, подождите.")
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            cache_dir=LOCAL_MODEL_DIR,
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            cache_dir=LOCAL_MODEL_DIR,
            local_files_only=True
        ).to(device)
    except Exception as e:
        print(f"\n❌ Ошибка загрузки модели из Hugging Face: {e}")
        return
    
    model.eval()

    # Команда для FFmpeg: захват звука по умолчанию в Windows (DirectSound) и вывод сырых float32 данных с частотой 16кГц в моно
    ffmpeg_cmd = f'{FFMPEG_BINARY} -f dshow -i {device_name} -ac 1 -ar {TARGET_SAMPLE_RATE} -f f32le -loglevel quiet -'

    try:
        # Запускаем фоновый процесс FFmpeg (для проверки ошибок - stderr=sys.stderr)
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
    except FileNotFoundError:
        print("\n❌ Ошибка: Утилита FFmpeg не найдена в вашей системе!")
        print("Пожалуйста, установите FFmpeg и добавьте её в PATH.")
        return

    print("👉 Переключитесь на Блокнот, Word или мессенджер и начните говорить...\n")

    recording_buffer = []
    is_speaking = False
    
    silence_chunks_limit = int(SILENCE_DURATION / CHUNK_DURATION)
    silence_counter = 0

    try:
        while True:
            # Читаем ровно 100 мс аудиосигнала из stdout процесса FFmpeg
            raw_data = process.stdout.read(CHUNK_SIZE)
            if not raw_data:
                print("\n⚠️ Поток FFmpeg неожиданно прервался.")
                break
                
            # Превращаем байты в массив float32
            chunk_mono = np.frombuffer(raw_data, dtype=np.float32)
            
            if len(chunk_mono) == 0:
                continue

            # Вычисляем громкость (RMS)
            rms = np.sqrt(np.mean(chunk_mono**2))

            if rms > VAD_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                recording_buffer.append(chunk_mono)
                silence_counter = 0  
            else:
                if is_speaking:
                    recording_buffer.append(chunk_mono)
                    silence_counter += 1
                    
                    if silence_counter > silence_chunks_limit:
                        sys.stdout.write("\n -> [Обработка нейросетью GigaAM...] \n")
                        sys.stdout.flush()
                        
                        # Соединяем чанки (звук уже в моно и 16000 Гц, librosa больше не нужна!)
                        full_audio_16k = np.concatenate(recording_buffer)
                        
                        # Инференс нейросети
                        inputs = processor(full_audio_16k, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            logits = model(**inputs).logits
                        
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids)
                        
                        text_result = transcription if isinstance(transcription, list) else transcription
                        if isinstance(text_result, list):
                            text_result = " ".join(text_result)
                        
                        if text_result.strip():
                            print(f"💬 Распознано: {text_result}")
                            type_text_via_clipboard(text_result)
                        else:
                            print("💬 [Голос не распознан]")
                        
                        recording_buffer = []
                        is_speaking = False
                        silence_counter = 0
                        
            draw_volume_bar(rms, VAD_THRESHOLD, is_speaking)
            
    except KeyboardInterrupt:
        print("\nПрограмма диктовки успешно остановлена.")
    finally:
        # Корректно закрываем фоновый процесс при выходе
        process.terminate()

if __name__ == "__main__":
    main()

# Сборка через PyInstaller
# pyinstaller --onefile --clean --name="GigaAM_Dictation" --hidden-import="torchaudio" --hidden-import="torchaudio.backends.soundfile_backend" --add-data "ffmpeg;ffmpeg" main_cpu.py
# pyinstaller --onefile --clean --name="GigaAM_Dictation" --hidden-import="torchaudio" --hidden-import="torchaudio.backends.soundfile_backend" --add-data "ffmpeg;ffmpeg" --upx-dir="." --upx-exclude="torch_python.dll" --upx-exclude="_torch.pyd" main_cpu.py
# Папка с моделью GigaAM (остается снаружи)
# Обратить внимание на синтаксис двоеточия ; — на Windows это разделитель между тем, какую папку взять и куда её положить внутри архива.Что важно знать после сборки:Размер файла: Итоговый .exe файл в папке dist будет весить около 3–5 Гигабайт, так как внутри него будут находиться тяжелые библиотеки PyTorch и сами веса нейросети.Время запуска: При каждом открытии этого .exe Windows будет тратить от 5 до 15 секунд на то, чтобы распаковать эти гигабайты во временную папку Temp и запустить процесс. Это нормальное поведение для огромных AI-моделей, упакованных в один файл.Права администратора: Обязательно запускайте готовый .exe правой кнопкой мыши -> От имени Администратора, иначе библиотека keyboard не сможет отправлять текст в окна других программ.Сборка займет приличное время (около 5–10 минут).

'''
VAD_THRESHOLD (Порог тишины): Если скрипт постоянно пишет [Запись голоса...], 
даже когда вы молчите, увеличьте этот параметр (например, до 0.04 или 0.06), 
чтобы отсечь фоновый шум комнаты. 
[1] (https://habr.com/ru/articles/1002260/)
SILENCE_DURATION: Сейчас выставлена 1.0 секунда. 
Если вы делаете длинные паузы между словами в пределах одного предложения, 
увеличьте это значение (например, до 1.5), чтобы скрипт не разрывал вашу речь на отдельные фразы.
'''