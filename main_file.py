import os
import torch
import torchaudio
import soundfile as sf
from transformers import AutoModel, AutoProcessor

# Отключаем предупреждения Python и transformers
import warnings
import logging
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_LEVEL"] = "-1"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Настройки путей
AUDIO_PATH = "audio.wav" 
LOCAL_MODEL_DIR = "./model"

# Создаем папку для модели, если её нет
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# 2. Загружаем аудио через soundfile и подготавливаем для модели
data, sr = sf.read(AUDIO_PATH)
wav = torch.FloatTensor(data)

# Стерео в моно
if wav.ndim > 1:
    wav = wav.mean(dim=-1)

# Ресемплинг в 16 кГц
if sr != 16000:
    wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
else:
    wav = wav.squeeze()

# Выбираем устройство (GPU или CPU)
device = "cpu"
print(f"Используется устройство: {device.upper()}")

# 3. Загрузка процессора и модели
# Передаем оригинальное имя репозитория, но жестко фиксируем cache_dir.
# Так библиотека создаст правильную структуру кэша внутри вашей папки проекта.
print("Загрузка модели GigaAM (может занять время при первом запуске)...")
processor = AutoProcessor.from_pretrained(
    "waveletdeboshir/gigaam-ctc", 
    trust_remote_code=True, 
    cache_dir=LOCAL_MODEL_DIR
)
model = AutoModel.from_pretrained(
    "waveletdeboshir/gigaam-ctc", 
    trust_remote_code=True, 
    cache_dir=LOCAL_MODEL_DIR
).to(device)

model.eval()

# 4. Превращаем аудио в признаки
input_features = processor(wav, sampling_rate=16000, return_tensors="pt")
input_features = {k: v.to(device) for k, v in input_features.items()}

# 5. Распознавание речи
print("Распознавание аудио...")
with torch.no_grad():
    logits = model(**input_features).logits

# Декодирование результатов
greedy_ids = logits.argmax(dim=-1)
transcription = processor.batch_decode(greedy_ids)

print("\n--- Результат распознавания ---")
print(transcription[0] if isinstance(transcription, list) else transcription)
