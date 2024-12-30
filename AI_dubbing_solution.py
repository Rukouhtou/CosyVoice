import sys, os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from hyperpyyaml import load_hyperpyyaml
import torch
import torchaudio
import whisper


# 입력으로 사용할 오디오
audio_path = 'custom_preprocessed/서울의봄-실패반역성공혁명대사.mp3'

# 프롬프트로 사용할 오디오(입력된 오디오와 같은 목소리 사용)
source_speech_16k = load_wav(audio_path, 16000)

# 모델 로드
cv1 = CosyVoice('pretrained_models/CosyVoice-300M', load_jit=True, load_onnx=False, fp16=True)

tl_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='pretrained_models')
tl_tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='pretrained_models')
translator = pipeline("translation", model=tl_model, tokenizer=tl_tokenizer)

whisper_model = whisper.load_model('base')
transcribed_text1 = whisper_model.transcribe(audio_path, language='ko')['text']

vi_text1 = translator(transcribed_text1, src_lang="kor_Hang", tgt_lang="vie_Latn")[0]['translation_text']
vi_text1 = f'<|vi|>{vi_text1}'

config_file = "./pretrained_models/CosyVoice-300M/cosyvoice.yaml"
override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}

with open(config_file, 'r') as f:
    configs = load_hyperpyyaml(f, overrides=override_dict)

model = configs["llm"]
checkpoint = torch.load("./pretrained_models/CosyVoice-300M/best_val/llm/epoch_19_whole.pt", map_location='cpu')
state_dict = {k: v for k, v in checkpoint.items() if k not in ["epoch", "step"]}
model.load_state_dict(state_dict)
cv1.model.llm = model.to("cuda").half()

# 오디오 합성
os.makedirs('output', exist_ok=True)
for i, j in enumerate(cv1.inference_cross_lingual(tts_text=vi_text1, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/vi_dubbed_{i}.wav', j['tts_speech'], cv1.sample_rate)

# 합성한 오디오가 베트남어로 인식 되는지 확인
trained_audio_path = 'output/vi_dubbed_0.wav'
trained_text = whisper_model.transcribe(trained_audio_path, language='vi')['text']
print('transcribed_vi:', trained_text)
