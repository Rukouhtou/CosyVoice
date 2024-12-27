# 코드 내에 파이썬패스 추가.
## (repo에 제시된 리눅스용 export PYTHONPATH=third_party/Matcha-TTS).
import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torchaudio
import whisper


# 모델 로드.
cv1 = CosyVoice('pretrained_models/CosyVoice-300M', load_jit=True, load_onnx=False, fp16=True)
cv2 = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)

# 합성할 텍스트와 프롬프트 텍스트.
tts_text = '존경하는 국민 여러분, 저는 대통령으로서 피를 토하는 심정으로 국민 여러분께 호소드립니다. 저는 북한 공산 세력의 위협으로부터 자유대한민국을 수호하고 우리 국민의 자유와 행복을 약탈하고 있는 파렴치한 종북 반국가 세력들을 일거에 척결하고 자유 헌정질서를 지키기 위해 비상계엄을 선포합니다.'

prompt_text1 = '실패하면 반역, 성공하면 혁명 아닙니까!'
prompt_text2 = '그니까 쥬얼슬롯을 얻을려면은? 어, 나 하이 쥬얼러 오브 없어. 발라야지만 열린다는거지? 아니 나 이거 말하는거 아닌데? 안 돼가지고. 패시브스킬 하나를 얻을 수가 있는데, 클릭이 안 돼. 할당이 안된다는 거지.'

# 프롬프트 음성과 소스 음성.
## speech1은 영화 서울의봄의 대사 일부이며 끝에 노이즈가 있습니다.
## speech2는 딕션이 좋지않은 일반인 지인의 마이크 녹음 음성입니다.
## source는 대통령의 비상계엄 대국민 담화문의 일부분입니다.
prompt_speech1_16k = load_wav('custom_preprocessed/서울의봄-실패반역성공혁명대사.mp3', 16000)
prompt_speech2_16k = load_wav('custom_preprocessed/mmrk_poe_shorter.wav', 16000)

source_speech_16k = load_wav('custom_preprocessed/ysy_martial_law_shorter.wav', 16000)


# <<1. 한국어 to 다국어 텍스트 번역 (다국어 번역모델 NLLB를 사용했습니다)>>
## 카자흐스탄, 베트남, 캄보디아, 탄자니아어의 번역.
## 디폴트로 cache_dir = "~/.cache/huggingface/hub" 인걸 cosyvoice의 모델경로로 변경했습니다.
tl_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='pretrained_models')
tl_tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='pretrained_models')
translator = pipeline("translation", model=tl_model, tokenizer=tl_tokenizer)

## English
en_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="eng_Latn")[0]['translation_text']
en_text = f'<|en|>{en_text}'

## Kazakh
# kk_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="kaz_Cyrl")[0]['translation_text']

## Vietnamese
# vi_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="vie_Latn")[0]['translation_text']
# vi_text = f'<|vi|>{vi_text}'

## Khmer (Cambodian)
# km_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="khm_Khmr")[0]['translation_text']

## Swahili (Tanzania)
# sw_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="swh_Latn")[0]['translation_text']


# <<2. 음성합성 및 저장>>
## voice conversion, zeroshot, cross lingual기능을 각각 두 개의 프롬프트 음성으로 테스트 해봤습니다.
## (1) cosyvoice1의 voice conversion.
for i, j in enumerate(cv1.inference_vc(source_speech_16k=source_speech_16k, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_voiceconversion1_{i}.wav', j['tts_speech'], cv1.sample_rate)

for i, j in enumerate(cv1.inference_vc(source_speech_16k=source_speech_16k, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/martial_law_voiceconversion2_{i}.wav', j['tts_speech'], cv1.sample_rate)

## (2) cosyvoice2.
##      (2는 vc기능이 아직없지만 음성합성 퀄리티가 더 높습니다.)
for i, j in enumerate(cv2.inference_zero_shot(tts_text=tts_text, prompt_text=prompt_text1, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_zeroshot1_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_zero_shot(tts_text=tts_text, prompt_text=prompt_text2, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/martial_law_zeroshot2_{i}.wav', j['tts_speech'], cv2.sample_rate)

## (3) 2를 이용하여 영문 텍스트를 음성으로 합성.
##      (zero_shot은 프롬프트를 잘 반영하고, cross_lingual은 억양을 살려줍니다.)
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=prompt_text1, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_zeroshot1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_cross_lingual1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

## (4) 소스음성(한국어) -> 텍스트로 transcribe -> 영어로 번역 -> 음성(영어)으로 합성해 보겠습니다.
##      whisper의 오디오 사용을 위해 터미널에서 ffmpeg를 설치합니다. (conda install ffmpeg -c conda-forge)
audio_path = 'custom_preprocessed/ysy_martial_law_shorter.wav'
whisper_model = whisper.load_model('base')
transcribed_text = whisper_model.transcribe(audio_path, language='ko')['text']

en_text = translator(transcribed_text, src_lang="kor_Hang", tgt_lang="eng_Latn")[0]['translation_text']
en_text = f'<|en|>{en_text}'

## 프롬프트 음성1:
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=prompt_text1, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

## 프롬프트 음성2:
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=prompt_text2, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot2_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual2_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

## 소스음성을 프롬프트로 사용:
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=transcribed_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot3_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual3_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

## 한국어 음성이 입력으로 주어질 때, 영어로 번역된 음성이 출력됩니다.


# <<3. 목표인 다국어로 음성번역>>
## 우선 베트남어로 테스트 해봤습니다.
## 새 언어를 위해선 문장앞에 language token을 넣어줄 필요가있어 베트남어 토큰인 <|vi|>를 붙였습니다.
vi_text = translator(transcribed_text, src_lang="kor_Hang", tgt_lang="vie_Latn")[0]['translation_text']
vi_text = f'<|vi|>{vi_text}'

for i, j in enumerate(cv2.inference_zero_shot(tts_text=vi_text, prompt_text=transcribed_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot3_vi_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=vi_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual3_vi_{i}.wav', j['tts_speech'], cv2.sample_rate)

# => 결과로 저장된 베트남어 음성(final_)은 의미없는 말만 나옵니다.
# tokenizer.py에 베트남어가 있긴하지만, 결국 모델을 파인튜닝 해야하는것 같습니다.
# https://github.com/FunAudioLLM/CosyVoice/issues/466,
# https://github.com/FunAudioLLM/CosyVoice/issues/344 를 참고해보니
# 새로운 언어의 cover를 위해선 적어도 5천시간 이상의 음성이 필요하다고 하지만,
# 저는 준비된 베트남어 음성이 없어서 구글 번역기의 음성으로 1~10초 정도의 샘플 16개를 만들어 사용했습니다.

# repo에 훈련을 위한 recipe로 올려둔 run.sh를 바탕으로, 전처리를 위한 샘플 구성을 하여
# 훈련용과 테스트용의 데이터 폴더를 각각 두 개씩 만들었습니다.
# 이 후 run.sh를 제 환경에 맞게 수정하고(examples/libritts/cosyvoie/run.sh 참고해주세요!)
# 저는 윈도우 환경이기에 처음으로 리눅스환경을 만들어 source run.sh하여 돌려보았습니다.
# (제 환경은 윈도우 + WSL2 + Ubuntu 24.04 에 rtx 2080 super 한 장입니다)

# 처음엔 train stage에서 오류가 났지만, torchaudio를 requirements와는 다른 2.0.1로 다운그레이드해주니 해결되었습니다.
# (https://github.com/FunAudioLLM/CosyVoice/issues/763 참고)

# 이후 아주 짧은 훈련시간(약 25분)후 llm, hifigan, flow 3종류의 모델(총 약 50GB)이 저장됐으며,
# (tensorboard폴더에 train 로그 확인해 주세요!)
# 훈련후 생성된 체크포인트들에서 가장 잘 나온 모델(best val)중 하나를 불러와 CosyVoice-300M으로 테스트 해봤습니다.
# (96번째 에포크 모델 경로: examples/libritts/cosyvoice/exp/cosyvoice/hifigan/torch_ddp/epoch_96_whole.pt)
import torch
from hyperpyyaml import load_hyperpyyaml


config_file = "./pretrained_models/CosyVoice-300M/cosyvoice.yaml"
override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}

with open(config_file, 'r') as f:
    configs = load_hyperpyyaml(f, overrides=override_dict)

model = configs["llm"]
model.load_state_dict(torch.load("./pretrained_models/CosyVoice-300M/epoch_96_whole.pt", map_location='cpu')["module"])
cv1.model.llm = model.to("cuda").half()


# 하지만 체크포인트가 불러와지지 않았습니다.(https://github.com/FunAudioLLM/CosyVoice/issues/490)
# model.py의 llm.load_state_dict의 strict=False로 하니 추론은 가능한걸보면, 저장된 가중치와 모델구조가 일치하지 않는것 같습니다.
for i, j in enumerate(cv1.inference_zero_shot(tts_text=vi_text, prompt_text=transcribed_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/testfinal_zeroshot3_vi_{i}.wav', j['tts_speech'], cv1.sample_rate)

for i, j in enumerate(cv1.inference_cross_lingual(tts_text=vi_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/testfinal_cross_lingual3_vi_{i}.wav', j['tts_speech'], cv1.sample_rate)

# transcribe로 과연 베트남 음성으로 인식되나 확인.
trained_audio_path1 = 'output/testfinal_zeroshot3_vi_0.wav'
trained_text1 = whisper_model.transcribe(trained_audio_path1, language='vi')['text']
trained_audio_path2 = 'output/testfinal_cross_lingual3_vi_0.wav'
trained_text2 = whisper_model.transcribe(trained_audio_path2, language='vi')['text']
print(trained_text1)
print(trained_text2)



# 오픈소스로 이런 간단한 프로젝트를 만들어보는건 처음이고, 부족함을 많이 느꼈습니다.
# 머신러닝 초보라서, 공부 중에 케글이나 깃헙에 올라오는 개발자들의 실력을 보면 매번 대단함을 느낍니다.
# 하지만 만드는 과정이 재미있고, 조금만 더 하면 할 수 있을 것 같다는 열정도 생깁니다.
# 저도 언젠가 이런 신기술들을 실생활에 적용하는 데에 기여할 수 있다면 좋겠다는 생각을 합니다.
# 언젠가 올 강 인공지능 시대에 뒤쳐지지 않기 위해 저의 머신러닝 역량을 키우고 싶습니다.