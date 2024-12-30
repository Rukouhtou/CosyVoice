# 코드 내에 파이썬패스 추가.
#   (repo에 제시된 리눅스용 export PYTHONPATH=third_party/Matcha-TTS).
import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from hyperpyyaml import load_hyperpyyaml
import torch
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
#   speech1은 영화 서울의봄의 대사 일부이며 끝에 노이즈가 있습니다.
#   speech2는 딕션이 좋지않은 일반인 지인의 노이즈 없는 마이크 녹음 음성입니다.
#   source는 대통령의 비상계엄 대국민 담화문의 일부분입니다.
prompt_speech1_16k = load_wav('custom_preprocessed/서울의봄-실패반역성공혁명대사.mp3', 16000)
prompt_speech2_16k = load_wav('custom_preprocessed/mmrk_poe_shorter.wav', 16000)

source_speech_16k = load_wav('custom_preprocessed/ysy_martial_law_shorter.wav', 16000)


# <<1. 한국어 to 다국어 텍스트 번역 (다국어 번역모델 NLLB를 사용했습니다)>>
#   카자흐스탄, 베트남, 캄보디아, 탄자니아어를 텍스트 번역하며, 우선 영어로 테스트합니다.
#   디폴트로 cache_dir = "~/.cache/huggingface/hub" 인걸 cosyvoice의 모델경로로 변경해 보기쉽게 한 곳에 모아놨습니다.
tl_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='pretrained_models')
tl_tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='pretrained_models')
translator = pipeline("translation", model=tl_model, tokenizer=tl_tokenizer)

#   English
en_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="eng_Latn")[0]['translation_text']
en_text = f'<|en|>{en_text}'

#   Kazakh
# kk_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="kaz_Cyrl")[0]['translation_text']

#   Vietnamese
# vi_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="vie_Latn")[0]['translation_text']
# vi_text = f'<|vi|>{vi_text}'

#   Khmer (Cambodian)
# km_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="khm_Khmr")[0]['translation_text']

#   Swahili (Tanzania)
# sw_text = translator(tts_text, src_lang="kor_Hang", tgt_lang="swh_Latn")[0]['translation_text']


# <<2. 음성합성 및 저장>>
#   voice conversion, zeroshot, cross lingual기능을 각각 두 개의 프롬프트 음성으로 테스트 해봤습니다.
#   (1) cosyvoice1의 voice conversion.
for i, j in enumerate(cv1.inference_vc(source_speech_16k=source_speech_16k, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_voiceconversion1_{i}.wav', j['tts_speech'], cv1.sample_rate)

for i, j in enumerate(cv1.inference_vc(source_speech_16k=source_speech_16k, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/martial_law_voiceconversion2_{i}.wav', j['tts_speech'], cv1.sample_rate)

#   (2) cosyvoice2.
#         (2는 vc기능이 아직없지만 음성합성 퀄리티가 더 높습니다.)
for i, j in enumerate(cv2.inference_zero_shot(tts_text=tts_text, prompt_text=prompt_text1, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_zeroshot1_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_zero_shot(tts_text=tts_text, prompt_text=prompt_text2, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/martial_law_zeroshot2_{i}.wav', j['tts_speech'], cv2.sample_rate)

#   (3) 2를 이용하여 영문 텍스트를 음성으로 합성.
#         (zero_shot은 프롬프트를 잘 반영하고, cross_lingual은 억양을 살려줍니다.)
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=prompt_text1, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_zeroshot1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/martial_law_cross_lingual1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

#   (4) 소스음성(한국어) -> 텍스트로 transcribe -> 영어로 번역 -> 음성(영어)으로 합성해 보겠습니다.
#         whisper의 오디오 사용을 위해 터미널에서 ffmpeg를 설치합니다. (conda install ffmpeg -c conda-forge)
audio_path = 'custom_preprocessed/ysy_martial_law_shorter.wav'
audio_path2 = 'custom_preprocessed/서울의봄-실패반역성공혁명대사.mp3'
whisper_model = whisper.load_model('base')
transcribed_text = whisper_model.transcribe(audio_path, language='ko')['text']
transcribed_text2 = whisper_model.transcribe(audio_path2, language='ko')['text']

en_text = translator(transcribed_text, src_lang="kor_Hang", tgt_lang="eng_Latn")[0]['translation_text']
en_text = f'<|en|>{en_text}'

#        프롬프트 음성1:
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=prompt_text1, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual1_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

#        프롬프트 음성2:
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=prompt_text2, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot2_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=prompt_speech2_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual2_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

#        소스음성을 프롬프트로 사용:
for i, j in enumerate(cv2.inference_zero_shot(tts_text=en_text, prompt_text=transcribed_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot3_en_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=en_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual3_en_{i}.wav', j['tts_speech'], cv2.sample_rate)


# <<3. 목표인 다국어로 음성번역>>
#   우선 베트남어로 테스트 해봤습니다.
#   새 언어를 위해선 문장앞에 language token을 넣어줄 필요가있어 베트남어 토큰인 <|vi|>를 붙였습니다.
vi_text = translator(transcribed_text, src_lang="kor_Hang", tgt_lang="vie_Latn")[0]['translation_text']
vi_text = f'<|vi|>{vi_text}'
vi_text2 = translator(transcribed_text2, src_lang="kor_Hang", tgt_lang="vie_Latn")[0]['translation_text']
vi_text2 = f'<|vi|>{vi_text2}'

for i, j in enumerate(cv2.inference_zero_shot(tts_text=vi_text, prompt_text=transcribed_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_zeroshot3_vi_{i}.wav', j['tts_speech'], cv2.sample_rate)

for i, j in enumerate(cv2.inference_cross_lingual(tts_text=vi_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/final_cross_lingual3_vi_{i}.wav', j['tts_speech'], cv2.sample_rate)

# => 결과로 저장된 베트남어 음성(final_..._vi)은 의미없는 말만 나옵니다.
# tokenizer.py에 베트남어가 있긴하지만, 결국 모델을 파인튜닝 해야하는것 같습니다.
# https://github.com/FunAudioLLM/CosyVoice/issues/466,
# https://github.com/FunAudioLLM/CosyVoice/issues/344 를 참고해보니
# 새로운 언어의 cover를 위해선 적어도 5천시간 이상의 음성이 필요하다고 하지만,
# 저는 준비된 베트남어 음성이 없어서 구글 번역기의 음성으로 1~10초 정도의 샘플들을 만들어 사용했습니다.

# repo에 올려진 학습을 위한 recipe인 run.sh를 참고하여 샘플을 구성했으며,
# 샘플음성 5개 + 텍스트 5개로 하나의 test폴더를 만들었습니다.
# 학습용의 데이터폴더: test1, test2, test3.
# 테스트용의 데이터폴더: test4, test5.
# 이 후 run.sh를 제 환경에 맞게 수정하고,
# 저는 윈도우 환경이기에 처음으로 학습을 위한 리눅스환경을 만들어 source run.sh하여 돌려보았습니다.
# (제 환경은 윈도우 + WSL2 + Ubuntu 22.04 에 rtx 2080 super 한 장입니다)

# 학습 코드는 examples/libritts/cosyvoie/run.sh 참고해주세요!

# 처음 돌린 후 train stage에서 오류가 났지만, 
# torchaudio를 requirements와는 다른 2.0.2로 다운그레이드해주니 해결되었습니다.
# (https://github.com/FunAudioLLM/CosyVoice/issues/758 참고)

# 이후 잘 나온 모델들을 averaging하는 stage에서도 오류가 나서 보니,
# torchaudio를 다운그레이드하는 과정에서 같이 다운그레이드된 torch버전이 맞지 않아서 발생하는 것이었습니다.
# 이 부분은 torch를 다시 2.2+로 다운받아주면 되겠지만,
# averaging stage에서 best epoch과 best val들을 뽑아주는건 정상 작동하므로
# best epoch인 체크포인트를 불러와 쓰는걸로 하겠습니다.
# (https://github.com/FunAudioLLM/CosyVoice/issues/399 참고)

# 현재 llm training만을 지원한다니 llm을 학습하였고, 총 200 epoch를 돌렸으며, 모델들은 1.15GB * 200 = 230GB 정도의 용량을 차지했습니다.
# (tensorboard폴더에 train 로그 확인해 주세요!)
# 학습후 생성된 체크포인트 중 가장 잘 나온 모델(best val)은 다음과 같으며,
# best val (epoch, step, loss, tag) = [[19, 40, 3.4150915145874023, 'CV'], [20, 42, 3.4169666290283205, 'CV'], [18, 38, 3.4186872959136965, 'CV'], [21, 44, 3.4231380939483644, 'CV'], [17, 36, 3.425944185256958, 'CV']]
# 저는 이들 중 loss가 가장 낮은 19번째 에포크 모델을 사용하겠습니다.
# (19번째 에포크 모델 경로: examples/libritts/cosyvoice/exp/cosyvoice/llm/torch_ddp/epoch_19_whole.pt)

#   추론을 위해 이 모델을 pretrained_models/CosyVoice-300M/best_val/llm/epoch_19_whole.pt 경로에 넣어주고,
#   해당 체크포인트를 불러와 cosyvoice모델에 가중치를 로드해보겠습니다.
config_file = "./pretrained_models/CosyVoice-300M/cosyvoice.yaml"
override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}

with open(config_file, 'r') as f:
    configs = load_hyperpyyaml(f, overrides=override_dict)

model = configs["llm"]
checkpoint = torch.load("./pretrained_models/CosyVoice-300M/best_val/llm/epoch_19_whole.pt", map_location='cpu')
state_dict = {k: v for k, v in checkpoint.items() if k not in ["epoch", "step"]}
model.load_state_dict(state_dict)
cv1.model.llm = model.to("cuda").half()

#   학습이 잘 되었나 확인해봅니다.
for i, j in enumerate(cv1.inference_zero_shot(tts_text=vi_text, prompt_text=transcribed_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/dubbed1_vi_ver1_{i}.wav', j['tts_speech'], cv1.sample_rate)

for i, j in enumerate(cv1.inference_cross_lingual(tts_text=vi_text, prompt_speech_16k=source_speech_16k, stream=False)):
    torchaudio.save(f'output/dubbed1_vi_ver2_{i}.wav', j['tts_speech'], cv1.sample_rate)

for i, j in enumerate(cv1.inference_zero_shot(tts_text=vi_text2, prompt_text=transcribed_text2, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/dubbed2_vi_ver1_{i}.wav', j['tts_speech'], cv1.sample_rate)

for i, j in enumerate(cv1.inference_cross_lingual(tts_text=vi_text2, prompt_speech_16k=prompt_speech1_16k, stream=False)):
    torchaudio.save(f'output/dubbed2_vi_ver2_{i}.wav', j['tts_speech'], cv1.sample_rate)

#   합성한 오디오가 베트남어로 인식 되는지 확인해봅니다.
trained_audio_path1_1 = 'output/dubbed1_vi_ver1_0.wav'
trained_text1_1 = whisper_model.transcribe(trained_audio_path1_1, language='vi')['text']
trained_audio_path1_2 = 'output/dubbed1_vi_ver2_0.wav'
trained_text1_2 = whisper_model.transcribe(trained_audio_path1_2, language='vi')['text']
print('vi1_1:', trained_text1_1)
print('vi1_2:', trained_text1_2)

trained_audio_path2_1 = 'output/dubbed2_vi_ver1_0.wav'
trained_text2_1 = whisper_model.transcribe(trained_audio_path2_1, language='vi')['text']
trained_audio_path2_2 = 'output/dubbed2_vi_ver2_0.wav'
trained_text2_2 = whisper_model.transcribe(trained_audio_path2_2, language='vi')['text']
print('vi2_1:', trained_text2_1)
print('vi2_2:', trained_text2_2)

# 결과, 긴 길이의 오디오(dubbed1_vi)는 베트남어로 인식되지 않았습니다.
# 하지만 dubbed2_vi_ver2(즉, 짧은 길이의 오디오를 입력으로 넣은 cross_lingual)이 베트남어를 합성했습니다!

# 입력 음성(한국어)이 번역되고, 합성하게 될 텍스트(베트남어):
#   <|vi|> Nếu thất bại, nếu cuộc phản đối thành công, thì Ibiza của cuộc cách mạng sẽ thành công.
#   (만약 실패하고, 시위가 성공한다면, 혁명의 이비자는 성공할 것입니다.)
# 합성한 오디오(베트남어)를 whisper로 인식한 텍스트:
#   Nhi sạch ta bý sân nâ sạch chẳng xe chung chú Nhi phụ sân chân sân hả tổ sàng hân chân sạch hân nhau
#   (아이가 깨끗하고, 마당이 깨끗하고, 차가 함께 있지 않습니다. 아이는 마당의 아버지입니다.)


# 물론 상당히 다른 말을 합성했지만, 개인적으로는 학습에 성공하여 그럴듯한 베트남어를 합성했다는 것이 정말 기뻤습니다.
# 음성모델 학습이라고는 예전에 RVC를 webui를 통해 간편히 했던게 전부니까요.
# 이렇게 오픈소스를 이리저리 만져보다보니 재미도 있고 머신러닝에 대한 자신감도 붙는 느낌입니다.
# CosyVoice모델 개발자의 말인 새로운 언어의 cover시간 5천시간에는 택도 없는 극소량의 데이터셋이었지만,
# 좀 더 풍부한 데이터셋과 함께, 더 좋은 환경에서, 훌륭한 사람들과 양질의 대화를 나누며 제 자신을 성장시키고 싶습니다!


# 오픈소스로 이런 간단한 프로젝트를 만들어보는건 처음이고, 부족함을 많이 느꼈습니다.
# 머신러닝 초보라서, 공부 중에 케글이나 깃헙에 올라오는 실력자들을 보면 매번 대단함을 느낍니다.
# 하지만 만드는 과정이 재미있고, 조금만 더 하면 할 수 있을 것 같다는 열정도 생깁니다.
# 저도 언젠가 이런 신기술들을 실생활에 적용하는 데에 기여할 수 있다면 좋겠다는 생각을 합니다.
# 언젠가 올 강 인공지능 시대에 뒤쳐지지 않기 위해 저의 머신러닝 역량을 키우고 싶습니다.
