##
[![SVG Banners](https://svg-banners.vercel.app/api?type=luminance&text1=AI더빙솔루션%20🤖&width=800&height=200)](https://github.com/Akshay090/svg-banners)  

##

## AI 더빙 솔루션 만들기
알파시그널(https://alphasignal.ai/) 을 통해 괜찮은 오디오 합성 모델을 찾게된 것 같아 포크했습니다.  
emotion을 감지하고, voice conversion, 제로샷 합성, 다국어음성을 지원하여 본 목적에 적합하다고 생각합니다.  

목표로하는 한국드라마의 음성을 다국어 음성으로 번역해서 출력하는 코드를 짜봤습니다.  
그 중 베트남어를 학습하여 베트남어를 합성하도록 해봤습니다.

<br/>

### 오픈소스의 repo와 이 포크버전이 다른곳  
- 베트남어 데이터셋 폴더 추가: `custom_data/vietnamese`  
- 프롬프트용 오디오 폴더 추가: `custom_preprocessed`  
- 트레이닝용 recipe 수정: `examples/libritts/cosyvoice/run.sh`  
- 합성한 음성 폴더 추가: `output`  
- 이 프로젝트의 주요 학습 과정이 담긴 코드 추가: `test_usage2.py`  
- 학습한 모델로 오디오를 합성하는 코드 추가: `AI_dubbing_solution.py`  
- training 로그를 담은 텐서보드 폴더 추가: `tensorboard`
  
### 살펴보실 주요 파일
- `test_usage2.py`  
- `examples/libritts/cosyvoice/run.sh`  
- `AI_dubbing_solution.py`  
- `tensorboard`

  이 네 개만 보셔도 될 것 같습니다!

  
### 실행
- 밑의 원본 CosyVoice의 설명대로 가상환경과 requirements까지 설치
- 모델은 `CosyVoice2-0.5B`, `CosyVoice-300M`만 설치

  `test_usage2.py`와 `AI_dubbing_solution.py`은 윈도우에서도 Run 가능합니다!  
  학습을 위한 `examples/libritts/cosyvoice/run.sh`은 리눅스환경(혹은 윈도우 + WSL2 + Ubuntu 환경)에서 실행 가능합니다!

<br/>

### 결과물 예시(en)
- 입력:
  
  https://github.com/user-attachments/assets/995d1199-9fad-4929-b484-ab5ee8e06ab3  

<br/>

- 출력(입력과 동일한 프롬프트) - zeroshot:
  
  https://github.com/user-attachments/assets/f6f63d78-13f3-488e-8dab-1daeeecded57  

- 출력(입력과 동일한 프롬프트) - cross lingual:

  https://github.com/user-attachments/assets/96d4cf19-76ac-4869-9b18-9d531c91b0a4


<br/><br/>

### 학습한 후 텐서보드 분석
CosyVoice모델 개발자 말로는 새로운 언어를 파는데에 최소 5천시간의 데이터셋이 필요하다고 합니다.(https://github.com/FunAudioLLM/CosyVoice/issues/466)  
하지만 그러한 양의 데이터는 갖고 있지 않고, 결과를 빠르게 보고싶었기에 구글 번역기로 급조한 극소량의 베트남어 데이터로 한 번 돌려봤습니다.  
![cosyvoice_train_log_llm](https://github.com/user-attachments/assets/566773f8-eed4-4ce9-ab94-f2058cc89671)

- TRAIN/acc(훈련 정확도):  
  100스텝 부근부터 급격하게 정확도가 증가해 1에 가깝게 수렴했습니다. 너무 급격히 증가했고 높은 정확도기에 과적합인 것 같습니다. 적은 데이터셋이 원인인 것 같습니다.
- TRAIN/loss(훈련 손실):  
  마찬가지로 급격히 감소하고 0과 가깝게 수렴하고 있으므로 과적합인 것 같습니다.
- CV/acc(검증 정확도):  
  50스텝이 조금 안 되는 지점까지는 정확도가 증가하다가 점점 감소하고 있습니다. 데이터가 얼마 안되어 생긴 과적합 같으므로, 데이터를 늘릴 필요가 있어보입니다.
- CV/loss(검증 손실):  
  역시 50스텝이 조금 안 되는 지점까지는 loss가 감소하다가 점점 증가하고 있습니다. 훈련 손실이 떨어졌던것과 비교하면 역시 일반화성능은 떨어져 과적합되었음을 보여줍니다.
- TRAIN/lr(학습률):  
  examples/libritts/cosyvoice/conf/cosyvoice.yaml 에서 셋팅한걸 보면, scheduler는 warmuplr으로 되어 있어 점점 learning rate이 증가하는 걸 볼 수 있습니다.
- TRAIN/epoch(훈련 에포크):  
  총 200에포크이며 에포크 당 2스텝으로 보여집니다.

꽤 큰 규모의 데이터셋(https://github.com/FunAudioLLM/CosyVoice/issues/344) 에서도 저와 비슷한 양상을 보이는 것 같기는 합니다.  
저대로도 러시아어를 꽤나 잘 말하게 됐다지만 결국 llm학습 이상의 것이 필요한 것 같다네요.  
지금의 전 기껏해야 recipe를 보고 따라하는 수준이기에 언젠간 성장해서 꼭 완벽하게 학습시켜보고 싶습니다.  

<br/>

### (선택)미리 학습한 체크포인트 다운
학습후 best val로 뽑힌 체크포인트를 pretrained_models/cosyvoice-300M/best_val/llm/ 의 경로에 두었습니다.  
혹시나 학습절차는 생략하고 미리 학습한 체크포인트를 다운받아 보시려면 밑의 커맨드를 실행해 주세요!  
``` sh
git clone https://huggingface.co/Usamimi/cosyvoice-300M_checkpoint19.git pretrained_models/cosyvoice-300M
```

<br/>

### 결과물 예시(vi): 학습한 모델로 합성
- 입력:

https://github.com/user-attachments/assets/5452b368-1b7c-44dd-a650-cda1ba459ff7  

<br/>

- 출력:

https://github.com/user-attachments/assets/29c9c1a5-71db-4972-b8d0-a2e7a0544756

<br/><br/>
데이터셋이 적다보니 원하는 말을 합성하진 못했습니다.  
하지만 위 결과물 음성을 whisper로 transcribe시,  
적어도 베트남어로 인식을 하는데에 성공 했습니다!  

앞으로 다량의 데이터 뿐만 아니라, 다양한 사람들과 양질의 대화를 나누며 성장해서 이런 모델쯤은 당연히 완벽하게 학습 시키는 사람으로 성장하고 싶습니다.  
이 프로젝트를 하면서 부족한 점도 많이 느꼈지만, 무엇보다 재밌었습니다.  
저는 강 인공지능 시대에 뒤쳐지지 않기 위해 탐구하는 자세를 굽히지 않고,
머신러닝에 대한 역량을 키워가고 싶습니다.
<br/><br/><br/><br/>

## 
[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice🤠&text2=Text-to-Speech%20💖%20Large%20Language%20Model&width=800&height=210)](https://github.com/Akshay090/svg-banners)

## 👉🏻 CosyVoice 👈🏻
**CosyVoice 2.0**: [Demos](https://funaudiollm.github.io/cosyvoice2/); [Paper](https://arxiv.org/abs/2412.10117); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B)

**CosyVoice 1.0**: [Demos](https://fun-audio-llm.github.io); [Paper](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice-300M)

## Highlight🔥

**CosyVoice 2.0** has been released! Compared to version 1.0, the new version offers more accurate, more stable, faster, and better speech generation capabilities.
### Multilingual
- **Support Language**: Chinese, English, Japanese, Korean, Chinese dialects (Cantonese, Sichuanese, Shanghainese, Tianjinese, Wuhanese, etc.)
- **Crosslingual & Mixlingual**：Support zero-shot voice cloning for cross-lingual and code-switching scenarios.
### Ultra-Low Latency
- **Bidirectional Streaming Support**: CosyVoice 2.0 integrates offline and streaming modeling technologies.
- **Rapid First Packet Synthesis**: Achieves latency as low as 150ms while maintaining high-quality audio output.
### High Accuracy
- **Improved Pronunciation**: Reduces pronunciation errors by 30% to 50% compared to CosyVoice 1.0.
- **Benchmark Achievements**: Attains the lowest character error rate on the hard test set of the Seed-TTS evaluation set.
### Strong Stability
- **Consistency in Timbre**: Ensures reliable voice consistency for zero-shot and cross-language speech synthesis.
- **Cross-language Synthesis**: Marked improvements compared to version 1.0.
### Natural Experience
- **Enhanced Prosody and Sound Quality**: Improved alignment of synthesized audio, raising MOS evaluation scores from 5.4 to 5.53.
- **Emotional and Dialectal Flexibility**: Now supports more granular emotional controls and accent adjustments.

## Roadmap

- [x] 2024/12

    - [x] 25hz cosyvoice 2.0 released

- [x] 2024/09

    - [x] 25hz cosyvoice base model
    - [x] 25hz cosyvoice voice conversion model

- [x] 2024/08

    - [x] Repetition Aware Sampling(RAS) inference for llm stability
    - [x] Streaming inference mode support, including kv cache and sdpa for rtf optimization

- [x] 2024/07

    - [x] Flow matching training support
    - [x] WeTextProcessing support when ttsfrd is not avaliable
    - [x] Fastapi server and client


## Install

**Clone and install**

- Clone the repo
``` sh
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd CosyVoice
git submodule update --init --recursive
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n cosyvoice python=3.10
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

**Model download**

We strongly recommend that you download our pretrained `CosyVoice2-0.5B` `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource.

If you are expert in this field, and you are only interested in training your own CosyVoice model from scratch, you can skip this step.

``` python
# SDK模型下载
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

Optionaly, you can unzip `ttsfrd` resouce and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use WeTextProcessing by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

**Basic Usage**

We strongly recommend using `CosyVoice2-0.5B` for better performance.
For zero_shot/cross_lingual inference, please use `CosyVoice-300M` model.
For sft inference, please use `CosyVoice-300M-SFT` model.
For instruct inference, please use `CosyVoice-300M-Instruct` model.
First, add `third_party/Matcha-TTS` to your `PYTHONPATH`.

``` sh
export PYTHONPATH=third_party/Matcha-TTS
```

``` python
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
```

**CosyVoice2 Usage**
```python
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

**CosyVoice Usage**
```python
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=True, load_onnx=False, fp16=True)
# sft usage
print(cosyvoice.list_avaliable_spks())
# change stream=True for chunk stream inference
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
    torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz') # or change to pretrained_models/CosyVoice-300M for 50Hz inference
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cross_lingual usage
prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
    torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# vc usage
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
source_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
    torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

**Start web demo**

You can use our web demo page to get familiar with CosyVoice quickly.
We support sft/zero_shot/cross_lingual/instruct inference in web demo.

Please see the demo website for details.

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

**Advanced Usage**

For advanced user, we have provided train and inference scripts in `examples/libritts/cosyvoice/run.sh`.
You can get familiar with CosyVoice following this recipie.

**Build for deployment**

Optionally, if you want to use grpc for service deployment,
you can run following steps. Otherwise, you can just ignore this step.

``` sh
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our official Dingding chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
