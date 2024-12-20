#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0
stop_stage=7

data_url=www.openslr.org/resources/60
# data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
data_dir=/home/renkolinux/custom_data/vietnamese
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
# pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

# stage -1을 실행해 데이터 구조를 파악 후, 준비한 베트남어 데이터폴더인 test1, 2, 3, 4만 for문에 필요하므로 수정합니다.
# prepare_data.py를 확인해보니 test폴더 밑에 하위폴더 두 개가 있어야하고, 그 안에 wav파일과 txt파일이 있는 식입니다.
# stage0 에서 custom_data/vietnamese경로에 있는 test폴더들이 example/libritts/cosyvoice/data경로로 전처리 준비되어집니다.
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in test1 test2 test3 test4; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in test1 test2 test3 test4; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in test1 test2 test3 test4; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
      # --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in test1 test2 test3 test4; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# 프롬프트가 test1을 참고하게 해줍니다.
# tts_text는 보아하니 {"wav파일이름":["txt내용"]}형식인것 같으니 준비한 베트남버전에 맞게 vi버전으로 지정해줍니다.
# inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for mode in sft zero_shot; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice.yaml \
      --prompt_data data/test1/parquet/data.list \
      --prompt_utt2data data/test1/parquet/utt2data.list \
      --tts_text `pwd`/tts_text_vi.json \
      --llm_model $pretrained_model_dir/llm.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir `pwd`/exp/cosyvoice/test1/$mode
  done
fi

# train과 dev로 저장될 데이터를 각각 준비한 {test1,2}, {test3,4}로 바꿔줍니다.
# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=1
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat data/{test1,test2}/parquet/data.list > data/train.data.list
  cat data/{test3,test4}/parquet/data.list > data/dev.data.list
  for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=4
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

# cosyvoice-300M 모델 경로에
# 훈련된 모델(llm.text_encoder.fp16.zip, llm.llm.fp16.zip, flow.encoder.fp32.zip)이 저장됩니다.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi