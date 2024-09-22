import os
import torch
import argparse
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
)

def get_parser() -> argparse.Namespace:
    '''argument parser'''
    parser = argparse.ArgumentParser(prog='Whisper inference')
    parser.add_argument(
        '--base-model', '-b',
        required=True,
        help='Pretrained model from huggingface'
    )
    parser.add_argument(
        '--pretrained-model', '-p',
        required=True,
        help='Finetuned model directory'
    )
    parser.add_argument(
        '--audio-dir', '-d',
        required=True,
        help='Directory containing audio (source) files for ASR processing'
    )
    parser.add_argument(
        '--lang',
        default='korean',
        help='Target language after processing, default: korean'
    )
    parser.add_argument(
        '--task',
        default='transcribe',
        help='ASR task: translate or transcribe, default: transcribe'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sampling rate for voice file, default: 16,000'
    )
    config = parser.parse_args()
    return config


class Inference:
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_type = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model, 
            torch_dtype=self.torch_type,
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            device_map="auto",  # GPU 자동 감지
            trust_remote_code=False
        )

        self.processor = AutoProcessor.from_pretrained(self.config.base_model)
        # device 인자를 제거합니다. 모델이 자동으로 GPU를 사용할 것입니다.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_type,
            chunk_length_s=30
        )

    def run(self) -> None:
        '''Perform inference on all audio files in the directory'''
        options = {
            'language': self.config.lang,
            'task': self.config.task
        }

        # Audio 파일이 있는 디렉토리 내 모든 .wav 파일을 처리
        audio_files = [f for f in os.listdir(self.config.audio_dir) if f.endswith('.wav')]
        for audio_file in audio_files:
            audio_path = os.path.join(self.config.audio_dir, audio_file)
            print(f"Processing file: {audio_file}")
            result = self.pipe(audio_path, generate_kwargs=options)

            # 결과 출력
            if isinstance(result, dict):
                print(f"Transcription for {audio_file}: {result['text']}\n")
            else:
                text = ''
                for chunk in result:
                    text += chunk['text']
                print(f"Transcription for {audio_file}: {text}\n")


if __name__=='__main__':
    config = get_parser()
    inference = Inference(config)
    inference.run()
