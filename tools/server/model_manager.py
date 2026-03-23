import threading

import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        bnb4: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
        max_seq_len: int = 4096,
        lazy_load: bool = True,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile
        self.bnb4 = bnb4
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_checkpoint_path = decoder_checkpoint_path
        self.decoder_config_name = decoder_config_name
        self.max_seq_len = max_seq_len
        self.lazy_load = lazy_load

        self.precision = torch.half if half else torch.bfloat16
        self.tts_inference_engine = None

        self._loaded = False
        self._load_lock = threading.Lock()
        if self.lazy_load:
            logger.info("ModelManager ready (lazy load enabled; models load on first request).")
        else:
            logger.info("ModelManager starting in eager mode.")
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """Trigger model loading on first call (thread-safe)."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            logger.info("First request received — loading models now...")
            self._load_all_models()
            self._loaded = True

    def _load_all_models(self) -> None:
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        self.load_llama_model(
            self.llama_checkpoint_path,
            self.device,
            self.precision,
            self.compile,
            self.mode,
            max_seq_len=self.max_seq_len,
            bnb4=self.bnb4,
        )
        self.load_decoder_model(
            self.decoder_config_name, self.decoder_checkpoint_path, self.device
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )
        if self.mode == "tts":
            self.warm_up(self.tts_inference_engine)

    def load_llama_model(
        self,
        checkpoint_path,
        device,
        precision,
        compile,
        mode,
        max_seq_len: int = 4096,
        bnb4: bool = False,
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
                max_seq_len=max_seq_len,
                bnb4=bnb4,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("Decoder model loaded.")

    def warm_up(self, tts_inference_engine) -> None:
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        list(inference(request, tts_inference_engine))
        logger.info("Models warmed up.")
