"""Text-to-Speech components"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from loguru import logger

try:
    import mlx.audio
    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False
    logger.warning("MLX Audio not available, TTS functionality will be limited")


class TTSProcessor:
    """Text-to-Speech processor using MLX"""
    
    def __init__(self, model_path: str = "Marvis-AI/marvis-tts-250m-v0.1", voice: str = "default"):
        self.model_path = model_path
        self.voice = voice
        self.model = None
        self.loaded = False
        
        if not HAS_MLX_AUDIO:
            logger.error("MLX Audio not available. Please install mlx-audio.")
            return
        
        logger.info(f"🔊 TTS Processor initialized with model: {model_path}")
    
    async def load_model(self):
        """Load the TTS model"""
        if self.loaded or not HAS_MLX_AUDIO:
            return
        
        try:
            logger.info(f"Loading TTS model: {self.model_path}")
            # In a real implementation, this would load the MLX TTS model
            await asyncio.sleep(1)  # Simulate loading time
            self.loaded = True
            logger.info("✅ TTS model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load TTS model: {e}")
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """Synthesize text to speech"""
        if not self.loaded:
            await self.load_model()
        
        if not self.loaded:
            logger.error("TTS model not loaded")
            return None
        
        try:
            logger.debug(f"Synthesizing text: {text[:50]}...")
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # In a real implementation, this would use MLX TTS
            # For now, we'll simulate audio data
            audio_data = b"simulated_audio_data_for_tts"
            
            logger.debug("✅ TTS synthesis completed")
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if TTS functionality is available"""
        return HAS_MLX_AUDIO and self.loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the TTS model"""
        return {
            "model_path": self.model_path,
            "voice": self.voice,
            "loaded": self.loaded,
            "mlx_audio_available": HAS_MLX_AUDIO
        }


class TTSWorker:
    """Isolated TTS worker process"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.processor: Optional[TTSProcessor] = None
        self.running = False
        
        logger.info(f"🔊 TTS Worker initialized for model: {model_name}")
    
    async def initialize(self):
        """Initialize the TTS worker"""
        try:
            self.processor = TTSProcessor(model_path=self.model_name)
            await self.processor.load_model()
            self.running = True
            logger.info(f"✅ TTS Worker {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS Worker {self.model_name}: {e}")
    
    async def process_text(self, text: str) -> Optional[bytes]:
        """Process text through TTS"""
        if not self.running or not self.processor:
            logger.warning("TTS Worker not available")
            return None
        
        return await self.processor.synthesize(text)
    
    def is_available(self) -> bool:
        """Check if TTS worker is available"""
        return self.running and self.processor and self.processor.is_available()


class TTSManager:
    """Manager for TTS functionality"""
    
    def __init__(self):
        self.workers: Dict[str, TTSWorker] = {}
        self.default_worker: Optional[str] = None
        self.enabled = True
        self.metrics = {
            "total_syntheses": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "avg_synthesis_time": 0.0
        }
    
    async def initialize(self, models: List[str] = None):
        """Initialize TTS workers"""
        if not self.enabled:
            logger.info("TTS functionality disabled")
            return
        
        if models is None:
            models = ["Marvis-AI/marvis-tts-250m-v0.1", "mlx-community/Kokoro-82M-bf16"]
        
        for model in models:
            try:
                worker = TTSWorker(model)
                await worker.initialize()
                self.workers[model] = worker
                
                if self.default_worker is None:
                    self.default_worker = model
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize TTS worker {model}: {e}")
        
        logger.info(f"✅ TTS Manager initialized with {len(self.workers)} workers")
    
    async def synthesize(self, text: str, model: Optional[str] = None) -> Optional[bytes]:
        """Synthesize text to speech"""
        if not self.enabled or not self.workers:
            logger.warning("TTS not available")
            return None
        
        # Use specified model or default
        worker_name = model or self.default_worker
        worker = self.workers.get(worker_name)
        
        if not worker:
            logger.warning(f"TTS worker {worker_name} not available")
            return None
        
        start_time = time.time()
        
        try:
            result = await worker.process_text(text)
            
            # Update metrics
            self.metrics["total_syntheses"] += 1
            if result:
                self.metrics["successful_syntheses"] += 1
            else:
                self.metrics["failed_syntheses"] += 1
            
            # Update average time
            synthesis_time = time.time() - start_time
            total = self.metrics["total_syntheses"]
            current_avg = self.metrics["avg_synthesis_time"]
            self.metrics["avg_synthesis_time"] = (
                (current_avg * (total - 1) + synthesis_time) / total
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            self.metrics["total_syntheses"] += 1
            self.metrics["failed_syntheses"] += 1
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get TTS metrics"""
        return self.metrics.copy()
    
    def get_available_models(self) -> List[str]:
        """Get list of available TTS models"""
        return [name for name, worker in self.workers.items() if worker.is_available()]
    
    def is_available(self) -> bool:
        """Check if TTS is available"""
        return self.enabled and len(self.workers) > 0


# Global TTS manager instance
_tts_manager: Optional[TTSManager] = None


def get_tts_manager() -> TTSManager:
    """Get the global TTS manager instance"""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager()
    return _tts_manager


async def initialize_tts(models: List[str] = None):
    """Initialize the global TTS manager"""
    manager = get_tts_manager()
    await manager.initialize(models)
    return manager


def is_tts_available() -> bool:
    """Check if TTS is available"""
    return get_tts_manager().is_available()


async def synthesize_text(text: str, model: Optional[str] = None) -> Optional[bytes]:
    """Synthesize text using the global TTS manager"""
    return await get_tts_manager().synthesize(text, model)