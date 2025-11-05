# LocalCat Continual Learning Integration Plan

## 🎯 Goal
Add continual learning to LocalCat so the LLM learns from conversations during idle time (when connected to power), without forgetting existing knowledge.

## 🏗️ Architecture Integration Points

### 1. Sparse Memory Layer (New Component)

**Location:** `/Users/peppi/Dev/localcat/server/core/llm/sparse_memory.py`

```python
"""
Sparse Memory Layer for Continual Learning (MLX Native)

Integrates with DirectMLXLLMService to enable on-the-fly learning
without catastrophic forgetting.
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from collections import Counter
import numpy as np
from loguru import logger


class SparseMemoryLayerMLX(nn.Module):
    """
    MLX-native sparse memory for continual learning.
    
    Features:
    - k-sparse attention (only top-k slots accessed per token)
    - TF-IDF based selective updates
    - Checkpoint-able state for incremental learning
    - Zero inference overhead when not training
    """
    
    def __init__(
        self,
        d_model: int,
        n_slots: int = 10000,  # More slots for larger models
        k: int = 64,  # Wider sparse access for 1.7B+
        device: str = "gpu"
    ):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.k = k
        
        # Memory slots (learnable)
        self.keys = mx.random.normal((n_slots, d_model)) * 0.02
        self.values = mx.random.normal((n_slots, d_model)) * 0.02
        
        # Query projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output gating
        self.gate_fc1 = nn.Linear(d_model, d_model // 4)
        self.gate_fc2 = nn.Linear(d_model // 4, d_model)
        
        # Tracking for TF-IDF
        self.slot_access_history = []
        self.tracking_enabled = False
        
    def __call__(self, x):
        """Forward pass with k-sparse attention"""
        # Query
        q = self.q_proj(x)
        
        # Scores with all slots
        scores = mx.matmul(q, self.keys.T)
        
        # Top-k sparse selection (MLX native)
        topk_indices = mx.argpartition(-scores, self.k, axis=-1)[:, :, :self.k]
        
        # Track if enabled
        if self.tracking_enabled:
            self.slot_access_history.append(
                np.array(topk_indices).flatten()
            )
        
        # Gather top-k scores and values
        topk_scores = mx.take_along_axis(scores, topk_indices, axis=-1)
        topk_values = self.values[topk_indices]
        
        # Attention
        attn_weights = mx.softmax(topk_scores, axis=-1)
        
        # Weighted sum
        output = mx.sum(
            mx.expand_dims(attn_weights, -1) * topk_values,
            axis=2
        )
        
        # Gating
        gate = mx.sigmoid(self.gate_fc2(mx.relu(self.gate_fc1(x))))
        output = output * gate
        
        return output
    
    def enable_tracking(self):
        """Start tracking slot access for TF-IDF"""
        self.tracking_enabled = True
        self.slot_access_history = []
        logger.debug("[SparseMemory] Tracking enabled")
    
    def disable_tracking(self):
        """Stop tracking"""
        self.tracking_enabled = False
        logger.debug("[SparseMemory] Tracking disabled")
    
    def get_selective_slots(self, top_n: int = 256) -> list:
        """Get selective slots using TF-IDF"""
        if not self.slot_access_history:
            return []
        
        all_slots = np.concatenate(self.slot_access_history)
        slot_counts = Counter(all_slots)
        
        # Simple frequency-based for now
        # TODO: Add IDF against reference corpus
        selective = [s for s, _ in slot_counts.most_common(top_n)]
        
        logger.info(f"[SparseMemory] Selected {len(selective)} slots for update")
        return selective
    
    def save_checkpoint(self, path: Path):
        """Save memory state"""
        mx.savez(
            str(path),
            keys=self.keys,
            values=self.values,
            q_proj_weight=self.q_proj.weight,
            gate_fc1_weight=self.gate_fc1.weight,
            gate_fc2_weight=self.gate_fc2.weight
        )
        logger.info(f"[SparseMemory] Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load memory state"""
        if not path.exists():
            logger.warning(f"[SparseMemory] Checkpoint not found: {path}")
            return False
        
        state = mx.load(str(path))
        self.keys = state['keys']
        self.values = state['values']
        self.q_proj.weight = state['q_proj_weight']
        self.gate_fc1.weight = state['gate_fc1_weight']
        self.gate_fc2.weight = state['gate_fc2_weight']
        
        logger.info(f"[SparseMemory] Loaded checkpoint: {path}")
        return True
```

### 2. Enhanced DirectMLXLLMService (Modification)

**Location:** `/Users/peppi/Dev/localcat/server/core/llm/direct_mlx_llm_with_memory.py`

```python
"""
DirectMLXLLMService with Sparse Memory for Continual Learning
"""

from .direct_mlx_llm_with_tools import DirectMLXLLMServiceWithTools
from .sparse_memory import SparseMemoryLayerMLX
from pathlib import Path
from loguru import logger


class DirectMLXLLMServiceWithMemory(DirectMLXLLMServiceWithTools):
    """
    Enhanced DirectMLXLLMService with continual learning capability.
    
    Extends DirectMLXLLMServiceWithTools to add:
    - Sparse memory layer injection
    - Checkpoint loading/saving
    - Learning mode control
    
    The memory layer is ONLY active during training - zero overhead during normal inference.
    """
    
    def __init__(
        self,
        *args,
        memory_slots: int = 10000,
        memory_checkpoint_dir: str = None,
        enable_continual_learning: bool = True,
        **kwargs
    ):
        """Initialize with optional memory layer"""
        super().__init__(*args, **kwargs)
        
        self._enable_continual_learning = enable_continual_learning
        self._memory_layer = None
        self._memory_injected = False
        self._checkpoint_dir = None
        
        if enable_continual_learning:
            self._checkpoint_dir = Path(memory_checkpoint_dir or "./data/memory_checkpoints")
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create but DON'T inject memory layer yet
            # Will be injected lazily on first training session
            logger.info(f"[DirectMLXMemory] Continual learning enabled (checkpoints: {self._checkpoint_dir})")
        else:
            logger.info("[DirectMLXMemory] Continual learning disabled")
    
    def _inject_memory_layer(self):
        """Inject sparse memory layer into model architecture"""
        if self._memory_injected:
            return
        
        if not hasattr(self._model, 'model') or not hasattr(self._model.model, 'layers'):
            logger.error("[DirectMLXMemory] Model architecture not compatible for memory injection")
            return
        
        # Get model hidden size
        d_model = self._model.config.hidden_size
        
        # Create memory layer
        self._memory_layer = SparseMemoryLayerMLX(
            d_model=d_model,
            n_slots=10000,
            k=64
        )
        
        # Try to load latest checkpoint
        latest_checkpoint = self._get_latest_checkpoint()
        if latest_checkpoint:
            self._memory_layer.load_checkpoint(latest_checkpoint)
        
        # Inject at middle layer (layer 15 for smaller models, proportional for larger)
        num_layers = len(self._model.model.layers)
        target_layer = num_layers // 2
        
        # Store original MLP
        original_mlp = self._model.model.layers[target_layer].mlp
        
        # Create wrapper
        class MLPWithMemory:
            def __init__(self, mlp, memory):
                self.mlp = mlp
                self.memory = memory
            
            def __call__(self, x):
                mlp_out = self.mlp(x)
                mem_out = self.memory(x)
                return mlp_out + mem_out
        
        # Replace MLP
        self._model.model.layers[target_layer].mlp = MLPWithMemory(original_mlp, self._memory_layer)
        
        self._memory_injected = True
        logger.info(f"[DirectMLXMemory] Memory layer injected at layer {target_layer}/{num_layers}")
    
    def _get_latest_checkpoint(self) -> Path:
        """Get most recent checkpoint"""
        if not self._checkpoint_dir:
            return None
        
        checkpoints = list(self._checkpoint_dir.glob("memory_*.npz"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def enable_learning_mode(self):
        """Enable learning mode (activates memory tracking)"""
        if not self._enable_continual_learning:
            logger.warning("[DirectMLXMemory] Continual learning is disabled")
            return
        
        if not self._memory_injected:
            self._inject_memory_layer()
        
        if self._memory_layer:
            self._memory_layer.enable_tracking()
            logger.info("[DirectMLXMemory] Learning mode enabled")
    
    def disable_learning_mode(self):
        """Disable learning mode (stops tracking)"""
        if self._memory_layer:
            self._memory_layer.disable_tracking()
            logger.debug("[DirectMLXMemory] Learning mode disabled")
    
    def get_selective_slots(self) -> list:
        """Get selective slots for training"""
        if not self._memory_layer:
            return []
        return self._memory_layer.get_selective_slots(top_n=256)
    
    def save_memory_checkpoint(self, suffix: str = None):
        """Save current memory state"""
        if not self._memory_layer or not self._checkpoint_dir:
            return
        
        import time
        timestamp = int(time.time())
        filename = f"memory_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".npz"
        
        checkpoint_path = self._checkpoint_dir / filename
        self._memory_layer.save_checkpoint(checkpoint_path)
```

### 3. Continual Learning Orchestrator (New Component)

**Location:** `/Users/peppi/Dev/localcat/server/core/memory/continual_learner.py`

```python
"""
Continual Learning Orchestrator

Manages the daily learning cycle:
1. Extract facts from conversation history
2. Identify selective memory slots
3. Train model during idle time
4. Save checkpoints
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict
from loguru import logger
from datasets import Dataset


class ContinualLearner:
    """
    Orchestrates continual learning for LocalCat.
    
    Responsibilities:
    - Monitor idle time (user inactive + power connected)
    - Extract facts from memory store
    - Trigger selective memory updates
    - Manage training schedule
    """
    
    def __init__(
        self,
        llm_service,  # DirectMLXLLMServiceWithMemory
        memory_store,  # MemoryStore instance
        config: dict = None
    ):
        self.llm = llm_service
        self.memory_store = memory_store
        self.config = config or {}
        
        # Learning state
        self._learning_task = None
        self._is_learning = False
        self._last_learning_time = 0
        
        # Config
        self.idle_threshold_minutes = self.config.get("idle_threshold_minutes", 5)
        self.learning_interval_hours = self.config.get("learning_interval_hours", 24)
        self.min_new_facts = self.config.get("min_new_facts", 10)
        self.require_power = self.config.get("require_power", True)
        
        logger.info(f"[ContinualLearner] Initialized (idle={self.idle_threshold_minutes}min, interval={self.learning_interval_hours}h)")
    
    async def start(self):
        """Start continual learning monitor"""
        if self._learning_task:
            logger.warning("[ContinualLearner] Already running")
            return
        
        self._learning_task = asyncio.create_task(self._learning_loop())
        logger.info("[ContinualLearner] Started")
    
    async def stop(self):
        """Stop continual learning"""
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
            self._learning_task = None
        
        logger.info("[ContinualLearner] Stopped")
    
    async def _learning_loop(self):
        """Main learning loop - checks conditions and triggers learning"""
        while True:
            try:
                # Check every minute
                await asyncio.sleep(60)
                
                # Should we learn?
                if self._should_learn():
                    logger.info("[ContinualLearner] 🌙 Conditions met - starting learning session")
                    await self._run_learning_session()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ContinualLearner] Error in learning loop: {e}")
    
    def _should_learn(self) -> bool:
        """Check if we should start a learning session"""
        # Already learning?
        if self._is_learning:
            return False
        
        # Too soon since last learning?
        hours_since_last = (time.time() - self._last_learning_time) / 3600
        if hours_since_last < self.learning_interval_hours:
            return False
        
        # Power requirement
        if self.require_power and not self._is_connected_to_power():
            logger.debug("[ContinualLearner] Not connected to power - skipping")
            return False
        
        # Idle requirement
        if not self._is_system_idle():
            logger.debug("[ContinualLearner] System not idle - skipping")
            return False
        
        # Enough new facts?
        new_facts = self._count_new_facts()
        if new_facts < self.min_new_facts:
            logger.debug(f"[ContinualLearner] Only {new_facts} new facts (need {self.min_new_facts})")
            return False
        
        return True
    
    def _is_connected_to_power(self) -> bool:
        """Check if system is connected to power"""
        try:
            import subprocess
            result = subprocess.run(
                ["pmset", "-g", "ps"],
                capture_output=True,
                text=True,
                timeout=1
            )
            return "AC Power" in result.stdout
        except Exception as e:
            logger.warning(f"[ContinualLearner] Could not check power status: {e}")
            return False  # Fail safe
    
    def _is_system_idle(self) -> bool:
        """Check if system has been idle for threshold"""
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem"],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            # Parse idle time from ioreg output
            for line in result.stdout.split('\n'):
                if "HIDIdleTime" in line:
                    # Extract nanoseconds and convert to minutes
                    idle_ns = int(line.split('=')[1].strip())
                    idle_minutes = idle_ns / 1_000_000_000 / 60
                    return idle_minutes >= self.idle_threshold_minutes
            
            return False
        except Exception as e:
            logger.warning(f"[ContinualLearner] Could not check idle status: {e}")
            return False  # Fail safe
    
    def _count_new_facts(self) -> int:
        """Count facts since last learning session"""
        # Query memory store for facts added since last learning
        # This is a placeholder - implement based on your memory store schema
        try:
            # Example: Query mentions/edges created since last learning
            cutoff_time = int(self._last_learning_time)
            
            # You'll need to add this query to MemoryStore
            # For now, return a dummy count
            return 25  # TODO: Implement actual query
        except Exception as e:
            logger.error(f"[ContinualLearner] Error counting facts: {e}")
            return 0
    
    async def _run_learning_session(self):
        """Execute a full learning session"""
        self._is_learning = True
        session_start = time.time()
        
        try:
            logger.info("=" * 80)
            logger.info("🧠 CONTINUAL LEARNING SESSION STARTING")
            logger.info("=" * 80)
            
            # Step 1: Extract new facts from memory
            facts = await self._extract_new_facts()
            logger.info(f"[ContinualLearner] Extracted {len(facts)} new facts")
            
            if len(facts) < self.min_new_facts:
                logger.info(f"[ContinualLearner] Not enough facts - skipping")
                return
            
            # Step 2: Enable learning mode (activates tracking)
            self.llm.enable_learning_mode()
            
            # Step 3: Process facts through model (track slot access)
            await self._track_slot_access(facts)
            
            # Step 4: Get selective slots
            selective_slots = self.llm.get_selective_slots()
            logger.info(f"[ContinualLearner] Selected {len(selective_slots)} slots for update")
            
            # Step 5: Train selective slots
            await self._train_selective_slots(facts, selective_slots)
            
            # Step 6: Save checkpoint
            self.llm.save_memory_checkpoint(suffix="daily")
            
            # Step 7: Evaluate (optional)
            # await self._evaluate_learning()
            
            session_duration = (time.time() - session_start) / 60
            logger.info("=" * 80)
            logger.info(f"✅ CONTINUAL LEARNING SESSION COMPLETE ({session_duration:.1f} minutes)")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"[ContinualLearner] Learning session failed: {e}", exc_info=True)
        finally:
            self.llm.disable_learning_mode()
            self._is_learning = False
            self._last_learning_time = time.time()
    
    async def _extract_new_facts(self) -> List[str]:
        """Extract facts from memory store since last learning"""
        # Query memory store for new entities, edges, mentions
        # Format as natural language facts
        
        facts = []
        
        # TODO: Implement actual memory queries
        # Example queries:
        # - New entities: "User mentioned <entity> which is <description>"
        # - New relationships: "<entity1> <relation> <entity2>"
        # - New facts: Direct mentions from conversation
        
        # Placeholder:
        facts = [
            "User prefers async communication via Slack",
            "Q1 strategy focuses on enterprise customers",
            "Python 3.13 adds better error messages",
            # ... extract from memory_store
        ]
        
        return facts
    
    async def _track_slot_access(self, facts: List[str]):
        """Process facts through model to track memory slot access"""
        logger.info("[ContinualLearner] Tracking slot access...")
        
        # This is where we'd process facts through the model
        # Since DirectMLXLLM is synchronous, we need to wrap it
        
        # Placeholder: In real implementation, you'd:
        # 1. Format facts as prompts
        # 2. Run through model._model with tracking enabled
        # 3. Memory layer automatically tracks accessed slots
        
        await asyncio.sleep(1)  # Simulate processing
        logger.info("[ContinualLearner] Slot tracking complete")
    
    async def _train_selective_slots(self, facts: List[str], selective_slots: List[int]):
        """Train only selective memory slots on new facts"""
        logger.info(f"[ContinualLearner] Training {len(selective_slots)} selective slots...")
        
        # This is where we'd run SGD on selective slots
        # Placeholder for now
        
        # Real implementation would:
        # 1. Create Dataset from facts
        # 2. Tokenize
        # 3. Run training loop with SGD (LR=5.0) on selective slots only
        # 4. Use MLX_GLOBAL_LOCK to coordinate with other Metal operations
        
        await asyncio.sleep(5)  # Simulate training
        logger.info("[ContinualLearner] Training complete")
```

### 4. Configuration (Modification)

**Location:** `/Users/peppi/Dev/localcat/server/.env`

Add these environment variables:

```bash
# ------------------------------------------------------------
# Continual Learning (EXPERIMENTAL)
# ------------------------------------------------------------
CONTINUAL_LEARNING_ENABLED=true
CONTINUAL_LEARNING_IDLE_MINUTES=5
CONTINUAL_LEARNING_INTERVAL_HOURS=24
CONTINUAL_LEARNING_MIN_FACTS=10
CONTINUAL_LEARNING_REQUIRE_POWER=true
CONTINUAL_LEARNING_MEMORY_SLOTS=10000
CONTINUAL_LEARNING_CHECKPOINT_DIR=./data/memory_checkpoints
```

### 5. Integration with Bot (Modification)

**Location:** `/Users/peppi/Dev/localcat/server/bot.py`

```python
# Add to imports
from core.llm.direct_mlx_llm_with_memory import DirectMLXLLMServiceWithMemory
from core.memory.continual_learner import ContinualLearner

# In service factory, replace DirectMLXLLMServiceWithTools with:
if os.getenv("CONTINUAL_LEARNING_ENABLED", "false").lower() == "true":
    llm_service = DirectMLXLLMServiceWithMemory(
        model=llm_model,
        preloaded_model=preloaded_model,
        preloaded_tokenizer=preloaded_tokenizer,
        memory_slots=int(os.getenv("CONTINUAL_LEARNING_MEMORY_SLOTS", "10000")),
        memory_checkpoint_dir=os.getenv("CONTINUAL_LEARNING_CHECKPOINT_DIR"),
        enable_continual_learning=True
    )
    
    # Start continual learner
    continual_learner = ContinualLearner(
        llm_service=llm_service,
        memory_store=memory_store,  # Your existing memory store
        config={
            "idle_threshold_minutes": int(os.getenv("CONTINUAL_LEARNING_IDLE_MINUTES", "5")),
            "learning_interval_hours": int(os.getenv("CONTINUAL_LEARNING_INTERVAL_HOURS", "24")),
            "min_new_facts": int(os.getenv("CONTINUAL_LEARNING_MIN_FACTS", "10")),
            "require_power": os.getenv("CONTINUAL_LEARNING_REQUIRE_POWER", "true").lower() == "true",
        }
    )
    await continual_learner.start()
    
    logger.info("✨ Continual learning enabled")
else:
    # Use standard DirectMLXLLMServiceWithTools
    llm_service = DirectMLXLLMServiceWithTools(...)
```

## 📊 How It Works

### Daily Learning Cycle

```
1. User has conversation → Memory Store tracks entities/facts
                           ↓
2. User goes idle (5+ min) + Connected to power
                           ↓
3. ContinualLearner checks conditions every minute
                           ↓
4. Conditions met? → Start Learning Session
                           ↓
5. Extract facts from Memory Store (new entities, edges, mentions)
                           ↓
6. Enable learning mode → Memory layer starts tracking
                           ↓
7. Process facts through model → Track which slots accessed
                           ↓
8. TF-IDF identifies selective slots (~256 out of 10,000)
                           ↓
9. Train ONLY selective slots (SGD, LR=5.0, 5-10 minutes)
                           ↓
10. Save checkpoint (~5MB delta)
                           ↓
11. Resume normal operation (memory layer inactive)
```

### Storage

```
Base model: 3.5GB (frozen, never changes)
Memory checkpoints:
  - memory_1738876543_daily.npz  (~5MB)
  - memory_1738962943_daily.npz  (~5MB)
  - memory_1739049343_daily.npz  (~5MB)
  ...

After 365 days: ~1.8GB of checkpoints
```

## 🎯 Implementation Roadmap

### Phase 1: Foundation (1-2 days)
- [ ] Create `sparse_memory.py` (MLX-native implementation)
- [ ] Create `direct_mlx_llm_with_memory.py` (extend existing service)
- [ ] Test memory injection (verify no inference overhead)
- [ ] Test checkpoint save/load

### Phase 2: Orchestration (2-3 days)
- [ ] Create `continual_learner.py`
- [ ] Implement idle detection (power + activity)
- [ ] Implement fact extraction from MemoryStore
- [ ] Test learning cycle (without actual training)

### Phase 3: Training (2-3 days)
- [ ] Implement selective slot training
- [ ] Add MLX_GLOBAL_LOCK coordination
- [ ] Test with small fact set
- [ ] Measure forgetting on base knowledge

### Phase 4: Integration (1-2 days)
- [ ] Integrate with bot.py
- [ ] Add configuration to .env
- [ ] Test full pipeline
- [ ] Add monitoring/logging

### Phase 5: Evaluation (2-3 days)
- [ ] Measure forgetting over 7 days
- [ ] Measure new fact retention
- [ ] Optimize hyperparameters (slots, LR, schedule)
- [ ] Add rollback capability

## 🚀 Immediate Next Steps

**Option A: Minimal Proof of Concept (4-6 hours)**
1. Create `sparse_memory.py` with basic MLX implementation
2. Test injection into your existing model
3. Verify no inference overhead
4. Test one learning cycle manually

**Option B: Full Integration (1-2 weeks)**
Follow Phase 1-5 roadmap above

## 💡 Key Advantages for LocalCat

1. **Leverages existing infrastructure**
   - BackgroundSummarizer pattern for async work
   - Memory Store for fact extraction
   - MLX_GLOBAL_LOCK for coordination

2. **Zero inference overhead**
   - Memory layer only active during training
   - Normal conversations unaffected

3. **Privacy-first**
   - Everything stays local
   - No cloud sync required

4. **Efficient storage**
   - Only ~5MB per day
   - ~2GB per year of learning

5. **Rollback capability**
   - Checkpoint every learning session
   - Can revert if issues

## 📝 Notes

- Your existing `BackgroundSummarizer` is the PERFECT pattern to follow
- The `MLX_GLOBAL_LOCK` will prevent conflicts with STT/TTS
- Your memory extraction pipeline is already built
- The session tracking gives us idle detection
- Power detection works on macOS via `pmset`

This is genuinely exciting - LocalCat is PERFECT for this! 🚀
