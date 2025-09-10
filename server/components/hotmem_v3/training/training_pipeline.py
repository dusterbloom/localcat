"""
HotMem V3 Complete Training Pipeline
Integrates dataset preparation + streaming augmentation + training preparation

This script prepares the complete training pipeline for HotMem v3.0:
1. Load and prepare datasets per HOT MEM V3 recipe
2. Apply streaming augmentation for real-time extraction
3. Format for Qwen2.5 training
4. Save in multiple formats for different training environments
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datasets import Dataset
import random

# Import our modules
from .dataset_preparation import HotMemDatasetPreparer
from ..augmentation.streaming_augmentation import StreamingAugmentor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotMemTrainingPipeline:
    """Complete training pipeline for HotMem v3.0"""
    
    def __init__(self, cache_dir: str = "./dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset_preparer = HotMemDatasetPreparer(cache_dir)
        self.streaming_augmentor = StreamingAugmentor()
        
    def load_or_create_base_dataset(self) -> List[Dict[str, Any]]:
        """Load or create the base dataset"""
        cache_file = self.cache_dir / "hotmem_v3_base_dataset.json"
        
        if cache_file.exists():
            logger.info("Loading cached base dataset")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logger.info("Creating base dataset from scratch")
        base_dataset = self.dataset_preparer.create_training_mix()
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(base_dataset, f, indent=2)
        
        return base_dataset
    
    def apply_streaming_augmentation(self, base_dataset: List[Dict[str, Any]], 
                                   augmentation_factor: float = 2.0) -> List[Dict[str, Any]]:
        """Apply streaming augmentation to the base dataset"""
        cache_file = self.cache_dir / f"hotmem_v3_augmented_{augmentation_factor}x.json"
        
        if cache_file.exists():
            logger.info("Loading cached augmented dataset")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logger.info("Applying streaming augmentation")
        augmented_dataset = self.streaming_augmentor.augment_dataset(
            base_dataset, 
            augmentation_factor=augmentation_factor
        )
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(augmented_dataset, f, indent=2)
        
        return augmented_dataset
    
    def format_for_qwen_training(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format dataset for Qwen2.5 structured output training"""
        logger.info("Formatting dataset for Qwen2.5 training")
        
        formatted_examples = []
        
        for i, example in enumerate(dataset):
            # Create instruction
            if example.get('is_streaming', False):
                instruction = "Extract entities and relations from this partial speech. Output in JSON format."
            else:
                instruction = "Extract entities and relations from the text. Output in JSON format."
            
            # Create input text
            input_text = example['text']
            
            # Create structured output
            output = {
                "entities": example['entities'],
                "relations": example['relations'],
                "confidence": example.get('confidence', 0.9),
                "metadata": {
                    "domain": example.get('domain', 'general'),
                    "source": example.get('source', 'unknown'),
                    "is_streaming": example.get('is_streaming', False),
                    "is_complete": example.get('is_complete', True)
                }
            }
            
            # Add streaming-specific metadata
            if example.get('is_streaming', False):
                output["metadata"].update({
                    "streaming_type": example.get('streaming_type', 'unknown'),
                    "position": example.get('position', 0)
                })
            
            formatted_example = {
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(output, indent=2),
                "domain": example.get('domain', 'general'),
                "is_streaming": example.get('is_streaming', False),
                "difficulty": self._calculate_difficulty(example)
            }
            
            formatted_examples.append(formatted_example)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Formatted {i + 1} examples...")
        
        logger.info(f"Formatted {len(formatted_examples)} examples for training")
        return formatted_examples
    
    def _calculate_difficulty(self, example: Dict[str, Any]) -> str:
        """Calculate difficulty level for curriculum learning"""
        text_length = len(example['text'])
        num_entities = len(example['entities'])
        num_relations = len(example['relations'])
        
        # Simple difficulty scoring
        if example.get('is_streaming', False):
            if not example.get('is_complete', True):
                return 'hard'  # Partial streaming is hardest
            else:
                return 'medium'
        else:
            if text_length > 200 or num_relations > 5:
                return 'medium'
            else:
                return 'easy'
    
    def create_curriculum_splits(self, formatted_dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create curriculum learning splits"""
        logger.info("Creating curriculum learning splits")
        
        # Separate by difficulty
        easy_examples = [ex for ex in formatted_dataset if ex['difficulty'] == 'easy']
        medium_examples = [ex for ex in formatted_dataset if ex['difficulty'] == 'medium']
        hard_examples = [ex for ex in formatted_dataset if ex['difficulty'] == 'hard']
        
        logger.info(f"Difficulty distribution:")
        logger.info(f"  - Easy: {len(easy_examples)} examples")
        logger.info(f"  - Medium: {len(medium_examples)} examples")
        logger.info(f"  - Hard: {len(hard_examples)} examples")
        
        return {
            'easy': easy_examples,
            'medium': medium_examples,
            'hard': hard_examples,
            'all': formatted_dataset
        }
    
    def save_for_training(self, datasets: Dict[str, List[Dict[str, Any]]], 
                         output_prefix: str = "hotmem_v3_training"):
        """Save datasets in multiple formats for training"""
        logger.info("Saving datasets for training")
        
        # Save JSON format
        for split_name, dataset in datasets.items():
            # JSON format
            json_file = self.cache_dir / f"{output_prefix}_{split_name}.json"
            with open(json_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            # HuggingFace Dataset format
            hf_dataset = Dataset.from_list(dataset)
            hf_dir = self.cache_dir / f"{output_prefix}_{split_name}.hf"
            hf_dataset.save_to_disk(str(hf_dir))
            
            # Create training/validation splits (80/20)
            if len(dataset) > 1000:  # Only for larger datasets
                train_size = int(len(dataset) * 0.8)
                train_data = dataset[:train_size]
                val_data = dataset[train_size:]
                
                # Save splits
                train_json = self.cache_dir / f"{output_prefix}_{split_name}_train.json"
                val_json = self.cache_dir / f"{output_prefix}_{split_name}_val.json"
                
                with open(train_json, 'w') as f:
                    json.dump(train_data, f, indent=2)
                
                with open(val_json, 'w') as f:
                    json.dump(val_data, f, indent=2)
                
                # HuggingFace splits
                train_hf = Dataset.from_list(train_data)
                val_hf = Dataset.from_list(val_data)
                
                train_hf.save_to_disk(str(self.cache_dir / f"{output_prefix}_{split_name}_train.hf"))
                val_hf.save_to_disk(str(self.cache_dir / f"{output_prefix}_{split_name}_val.hf"))
        
        logger.info("All datasets saved successfully")
    
    def create_training_summary(self, datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create training summary statistics"""
        summary = {
            "total_examples": sum(len(dataset) for dataset in datasets.values()),
            "splits": {},
            "domain_distribution": {},
            "streaming_distribution": {},
            "difficulty_distribution": {}
        }
        
        # Analyze each split
        for split_name, dataset in datasets.items():
            split_stats = {
                "count": len(dataset),
                "domains": {},
                "streaming_ratio": 0,
                "difficulty": {"easy": 0, "medium": 0, "hard": 0}
            }
            
            streaming_count = 0
            for example in dataset:
                # Domain distribution
                domain = example.get('domain', 'unknown')
                split_stats["domains"][domain] = split_stats["domains"].get(domain, 0) + 1
                
                # Streaming ratio
                if example.get('is_streaming', False):
                    streaming_count += 1
                
                # Difficulty distribution
                difficulty = example.get('difficulty', 'easy')
                split_stats["difficulty"][difficulty] += 1
            
            split_stats["streaming_ratio"] = streaming_count / len(dataset) if dataset else 0
            summary["splits"][split_name] = split_stats
        
        # Calculate overall distributions
        all_examples = []
        for dataset in datasets.values():
            all_examples.extend(dataset)
        
        for example in all_examples:
            # Overall domain distribution
            domain = example.get('domain', 'unknown')
            summary["domain_distribution"][domain] = summary["domain_distribution"].get(domain, 0) + 1
            
            # Overall streaming distribution
            is_streaming = example.get('is_streaming', False)
            stream_key = "streaming" if is_streaming else "complete"
            summary["streaming_distribution"][stream_key] = summary["streaming_distribution"].get(stream_key, 0) + 1
            
            # Overall difficulty distribution
            difficulty = example.get('difficulty', 'easy')
            summary["difficulty_distribution"][difficulty] = summary["difficulty_distribution"].get(difficulty, 0) + 1
        
        # Convert to percentages
        for dist_type in ["domain_distribution", "streaming_distribution", "difficulty_distribution"]:
            total = sum(summary[dist_type].values())
            if total > 0:
                for key in summary[dist_type]:
                    summary[dist_type][key] = summary[dist_type][key] / total * 100
        
        return summary
    
    def run_full_pipeline(self, augmentation_factor: float = 2.0) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("Starting HotMem v3 training pipeline...")
        
        # Step 1: Load or create base dataset
        logger.info("Step 1: Creating base dataset")
        base_dataset = self.load_or_create_base_dataset()
        
        # Step 2: Apply streaming augmentation
        logger.info("Step 2: Applying streaming augmentation")
        augmented_dataset = self.apply_streaming_augmentation(base_dataset, augmentation_factor)
        
        # Step 3: Format for training
        logger.info("Step 3: Formatting for Qwen2.5 training")
        formatted_dataset = self.format_for_qwen_training(augmented_dataset)
        
        # Step 4: Create curriculum splits
        logger.info("Step 4: Creating curriculum learning splits")
        curriculum_datasets = self.create_curriculum_splits(formatted_dataset)
        
        # Step 5: Save for training
        logger.info("Step 5: Saving datasets for training")
        self.save_for_training(curriculum_datasets, "hotmem_v3_final")
        
        # Step 6: Create summary
        logger.info("Step 6: Creating training summary")
        summary = self.create_training_summary(curriculum_datasets)
        
        # Save summary
        summary_file = self.cache_dir / "hotmem_v3_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Pipeline completed successfully!")
        return summary

def main():
    """Main execution function"""
    pipeline = HotMemTrainingPipeline()
    
    # Run the full pipeline
    summary = pipeline.run_full_pipeline(augmentation_factor=2.0)
    
    # Print summary
    print("\n" + "="*50)
    print("HOT MEM V3 TRAINING PIPELINE SUMMARY")
    print("="*50)
    print(f"Total examples: {summary['total_examples']:,}")
    print(f"Augmentation ratio: {summary['total_examples'] / (summary['total_examples'] / 3):.1f}x")
    
    print("\nDomain Distribution:")
    for domain, pct in summary['domain_distribution'].items():
        print(f"  - {domain}: {pct:.1f}%")
    
    print("\nStreaming Distribution:")
    for stream_type, pct in summary['streaming_distribution'].items():
        print(f"  - {stream_type}: {pct:.1f}%")
    
    print("\nDifficulty Distribution:")
    for difficulty, pct in summary['difficulty_distribution'].items():
        print(f"  - {difficulty}: {pct:.1f}%")
    
    print("\nTraining files created:")
    cache_dir = Path("./dataset_cache")
    for file_path in cache_dir.glob("hotmem_v3_final_*.json"):
        print(f"  - {file_path.name}")

if __name__ == "__main__":
    main()