"""
Cloud Training Script for HotMem V3
Designed to run on Google Colab with free T4 GPU

Usage:
1. Upload this to Google Colab
2. Set runtime to GPU (T4)
3. Run all cells
4. Download trained model
"""

# Step 1: Install dependencies
print("🔧 Installing dependencies...")
!pip install -q unsloth datasets torch accelerate
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q git+https://github.com/huggingface/peft.git
!pip install -q bitsandbytes trl

# Step 2: Import libraries
import torch
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
import json
import os
from tqdm import tqdm

print("✅ Dependencies installed")

# Step 3: Load and prepare datasets
print("📚 Loading datasets...")

def load_and_prepare_datasets():
    """Load and prepare all datasets for training"""
    
    datasets_dict = {}
    
    # REBEL Dataset - 40% of training
    print("Loading REBEL dataset...")
    rebel = load_dataset("Babelscape/rebel-dataset", split="train[:40000]")
    datasets_dict['rebel'] = rebel
    
    # DialogRE Dataset - 25% of training  
    print("Loading DialogRE dataset...")
    dialogre = load_dataset("dialogre", split="train[:30000]")
    datasets_dict['dialogre'] = dialogre
    
    # ConvQuestions - 15% of training
    print("Loading ConvQuestions dataset...")
    convquestions = load_dataset("convquestions", split="train[:15000]")
    datasets_dict['convquestions'] = convquestions
    
    # Wizard of Wikipedia - 10% of training
    print("Loading Wizard of Wikipedia dataset...")
    wizard = load_dataset("wizard_of_wikipedia", split="train[:10000]")
    datasets_dict['wizard'] = wizard
    
    # MultiWOZ - 10% of training
    print("Loading MultiWOZ dataset...")
    multiwoz = load_dataset("multi_woz_v22", split="train[:10000]")
    datasets_dict['multiwoz'] = multiwoz
    
    return datasets_dict

# Step 4: Convert datasets to unified format
def convert_to_training_format(datasets_dict):
    """Convert all datasets to unified training format"""
    
    training_examples = []
    
    # Process REBEL dataset
    print("Converting REBEL dataset...")
    for item in tqdm(datasets_dict['rebel'][:40000]):
        triples = item.get('triples', [])
        text = item.get('text', '')
        
        if text and triples:
            # Convert REBEL triples to entities and relations
            entities = set()
            relations = []
            
            for triple in triples:
                if isinstance(triple, dict):
                    entities.add(triple.get('head', ''))
                    entities.add(triple.get('tail', ''))
                    relations.append({
                        'subject': triple.get('head', ''),
                        'predicate': triple.get('type', ''),
                        'object': triple.get('tail', '')
                    })
            
            training_examples.append({
                'text': text,
                'entities': list(entities),
                'relations': relations,
                'domain': 'general',
                'source': 'rebel'
            })
    
    # Process DialogRE dataset
    print("Converting DialogRE dataset...")
    for item in tqdm(datasets_dict['dialogre'][:30000]):
        dialogue = item.get('dialogue', [])
        relations = item.get('relations', [])
        
        if dialogue and relations:
            text = ' '.join(dialogue)
            entities = set()
            
            for rel in relations:
                if len(rel) >= 3:
                    entities.add(rel[0])  # subject
                    entities.add(rel[2])  # object
            
            training_examples.append({
                'text': text,
                'entities': list(entities),
                'relations': [{'subject': r[0], 'predicate': r[1], 'object': r[2]} for r in relations if len(r) >= 3],
                'domain': 'conversation',
                'source': 'dialogre'
            })
    
    # Process other datasets (simplified for demo)
    print("Converting other datasets...")
    
    return training_examples

# Step 5: Create streaming augmentation
def augment_for_streaming(examples):
    """Create partial sentence examples for streaming training"""
    
    augmented_examples = []
    
    for example in tqdm(examples[:10000]):  # Subset for demo
        text = example['text']
        words = text.split()
        
        # Create progressive examples
        for i in range(3, len(words) + 1):
            partial = ' '.join(words[:i])
            
            # Determine what's extractable so far
            partial_entities = [e for e in example['entities'] if e in partial]
            partial_relations = [r for r in example['relations'] 
                                if r['subject'] in partial and r['object'] in partial]
            
            augmented_examples.append({
                'text': partial,
                'entities': partial_entities,
                'relations': partial_relations,
                'is_partial': i < len(words),
                'domain': example['domain'],
                'source': example['source']
            })
    
    return augmented_examples

# Step 6: Format for Qwen2.5 training
def format_for_qwen(examples):
    """Format examples for Qwen2.5 structured output training"""
    
    formatted_examples = []
    
    for example in tqdm(examples):
        # Create training prompt
        prompt = f"""Extract entities and relations from the following text. Output in JSON format.

Text: {example['text']}

Output JSON:
"""
        
        # Create target output
        output = {
            "entities": example['entities'],
            "relations": example['relations'],
            "confidence": 0.9
        }
        
        formatted_examples.append({
            'instruction': 'Extract entities and relations as JSON',
            'input': example['text'],
            'output': json.dumps(output, indent=2),
            'domain': example['domain']
        })
    
    return formatted_examples

# Step 7: Main training function
def train_hotmem_v3():
    """Main training function for HotMem V3"""
    
    print("🚀 Starting HotMem V3 training...")
    
    # Load datasets
    datasets_dict = load_and_prepare_datasets()
    
    # Convert to training format
    training_examples = convert_to_training_format(datasets_dict)
    print(f"✅ Created {len(training_examples)} training examples")
    
    # Augment for streaming
    augmented_examples = augment_for_streaming(training_examples)
    print(f"✅ Created {len(augmented_examples)} augmented examples")
    
    # Combine examples
    all_examples = training_examples + augmented_examples
    print(f"✅ Total examples: {len(all_examples)}")
    
    # Format for Qwen2.5
    formatted_examples = format_for_qwen(all_examples)
    print(f"✅ Formatted {len(formatted_examples)} examples for training")
    
    # Save formatted data
    with open('hotmem_v3_training_data.json', 'w') as f:
        json.dump(formatted_examples, f, indent=2)
    
    print("💾 Saved training data to hotmem_v3_training_data.json")
    
    return formatted_examples

# Step 8: Setup Unsloth training
def setup_unsloth_training():
    """Setup Unsloth training environment"""
    
    print("🔧 Setting up Unsloth training...")
    
    # Load Qwen2.5-0.5B with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ Unsloth model ready")
    return model, tokenizer

# Step 9: Run training
def run_training(model, tokenizer, training_data):
    """Run the actual training"""
    
    print("🏃 Starting training...")
    
    # Convert to dataset format
    from datasets import Dataset
    
    # Use subset for demo (remove subset for full training)
    train_dataset = Dataset.from_list(training_data[:5000])
    
    # Setup training arguments
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,  # Increase for full training
            learning_rate=2e-4,
            fp16=torch.cuda.is_bf16_supported(),
            bf16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="hotmem_v3_outputs",
        ),
    )
    
    # Train the model
    trainer_stats = trainer.train()
    print("✅ Training completed!")
    
    return model, trainer_stats

# Step 10: Save and download
def save_and_download(model, tokenizer):
    """Save model for download"""
    
    print("💾 Saving model...")
    
    # Save to 16bit for inference
    model.save_pretrained_merged("hotmem_v3_qwen", tokenizer, save_method="merged_16bit")
    
    # Save to GGUF for llama.cpp
    model.save_pretrained_gguf("hotmem_v3", tokenizer, quantization_method="q4_k_m")
    
    print("✅ Model saved!")
    print("📥 Download the model files:")
    print("- hotmem_v3_qwen: For 16-bit inference")
    print("- hotmem_v3-*.gguf: For llama.cpp")
    
    # Create download links
    from google.colab import files
    !zip -r hotmem_v3_model.zip hotmem_v3_qwen/
    files.download('hotmem_v3_model.zip')

# Main execution
if __name__ == "__main__":
    print("🚀 HotMem V3 Cloud Training")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ No GPU available. Please enable GPU in Colab runtime settings.")
        exit()
    
    # Run training pipeline
    try:
        # Step 1: Prepare data
        training_data = train_hotmem_v3()
        
        # Step 2: Setup model
        model, tokenizer = setup_unsloth_training()
        
        # Step 3: Run training
        model, stats = run_training(model, tokenizer, training_data)
        
        # Step 4: Save and download
        save_and_download(model, tokenizer)
        
        print("\n🎉 Training completed successfully!")
        print("Next steps:")
        print("1. Download the model files")
        print("2. Copy to your Mac")
        print("3. Integrate with HotMem v3")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()