#!/usr/bin/env python3
"""
Detailed Quality Analysis - Examine actual relationships extracted
Focus on understanding why 38 entities only yield 15 relationships
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

class QualityAnalyzer:
    """Analyze the quality of extracted relationships"""
    
    def __init__(self):
        self.extractor = TieredRelationExtractor()
        
        # Test case with high entity count
        self.complex_text = "Dr. Sarah Chen, the AI research director at OpenAI who joined the company in 2021 after completing her PhD at Stanford under the supervision of Dr. Michael Jordan, recently published a groundbreaking paper on neural architecture search that builds upon her previous work on transformer optimization done during her internship at Google Brain in 2019 where she collaborated with Dr. Fei-Fei Li before moving to Stanford University to teach machine learning courses while maintaining her research position at OpenAI."
    
    def analyze_relationship_quality(self):
        """Analyze the actual quality of relationships extracted"""
        
        print("🔍 DETAILED RELATIONSHIP QUALITY ANALYSIS")
        print("=" * 80)
        print(f"Text: {self.complex_text}")
        print("-" * 80)
        
        # Warm up
        self.extractor._extract_tier1("Warm up.")
        self.extractor._extract_tier2("Warm up.")
        
        # Test Tier1
        print("\n🏗️  TIER1 ANALYSIS:")
        tier1_start = time.perf_counter()
        tier1_result = self.extractor._extract_tier1(self.complex_text)
        tier1_time = (time.perf_counter() - tier1_start) * 1000
        
        print(f"⏱️  Time: {tier1_time:.1f}ms")
        print(f"📊 Entities: {len(tier1_result.entities)}")
        print(f"🎯 Relationships: {len(tier1_result.relationships)}")
        
        print(f"\n📝 ENTITIES:")
        for i, entity in enumerate(tier1_result.entities[:20]):  # Show first 20
            print(f"   {i+1:2d}. {entity}")
        if len(tier1_result.entities) > 20:
            print(f"   ... and {len(tier1_result.entities) - 20} more")
        
        print(f"\n🔗 RELATIONSHIPS:")
        for i, rel in enumerate(tier1_result.relationships):
            print(f"   {i+1:2d}. {rel}")
        
        # Test Tier2
        print(f"\n🤖 TIER2 ANALYSIS:")
        tier2_start = time.perf_counter()
        tier2_result = self.extractor._extract_tier2(self.complex_text)
        tier2_time = (time.perf_counter() - tier2_start) * 1000
        
        print(f"⏱️  Time: {tier2_time:.1f}ms")
        print(f"📊 Entities: {len(tier2_result.entities)}")
        print(f"🎯 Relationships: {len(tier2_result.relationships)}")
        
        print(f"\n📝 ENTITIES:")
        for i, entity in enumerate(tier2_result.entities[:20]):  # Show first 20
            print(f"   {i+1:2d}. {entity}")
        if len(tier2_result.entities) > 20:
            print(f"   ... and {len(tier2_result.entities) - 20} more")
        
        print(f"\n🔗 RELATIONSHIPS:")
        for i, rel in enumerate(tier2_result.relationships):
            print(f"   {i+1:2d}. {rel}")
        
        # Analysis
        print(f"\n📈 QUALITY ANALYSIS:")
        print(f"   Entity count ratio: {len(tier1_result.entities)} vs {len(tier2_result.entities)}")
        print(f"   Relationship count ratio: {len(tier1_result.relationships)} vs {len(tier2_result.relationships)}")
        print(f"   Entity-to-relation ratio Tier1: {len(tier1_result.relationships) / max(len(tier1_result.entities), 1):.2f}")
        print(f"   Entity-to-relation ratio Tier2: {len(tier2_result.relationships) / max(len(tier2_result.entities), 1):.2f}")
        
        # Expected relationships analysis
        print(f"\n🎯 EXPECTED RELATIONSHIPS ANALYSIS:")
        expected_relationships = [
            "Sarah Chen works at OpenAI",
            "Sarah Chen joined OpenAI in 2021", 
            "Sarah Chen has PhD from Stanford",
            "Michael Jordan supervised Sarah Chen",
            "Sarah Chen published paper on neural architecture search",
            "Sarah Chen worked at Google Brain in 2019",
            "Sarah Chen collaborated with Fei-Fei Li",
            "Sarah Chen teaches at Stanford University",
            "Sarah Chen maintains research position at OpenAI",
            "Paper builds on transformer optimization work",
            "Work was done during internship",
            "Michael Jordan is at Stanford",
            "Fei-Fei Li is at Google Brain",
            "Stanford University offers machine learning courses"
        ]
        
        print(f"   Expected relationships: {len(expected_relationships)}")
        print(f"   Tier1 captured: {len(tier1_result.relationships)} ({len(tier1_result.relationships)/len(expected_relationships)*100:.1f}%)")
        print(f"   Tier2 captured: {len(tier2_result.relationships)} ({len(tier2_result.relationships)/len(expected_relationships)*100:.1f}%)")
        
        # Check for key relationships
        key_relationships = [
            "sarah chen works at openai",
            "sarah chen joined openai", 
            "sarah chen has phd from stanford",
            "michael jordan supervised sarah chen",
            "sarah chen worked at google brain",
            "sarah chen collaborated with fei-fei li"
        ]
        
        print(f"\n🔑 KEY RELATIONSHIP DETECTION:")
        tier1_key_count = 0
        tier2_key_count = 0
        
        for key_rel in key_relationships:
            # Convert relationships to strings for comparison
            tier1_rel_strings = [f"{rel[0]} {rel[1]} {rel[2]}" for rel in tier1_result.relationships]
            tier2_rel_strings = [f"{rel[0]} {rel[1]} {rel[2]}" for rel in tier2_result.relationships]
            
            tier1_has = any(key_rel in rel_str.lower() for rel_str in tier1_rel_strings)
            tier2_has = any(key_rel in rel_str.lower() for rel_str in tier2_rel_strings)
            
            print(f"   {key_rel}:")
            print(f"      Tier1: {'✅' if tier1_has else '❌'}")
            print(f"      Tier2: {'✅' if tier2_has else '❌'}")
            
            if tier1_has: tier1_key_count += 1
            if tier2_has: tier2_key_count += 1
        
        print(f"\n📊 KEY RELATIONSHIP SUMMARY:")
        print(f"   Tier1: {tier1_key_count}/{len(key_relationships)} key relationships ({tier1_key_count/len(key_relationships)*100:.1f}%)")
        print(f"   Tier2: {tier2_key_count}/{len(key_relationships)} key relationships ({tier2_key_count/len(key_relationships)*100:.1f}%)")
        
        return {
            'tier1': {
                'time': tier1_time,
                'entities': len(tier1_result.entities),
                'relationships': len(tier1_result.relationships),
                'entity_relation_ratio': len(tier1_result.relationships) / max(len(tier1_result.entities), 1),
                'key_relationships': tier1_key_count,
                'actual_entities': tier1_result.entities,
                'actual_relationships': tier1_result.relationships
            },
            'tier2': {
                'time': tier2_time,
                'entities': len(tier2_result.entities),
                'relationships': len(tier2_result.relationships),
                'entity_relation_ratio': len(tier2_result.relationships) / max(len(tier2_result.entities), 1),
                'key_relationships': tier2_key_count,
                'actual_entities': tier2_result.entities,
                'actual_relationships': tier2_result.relationships
            }
        }

if __name__ == "__main__":
    analyzer = QualityAnalyzer()
    results = analyzer.analyze_relationship_quality()