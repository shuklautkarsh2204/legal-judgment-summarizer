"""
Gap 3: Passage Relationship Signal Experiment

Investigates whether relationships between legally functional passages can be detected using:
- Semantic similarity
- Document proximity / passage distance
- Legal-function information (actors, argument types)

CRITICAL FIX:
- This version uses ORIGINAL DOCUMENT TEXTS from RMU:ECHR XMI files
- All character offsets refer to the ORIGINAL documents, NOT synthetic reconstructions
- No synthetic "[... X chars omitted ...]" text is used
- Annotation offsets are validated against the original document text

This is an exploratory experiment. Results are DESCRIPTIVE ONLY.
No final relationship classifier is built.
No unsupported causal claims are made.
"""

import json
import sys
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.services.semantic_analyzer import SemanticAnalyzer
from backend.app.services.preprocessing import split_into_sentences
from backend.app.services.passage_builder import PassageBuilder
from backend.app.experiment_ip.gap3.document_loader import DocumentLoader

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class Annotation:
    """RMU:ECHR annotation record."""
    document_id: str
    text: str
    begin: int
    end: int
    actor: str
    argument_type: str


@dataclass
class PassageInfo:
    """Enhanced passage with annotation metadata."""
    document_id: str
    passage_index: int
    text: str
    begin_char: int
    end_char: int
    embedding: np.ndarray
    overlapping_annotations: List[Annotation]
    
    def get_actors(self):
        """Get unique actors in overlapping annotations."""
        return sorted({a.actor for a in self.overlapping_annotations})
    
    def get_argument_types(self):
        """Get unique argument types in overlapping annotations."""
        return sorted({a.argument_type for a in self.overlapping_annotations})
    
    def primary_actor(self):
        """Get the unique dominant actor, or None when the highest count ties."""
        counts = Counter(a.actor for a in self.overlapping_annotations)
        if not counts:
            return None
        highest_count = max(counts.values())
        dominant = [actor for actor, count in counts.items() if count == highest_count]
        return dominant[0] if len(dominant) == 1 else None
    
    def primary_argument_type(self):
        """Get the unique dominant argument type, or None when the highest count ties."""
        counts = Counter(a.argument_type for a in self.overlapping_annotations)
        if not counts:
            return None
        highest_count = max(counts.values())
        dominant = [argument_type for argument_type, count in counts.items()
                    if count == highest_count]
        return dominant[0] if len(dominant) == 1 else None


@dataclass
class PassagePair:
    """A pair of passages for relationship analysis."""
    doc_id: str
    passage_i: PassageInfo
    passage_j: PassageInfo
    pair_type: str  # 'local' or 'distant'
    
    # Computed features
    semantic_similarity: float
    passage_distance: int
    char_distance: int
    same_actor: Optional[bool]  # True, False, or None (Unknown)
    actor_transition: Optional[Tuple[str, str]]
    same_argument_type: Optional[bool]
    argument_type_transition: Optional[Tuple[str, str]]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'doc_id': self.doc_id,
            'passage_i': self.passage_i.passage_index,
            'passage_j': self.passage_j.passage_index,
            'pair_type': self.pair_type,
            'semantic_similarity': float(self.semantic_similarity),
            'passage_distance': self.passage_distance,
            'char_distance': self.char_distance,
            'same_actor': self.same_actor,
            'actor_transition': self.actor_transition,
            'same_argument_type': self.same_argument_type,
            'argument_type_transition': self.argument_type_transition,
        }


class Gap3Analyzer:
    """Main orchestrator for Gap 3 experiment."""
    
    def __init__(self, random_seed=RANDOM_SEED):
        self.random_seed = random_seed
        self.passage_builder = PassageBuilder(sentences_per_passage=5)
        self.semantic_analyzer = SemanticAnalyzer()
        
        self.annotations: List[Annotation] = []
        self.annotations_by_doc = defaultdict(list)
        self.passages_by_doc: Dict[str, List[PassageInfo]] = defaultdict(list)
        self.pairs: List[PassagePair] = []
        
        self.results = {}
        self.loader: Optional[DocumentLoader] = None
        self.document_texts: Dict[str, str] = {}
        self.validation_report = {}
    
    def validate_annotations_and_documents(
        self,
        document_texts: Dict[str, str],
        loader: DocumentLoader
    ):
        """
        Validate that annotations have correct offsets in original documents.
        
        Args:
            document_texts: Dict of document_id -> text
            loader: DocumentLoader instance
        """
        print("\n" + "="*70)
        print("VALIDATION: ANNOTATIONS AND ORIGINAL DOCUMENTS")
        print("="*70)
        
        self.loader = loader
        self.document_texts = document_texts
        
        # Overall statistics
        total_annotations = len(self.annotations)
        docs_with_text = len(document_texts)
        docs_with_annotations = len(set(a.document_id for a in self.annotations))
        
        print(f"\nDocuments with original text: {docs_with_text}")
        print(f"Documents referenced in annotations: {docs_with_annotations}")
        
        # Validate every annotation against the original document coordinate system.
        valid_count = 0
        invalid_count = 0
        text_mismatch_count = 0
        invalid_offset_details = []
        text_mismatch_details = []

        print(f"\nValidating FULL annotation set ({total_annotations} annotations)...")

        for ann in self.annotations:
            doc_text = document_texts.get(ann.document_id)
            if doc_text is None:
                invalid_count += 1
                invalid_offset_details.append({
                    'document_id': ann.document_id,
                    'begin': ann.begin,
                    'end': ann.end,
                    'reason': 'Document not found',
                })
                continue
            
            # Validate offset
            is_valid, msg = loader.validate_annotation_offset(
                ann.document_id,
                ann.begin,
                ann.end,
                ann.text
            )
            
            if is_valid:
                valid_count += 1
                if "mismatch" in msg.lower():
                    text_mismatch_count += 1
                    text_mismatch_details.append({
                        'document_id': ann.document_id,
                        'begin': ann.begin,
                        'end': ann.end,
                        'message': msg,
                    })
            else:
                invalid_count += 1
                invalid_offset_details.append({
                    'document_id': ann.document_id,
                    'begin': ann.begin,
                    'end': ann.end,
                    'reason': msg,
                })
                print(f"  INVALID: {ann.document_id} [{ann.begin}:{ann.end}]: {msg}")
        
        self.validation_report = {
            'total_annotations': total_annotations,
            'docs_with_original_text': docs_with_text,
            'docs_referenced': docs_with_annotations,
            'validation_scope': 'FULL annotation set',
            'validated_annotations': total_annotations,
            'valid_offsets': valid_count,
            'invalid_offsets': invalid_count,
            'text_mismatches': text_mismatch_count,
            'invalid_offset_details': invalid_offset_details,
            'text_mismatch_details': text_mismatch_details,
            'synthetic_reconstruction_used': False,
        }
        
        print(f"\nValidation Results (FULL annotation set of {total_annotations}):")
        print(f"  Valid offsets: {valid_count}")
        print(f"  Invalid offsets: {invalid_count}")
        print(f"  Text mismatches: {text_mismatch_count}")
        
        if invalid_count > 0:
            print(f"\nWARNING: Found {invalid_count} invalid annotations in full set")
            print("Some annotations may have incorrect offsets in original documents")
        
        if text_mismatch_count > 0:
            print(f"\nNOTE: Found {text_mismatch_count} text mismatches")
            print("This may indicate whitespace normalization or minor formatting differences")
        
        return valid_count > 0
    
    def load_annotations(self, annotation_path):
        """Load RMU:ECHR annotations."""
        print(f"Loading annotations from {annotation_path}...")
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = [
            Annotation(
                document_id=r['document_id'],
                text=r['text'],
                begin=r['begin'],
                end=r['end'],
                actor=r['actor'],
                argument_type=r['argument_type']
            )
            for r in data
        ]
        self.annotations_by_doc = defaultdict(list)
        for annotation in self.annotations:
            self.annotations_by_doc[annotation.document_id].append(annotation)
        
        print(f"Loaded {len(self.annotations)} annotations")
        
        # Summary statistics
        unique_docs = set(a.document_id for a in self.annotations)
        actors = set(a.actor for a in self.annotations)
        arg_types = set(a.argument_type for a in self.annotations)
        
        print(f"Unique documents: {len(unique_docs)}")
        print(f"Unique actors: {len(actors)} — {sorted(actors)}")
        print(f"Unique argument types: {len(arg_types)}")
        
        return unique_docs
    
    def build_passages_for_docs(self, document_texts: Dict[str, str]):
        """
        Build passages for documents.
        
        Args:
            document_texts: Dict mapping document_id to full text
        """
        print(f"\nBuilding passages for {len(document_texts)} documents...")
        
        total_documents = len(document_texts)
        for document_index, (doc_id, text) in enumerate(document_texts.items(), start=1):
            document_start = time.perf_counter()
            if not text or not text.strip():
                print(
                    f"[{document_index}/{total_documents}] {doc_id} | passages: 0 "
                    f"| encoding_time: 0.00s | total_time: {time.perf_counter() - document_start:.2f}s"
                )
                continue
            
            # Build passages
            construction_start = time.perf_counter()
            passages = self.passage_builder.build_passages(text, document_id=doc_id)
            construction_time = time.perf_counter() - construction_start
            
            # Encode passages
            encoding_start = time.perf_counter()
            encoding_result = self.semantic_analyzer.encode_passages(passages, batch_size=32)
            encoded_passages = encoding_result['passages']
            encoding_time = time.perf_counter() - encoding_start
            
            # Create PassageInfo objects with annotation overlaps
            overlap_start = time.perf_counter()
            document_annotations = self.annotations_by_doc.get(doc_id, [])
            for i, passage in enumerate(encoded_passages):
                # Find overlapping annotations
                overlapping = [
                    annotation for annotation in document_annotations
                    if not (
                        annotation.end <= passage['original_position']['start_char']
                        or annotation.begin >= passage['original_position']['end_char']
                    )
                ]
                
                passage_info = PassageInfo(
                    document_id=doc_id,
                    passage_index=i,
                    text=passage['text'],
                    begin_char=passage['original_position']['start_char'],
                    end_char=passage['original_position']['end_char'],
                    embedding=passage['embedding'],
                    overlapping_annotations=overlapping
                )
                
                self.passages_by_doc[doc_id].append(passage_info)
            overlap_time = time.perf_counter() - overlap_start
            total_time = time.perf_counter() - document_start
            print(
                f"[{document_index}/{total_documents}] {doc_id} | passages: {len(encoded_passages)} "
                f"| construction_time: {construction_time:.2f}s | encoding_time: {encoding_time:.2f}s "
                f"| overlap_time: {overlap_time:.2f}s | total_time: {total_time:.2f}s"
            )
        
        total_passages = sum(len(p) for p in self.passages_by_doc.values())
        docs_with_passages = len(self.passages_by_doc)
        print(f"Built passages for {docs_with_passages} documents ({total_passages} total passages)")
    
    def construct_candidate_pairs(self, local_window=10, distant_sampling_rate=0.1):
        """
        Construct local and distant passage pairs.
        
        Args:
            local_window: Consider passages within ±N positions as local candidates
            distant_sampling_rate: Fraction of distant pairs to sample
        """
        print(f"\nConstructing candidate pairs (local_window={local_window})...")
        
        pairs = []
        
        for doc_id, passages in self.passages_by_doc.items():
            n = len(passages)
            
            if n < 2:
                continue
            
            # Local candidate pairs: nearby passages
            for i in range(n):
                for j in range(i+1, min(i + local_window, n)):
                    pairs.append((doc_id, passages[i], passages[j], 'local'))
            
            # Distant baseline pairs: far passages
            # Sample from pairs where distance >= local_window
            potential_distant = [
                (doc_id, passages[i], passages[j], 'distant')
                for i in range(n)
                for j in range(i + local_window, n)
            ]
            
            if potential_distant:
                num_distant = max(1, int(len(potential_distant) * distant_sampling_rate))
                distant_sample = random.sample(potential_distant, min(num_distant, len(potential_distant)))
                pairs.extend(distant_sample)
        
        self.pairs = pairs
        local_count = sum(1 for _, _, _, t in pairs if t == 'local')
        distant_count = sum(1 for _, _, _, t in pairs if t == 'distant')
        
        print(f"Constructed {len(pairs)} pairs:")
        print(f"  - Local candidates: {local_count}")
        print(f"  - Distant baseline: {distant_count}")
    
    def compute_pair_features(self):
        """Compute features for all pairs."""
        print(f"\nComputing pair features for {len(self.pairs)} pairs...")
        
        passage_pairs = []
        
        for doc_id, passage_i, passage_j, pair_type in self.pairs:
            # Semantic similarity
            sim = float(np.dot(passage_i.embedding, passage_j.embedding) / 
                       (np.linalg.norm(passage_i.embedding) * np.linalg.norm(passage_j.embedding)))
            
            # Passage distance
            passage_dist = abs(passage_j.passage_index - passage_i.passage_index)
            
            # Character distance
            char_dist = passage_j.begin_char - passage_i.end_char
            if char_dist < 0:
                char_dist = passage_i.begin_char - passage_j.end_char
            
            # Actor information
            actors_i = passage_i.get_actors()
            actors_j = passage_j.get_actors()
            
            if actors_i and actors_j:
                same_actor = len(set(actors_i) & set(actors_j)) > 0
                actor_transition = (
                    passage_i.primary_actor(),
                    passage_j.primary_actor(),
                )
            else:
                same_actor = None
                actor_transition = None
            
            # Argument type information
            types_i = passage_i.get_argument_types()
            types_j = passage_j.get_argument_types()
            
            if types_i and types_j:
                same_type = len(set(types_i) & set(types_j)) > 0
                type_transition = (
                    passage_i.primary_argument_type(),
                    passage_j.primary_argument_type(),
                )
            else:
                same_type = None
                type_transition = None
            
            pair = PassagePair(
                doc_id=doc_id,
                passage_i=passage_i,
                passage_j=passage_j,
                pair_type=pair_type,
                semantic_similarity=sim,
                passage_distance=passage_dist,
                char_distance=char_dist,
                same_actor=same_actor,
                actor_transition=actor_transition,
                same_argument_type=same_type,
                argument_type_transition=type_transition,
            )
            
            passage_pairs.append(pair)
        
        self.passage_pairs = passage_pairs
        print(f"Computed features for {len(passage_pairs)} pairs")
    
    def analyze_semantic_similarity(self):
        """Analyze semantic similarity distribution by passage distance."""
        print("\n" + "="*70)
        print("SEMANTIC SIMILARITY ANALYSIS")
        print("="*70)
        
        # Group by distance
        distance_groups = defaultdict(list)
        for pair in self.passage_pairs:
            if pair.pair_type == 'local':
                dist = pair.passage_distance
                distance_groups[dist].append(pair.semantic_similarity)
        
        # Also create a 'distant' category
        distant_sims = [p.semantic_similarity for p in self.passage_pairs if p.pair_type == 'distant']
        
        analysis = {}
        
        for dist in sorted(distance_groups.keys()):
            sims = distance_groups[dist]
            analysis[f'distance_{dist}'] = {
                'count': len(sims),
                'mean': float(np.mean(sims)),
                'median': float(np.median(sims)),
                'std': float(np.std(sims)),
                'min': float(np.min(sims)),
                'max': float(np.max(sims)),
                'p25': float(np.percentile(sims, 25)),
                'p75': float(np.percentile(sims, 75)),
            }
        
        if distant_sims:
            analysis['distant_baseline'] = {
                'count': len(distant_sims),
                'mean': float(np.mean(distant_sims)),
                'median': float(np.median(distant_sims)),
                'std': float(np.std(distant_sims)),
                'min': float(np.min(distant_sims)),
                'max': float(np.max(distant_sims)),
                'p25': float(np.percentile(distant_sims, 25)),
                'p75': float(np.percentile(distant_sims, 75)),
            }
        
        self.results['semantic_similarity_analysis'] = analysis
        
        # Print summary
        for key, stats in sorted(analysis.items()):
            print(f"\n{key.upper()}:")
            print(f"  Pairs: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
    
    def compare_local_vs_distant(self):
        """Compare local candidates vs distant baseline."""
        print("\n" + "="*70)
        print("LOCAL vs DISTANT COMPARISON")
        print("="*70)
        
        local_sims = [p.semantic_similarity for p in self.passage_pairs if p.pair_type == 'local']
        distant_sims = [p.semantic_similarity for p in self.passage_pairs if p.pair_type == 'distant']
        
        if not local_sims or not distant_sims:
            print("Insufficient data for comparison")
            return
        
        local_mean = np.mean(local_sims)
        distant_mean = np.mean(distant_sims)
        diff = local_mean - distant_mean
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(local_sims)**2 + np.std(distant_sims)**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        comparison = {
            'local_mean': float(local_mean),
            'local_median': float(np.median(local_sims)),
            'local_count': len(local_sims),
            'distant_mean': float(distant_mean),
            'distant_median': float(np.median(distant_sims)),
            'distant_count': len(distant_sims),
            'mean_difference': float(diff),
            'effect_size_cohens_d': float(cohens_d),
            'interpretation': self._interpret_effect_size(cohens_d),
        }
        
        self.results['local_vs_distant'] = comparison
        
        print(f"\nLocal pairs:")
        print(f"  Mean similarity: {local_mean:.4f}")
        print(f"  Median similarity: {np.median(local_sims):.4f}")
        print(f"  Count: {len(local_sims)}")
        
        print(f"\nDistant pairs:")
        print(f"  Mean similarity: {distant_mean:.4f}")
        print(f"  Median similarity: {np.median(distant_sims):.4f}")
        print(f"  Count: {len(distant_sims)}")
        
        print(f"\nDifference:")
        print(f"  Mean difference: {diff:.4f}")
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
        print(f"  Interpretation: {comparison['interpretation']}")
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def analyze_actor_transitions(self):
        """Analyze actor transition patterns."""
        print("\n" + "="*70)
        print("ACTOR TRANSITION ANALYSIS")
        print("="*70)
        
        transitions = defaultdict(int)
        
        for pair in self.passage_pairs:
            if pair.actor_transition:
                transitions[pair.actor_transition] += 1
        
        # Sort by frequency
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        transition_data = {
            'total_transitions': len(sorted_transitions),
            'transitions': [
                {
                    'from': src,
                    'to': dst,
                    'count': count,
                    'percentage': round(count / sum(transitions.values()) * 100, 2)
                }
                for (src, dst), count in sorted_transitions
            ]
        }
        
        self.results['actor_transitions'] = transition_data
        
        print(f"\nTotal unique transitions: {len(sorted_transitions)}")
        print("\nTop 15 transitions:")
        for (src, dst), count in sorted_transitions[:15]:
            pct = count / sum(transitions.values()) * 100
            print(f"  {src} → {dst}: {count} ({pct:.1f}%)")
    
    def analyze_argument_type_transitions(self):
        """Analyze argument-type transition patterns."""
        print("\n" + "="*70)
        print("ARGUMENT-TYPE TRANSITION ANALYSIS")
        print("="*70)
        
        transitions = defaultdict(int)
        
        for pair in self.passage_pairs:
            if pair.argument_type_transition:
                transitions[pair.argument_type_transition] += 1
        
        # Sort by frequency
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        transition_data = {
            'total_transitions': len(sorted_transitions),
            'transitions': [
                {
                    'from': src,
                    'to': dst,
                    'count': count,
                    'percentage': round(count / sum(transitions.values()) * 100, 2)
                }
                for (src, dst), count in sorted_transitions
            ]
        }
        
        self.results['argument_type_transitions'] = transition_data
        
        print(f"\nTotal unique transitions: {len(sorted_transitions)}")
        print("\nTop 15 transitions:")
        for (src, dst), count in sorted_transitions[:15]:
            pct = count / sum(transitions.values()) * 100
            print(f"  {src} → {dst}: {count} ({pct:.1f}%)")
    
    def combined_signal_analysis(self):
        """
        Investigate combined signals from multiple features.
        
        Compute an exploratory "structural affinity score" that combines:
        - Semantic similarity (normalized)
        - Inverse passage distance (normalized)
        - Actor compatibility (binary, when available)
        - Argument-type compatibility (binary, when available)
        """
        print("\n" + "="*70)
        print("COMBINED STRUCTURAL SIGNAL ANALYSIS")
        print("="*70)
        
        scores = []
        
        for pair in self.passage_pairs:
            # Normalize similarity (already 0-1 range approximately)
            sim_score = pair.semantic_similarity
            
            # Inverse distance score (closer = higher)
            # Normalize to 0-1 range: 1 / (1 + distance)
            dist_score = 1.0 / (1.0 + pair.passage_distance)
            
            # Actor compatibility (0 if unknown, 1 if same, -0.5 if different)
            actor_score = 0
            if pair.same_actor is not None:
                actor_score = 1.0 if pair.same_actor else -0.5
            
            # Argument-type compatibility
            type_score = 0
            if pair.same_argument_type is not None:
                type_score = 1.0 if pair.same_argument_type else -0.5
            
            # Combined score (equal weights for now)
            # Formula: average of non-zero components
            components = [sim_score, dist_score]
            if actor_score != 0:
                components.append(actor_score)
            if type_score != 0:
                components.append(type_score)
            
            combined_score = np.mean(components)
            
            scores.append({
                'pair_index': len(scores),
                'doc_id': pair.doc_id,
                'passage_i': pair.passage_i.passage_index,
                'passage_j': pair.passage_j.passage_index,
                'pair_type': pair.pair_type,
                'semantic_similarity': float(sim_score),
                'distance_score': float(dist_score),
                'actor_score': float(actor_score),
                'type_score': float(type_score),
                'combined_score': float(combined_score),
            })
        
        # Analyze by pair type
        local_scores = [s['combined_score'] for s in scores if s['pair_type'] == 'local']
        distant_scores = [s['combined_score'] for s in scores if s['pair_type'] == 'distant']
        
        analysis = {
            'methodology': {
                'formula': 'average of (semantic_similarity, distance_score, actor_score, type_score)',
                'components': {
                    'semantic_similarity': 'direct (0-1)',
                    'distance_score': '1/(1+passage_distance)',
                    'actor_score': '1 if same else -0.5 (0 if unknown)',
                    'type_score': '1 if same else -0.5 (0 if unknown)',
                }
            },
            'local_candidates': {
                'count': len(local_scores),
                'mean': float(np.mean(local_scores)) if local_scores else 0,
                'median': float(np.median(local_scores)) if local_scores else 0,
                'std': float(np.std(local_scores)) if local_scores else 0,
            },
            'distant_baseline': {
                'count': len(distant_scores),
                'mean': float(np.mean(distant_scores)) if distant_scores else 0,
                'median': float(np.median(distant_scores)) if distant_scores else 0,
                'std': float(np.std(distant_scores)) if distant_scores else 0,
            },
            'sample_high_scores': [
                s for s in sorted(scores, key=lambda x: x['combined_score'], reverse=True)[:10]
            ]
        }
        
        self.results['combined_signal_analysis'] = analysis
        
        print(f"\nLocal candidates (n={len(local_scores)}):")
        if local_scores:
            print(f"  Mean affinity: {np.mean(local_scores):.4f}")
            print(f"  Median affinity: {np.median(local_scores):.4f}")
        
        print(f"\nDistant baseline (n={len(distant_scores)}):")
        if distant_scores:
            print(f"  Mean affinity: {np.mean(distant_scores):.4f}")
            print(f"  Median affinity: {np.median(distant_scores):.4f}")
        
        print("\nTop 10 highest combined affinity scores:")
        for i, score_info in enumerate(analysis['sample_high_scores'][:10], 1):
            print(f"  {i}. Passage {score_info['passage_i']}→{score_info['passage_j']}: "
                  f"{score_info['combined_score']:.4f} ({score_info['pair_type']})")
    
    def generate_results_json(self, output_path):
        """Save comprehensive results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'metadata': {
                'experiment': 'Gap 3 - Passage Relationship Signals (CORRECTED with Original Documents)',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'random_seed': self.random_seed,
                'coordinate_system': 'Original RMU:ECHR document character offsets (NOT synthetic)',
                'synthetic_reconstruction': False,
            },
            'validation': self.validation_report,
            'dataset_statistics': {
                'total_annotations': len(self.annotations),
                'unique_documents': len(self.passages_by_doc),
                'total_passages': sum(len(p) for p in self.passages_by_doc.values()),
                'unique_actors': len(set(a.actor for a in self.annotations)),
                'unique_argument_types': len(set(a.argument_type for a in self.annotations)),
            },
            'pair_statistics': {
                'total_pairs': len(self.passage_pairs),
                'local_candidate_pairs': sum(1 for p in self.passage_pairs if p.pair_type == 'local'),
                'distant_baseline_pairs': sum(1 for p in self.passage_pairs if p.pair_type == 'distant'),
            },
            **self.results,
            'limitations': [
                'RMU:ECHR annotations are not gold-standard passage relationship labels.',
                'Actor/type transitions are OBSERVED STRUCTURAL PATTERNS, not causal relationships.',
                'Semantic similarity is computed on passage text only, not on contextual legal function.',
                'Passages overlap annotations imperfectly; multiple labels per passage are possible.',
                'No external ground-truth relationship labels are used or inferred.',
                'This is an exploratory analysis to assess feasibility of passage-level relationship graphs.',
            ],
            'corrections_from_v1': [
                'NOW USES ORIGINAL RMU:ECHR DOCUMENT TEXTS (loaded from XMI files)',
                'Character offsets refer to original documents, NOT synthetic reconstructions',
                'Removed "[... X chars omitted ...]" synthetic document markers',
                'Annotations mapped to passages using original document coordinates',
                'All passage distances calculated from original document structure',
            ],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")
    
    def print_summary_report(self):
        """Print final experiment summary."""
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        print("\n1. DATASET STATISTICS")
        print(f"   Total annotations: {len(self.annotations)}")
        print(f"   Documents with passages: {len(self.passages_by_doc)}")
        print(f"   Total passages: {sum(len(p) for p in self.passages_by_doc.values())}")
        
        print("\n2. PAIR CONSTRUCTION")
        local_count = sum(1 for p in self.passage_pairs if p.pair_type == 'local')
        distant_count = sum(1 for p in self.passage_pairs if p.pair_type == 'distant')
        print(f"   Local candidate pairs: {local_count}")
        print(f"   Distant baseline pairs: {distant_count}")
        
        print("\n3. KEY FINDINGS")
        if 'local_vs_distant' in self.results:
            comparison = self.results['local_vs_distant']
            diff = comparison['mean_difference']
            sign = "higher" if diff > 0 else "lower"
            print(f"   Local pairs have {sign} semantic similarity than distant pairs.")
            print(f"   Mean difference: {diff:.4f}")
            print(f"   Effect size: {comparison['effect_size_cohens_d']:.4f} "
                  f"({comparison['interpretation']})")
        
        print("\n4. INTERPRETATION")
        print("   This experiment examines whether multiple signals (semantic similarity,")
        print("   document proximity, actor/type transitions) provide evidence that a")
        print("   passage-level reasoning graph may be feasible.")
        print()
        print("   IMPORTANT: These are DESCRIPTIVE FINDINGS ONLY.")
        print("   No causal claims about legal relationships are made.")
        print("   Results should guide future research directions.")
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)


def load_document_texts(annotation_path: Path):
    """
    Load ORIGINAL document texts from RMU:ECHR XMI files.
    
    The RMU:ECHR dataset annotations contain character offsets (begin/end)
    that refer to positions in the ORIGINAL judgment documents.
    
    To use these offsets correctly, we must load the original document texts
    from the RMU:ECHR XMI files, not construct synthetic documents.
    
    Args:
        annotation_path: Path to rmu_echr_annotations.json (used to identify documents)
        
    Returns:
        Dict mapping document_id to full original text
    """
    if not annotation_path.exists():
        print(f"ERROR: Annotation file not found at {annotation_path}")
        return {}
    
    print("\n" + "="*70)
    print("LOADING ORIGINAL DOCUMENT TEXTS FROM RMU:ECHR XMI FILES")
    print("="*70)
    
    # Load annotations to get list of document IDs
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Use DocumentLoader to load original texts
    loader = DocumentLoader()
    unique_doc_ids = set(ann['document_id'] for ann in annotations)
    
    print(f"\nLoading {len(unique_doc_ids)} unique documents from XMI files...")
    document_texts = loader.load_all_documents(unique_doc_ids)
    
    loader.print_load_report()
    
    if not document_texts:
        print("\nCRITICAL ERROR: No original document texts could be loaded.")
        print("Ensure RMU:ECHR gold_data directory is available at:")
        print(f"  {loader.gold_data_dir}")
        return {}
    
    print(f"\nSuccessfully loaded {len(document_texts)} document texts")
    print(f"Annotations reference {len(unique_doc_ids)} unique documents")
    print(f"Missing documents: {len(unique_doc_ids) - len(document_texts)}")
    
    return document_texts, loader, annotations


def main():
    """Main experiment runner."""
    print("="*70)
    print("RMU:ECHR GAP 3 — PASSAGE RELATIONSHIP SIGNAL EXPERIMENT")
    print("CORRECTED VERSION: Using Original Documents from RMU:ECHR XMI Files")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    annotation_path = project_root / "data" / "experiments" / "gap2" / "rmu_echr_annotations.json"
    output_dir = project_root / "data" / "experiments" / "gap3"
    
    if not annotation_path.exists():
        print(f"ERROR: Annotation file not found at {annotation_path}")
        return
    
    # Initialize analyzer
    analyzer = Gap3Analyzer(random_seed=RANDOM_SEED)
    
    # Load annotations
    unique_docs = analyzer.load_annotations(annotation_path)
    
    # Load ORIGINAL document texts from RMU:ECHR XMI files
    print("\nLoading original document texts from RMU:ECHR XMI files...")
    result = load_document_texts(annotation_path)
    
    if isinstance(result, tuple):
        document_texts, loader, annotations_from_json = result
    else:
        print("ERROR: Failed to load document texts")
        return
    
    if not document_texts:
        print("CRITICAL ERROR: No document texts could be loaded.")
        print("Cannot proceed without original RMU:ECHR documents.")
        return
    
    # Validate annotations against original documents
    validation_ok = analyzer.validate_annotations_and_documents(
        document_texts,
        loader
    )
    
    if not validation_ok:
        print("\nWARNING: Validation found some issues, but proceeding...")
    
    # Build passages with embeddings
    analyzer.build_passages_for_docs(document_texts)
    
    # Construct candidate pairs
    if analyzer.passages_by_doc:
        analyzer.construct_candidate_pairs(local_window=10, distant_sampling_rate=0.15)
        
        # Compute features
        analyzer.compute_pair_features()
        
        # Run analyses
        analyzer.analyze_semantic_similarity()
        analyzer.compare_local_vs_distant()
        analyzer.analyze_actor_transitions()
        analyzer.analyze_argument_type_transitions()
        analyzer.combined_signal_analysis()
        
        # Save results to new output file
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer.generate_results_json(
            output_dir / "gap3_real_document_relationship_results.json"
        )
        
        # Also save validation report separately
        with open(output_dir / "gap3_real_document_validation.json", 'w', encoding='utf-8') as f:
            json.dump(analyzer.validation_report, f, indent=2)
        
        # Print summary
        analyzer.print_summary_report()
    else:
        print("\nERROR: No passages were built. Cannot proceed with analysis.")
        print("Check that original documents were loaded successfully.")


if __name__ == "__main__":
    main()
