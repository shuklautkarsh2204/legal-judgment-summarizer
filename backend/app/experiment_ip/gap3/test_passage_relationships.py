"""
Gap 3: Passage Relationship Signal Experiment

Investigates whether relationships between legally functional passages can be detected using:
- Semantic similarity
- Document proximity / passage distance
- Legal-function information (actors, argument types)

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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.services.passage_builder import PassageBuilder
from backend.app.services.semantic_analyzer import SemanticAnalyzer
from backend.app.services.preprocessing import split_into_sentences

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
        return list(set(a.actor for a in self.overlapping_annotations))
    
    def get_argument_types(self):
        """Get unique argument types in overlapping annotations."""
        return list(set(a.argument_type for a in self.overlapping_annotations))
    
    def primary_actor(self):
        """Get dominant/primary actor if any."""
        actors = self.get_actors()
        return actors[0] if len(actors) == 1 else None
    
    def primary_argument_type(self):
        """Get dominant/primary argument type if any."""
        types = self.get_argument_types()
        return types[0] if len(types) == 1 else None


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
        self.passages_by_doc: Dict[str, List[PassageInfo]] = defaultdict(list)
        self.pairs: List[PassagePair] = []
        
        self.results = {}
    
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
        
        for doc_id, text in document_texts.items():
            if not text or not text.strip():
                continue
            
            # Build passages
            passages = self.passage_builder.build_passages(text, document_id=doc_id)
            
            # Encode passages
            encoding_result = self.semantic_analyzer.encode_passages(passages, batch_size=32)
            encoded_passages = encoding_result['passages']
            
            # Create PassageInfo objects with annotation overlaps
            for i, passage in enumerate(encoded_passages):
                # Find overlapping annotations
                overlapping = [
                    a for a in self.annotations
                    if a.document_id == doc_id
                    and not (a.end <= passage['original_position']['start_char']
                            or a.begin >= passage['original_position']['end_char'])
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
                actor_transition = (actors_i[0], actors_j[0])
            else:
                same_actor = None
                actor_transition = None
            
            # Argument type information
            types_i = passage_i.get_argument_types()
            types_j = passage_j.get_argument_types()
            
            if types_i and types_j:
                same_type = len(set(types_i) & set(types_j)) > 0
                type_transition = (types_i[0], types_j[0])
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
                'experiment': 'Gap 3 - Passage Relationship Signals',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'random_seed': self.random_seed,
            },
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
    Load or construct document texts from RMU:ECHR annotations.
    
    Since we don't have full original judgment texts for ECHR cases,
    we construct synthetic documents by concatenating annotation texts
    grouped by document_id. This preserves document structure for analysis.
    
    Args:
        annotation_path: Path to rmu_echr_annotations.json
        
    Returns:
        Dict mapping document_id to full text
    """
    document_texts = {}
    
    print("Loading annotations to construct document texts...")
    
    if not annotation_path.exists():
        print(f"WARNING: Annotation file not found at {annotation_path}")
        return {}
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Group annotations by document_id
    docs = defaultdict(list)
    for ann in annotations:
        docs[ann['document_id']].append(ann)
    
    # For each document, construct text by sorting annotations by character position
    # and concatenating them with separators
    for doc_id, doc_annotations in docs.items():
        # Sort by begin position
        doc_annotations_sorted = sorted(doc_annotations, key=lambda x: x['begin'])
        
        # Build synthetic document text with annotations and gaps
        # This preserves the order and positions as they appear in the original judgment
        text_parts = []
        last_end = 0
        
        for ann in doc_annotations_sorted:
            begin = ann['begin']
            end = ann['end']
            text = ann['text']
            
            # Add separator if there's a gap
            if begin > last_end and last_end > 0:
                gap_chars = begin - last_end
                text_parts.append(f" [... {gap_chars} chars omitted ...] ")
            
            text_parts.append(text)
            last_end = end
        
        full_text = "".join(text_parts)
        document_texts[doc_id] = full_text
    
    print(f"Constructed {len(document_texts)} synthetic documents from annotations")
    print(f"Average annotations per document: {len(annotations) / len(document_texts):.1f}")
    
    return document_texts


def main():
    """Main experiment runner."""
    print("="*70)
    print("RMU:ECHR GAP 3 — PASSAGE RELATIONSHIP SIGNAL EXPERIMENT")
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
    
    # Load/construct document texts from annotations
    print("\nLoading/constructing document texts from annotations...")
    document_texts = load_document_texts(annotation_path)
    
    if not document_texts:
        print("ERROR: No document texts could be loaded.")
        return
    
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
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer.generate_results_json(output_dir / "gap3_relationship_analysis.json")
        
        # Print summary
        analyzer.print_summary_report()
    else:
        print("\nWARNING: No passages were built. Cannot proceed with analysis.")
        print("Ensure document texts are available or modify load_document_texts().")


if __name__ == "__main__":
    main()
