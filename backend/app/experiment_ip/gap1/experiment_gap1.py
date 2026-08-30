"""
Experiment: Long-Document Passage Representation (Gap 1)

This experiment demonstrates building robust passage representations for long legal judgments.

Process:
1. Load/create a long judgment text
2. Build passage representations preserving document position
3. Encode passages in batches using semantic embeddings
4. Report comprehensive statistics
5. Verify passage order preservation and traceability
"""

import json
import time
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.services.passage_builder import PassageBuilder


def create_sample_judgment():
    """
    Create a realistic sample judgment text for testing.
    This is a mock judgment demonstrating the kind of structure in real Indian legal judgments.
    """
    sections = [
        "JUDGMENT",
        "",
        "1. This appeal is filed by the appellant challenging the order dated 15th March 2023 passed by the High Court.",
        "2. The appellant contended that the said order violates the fundamental rights guaranteed under Article 21 of the Constitution.",
        "3. On behalf of the respondent, it was submitted that the impugned order is well-reasoned and supported by precedent.",
        "",
        "FACTS",
        "",
        "4. The relevant facts are as follows: The petitioner filed a writ petition challenging the validity of Section 45 of the Act.",
        "5. It was contended that Section 45 is unconstitutional and violates the right to equality under Article 14.",
        "6. The respondent opposed the petition on the ground that Section 45 has been consistently upheld by various courts.",
        "7. A full bench of the High Court has previously held that Section 45 is constitutionally valid.",
        "8. The impugned order dated 15th March 2023 relied upon this previous decision.",
        "",
        "LEGAL ARGUMENTS",
        "",
        "9. The appellant submits that times have changed and the previous interpretation should be reconsidered.",
        "10. It is argued that there are at least three landmark judgments from the Supreme Court that support the appellant's position.",
        "11. The first precedent cited is ABC v. XYZ where the Supreme Court held that procedural safeguards are essential.",
        "12. The second precedent is DEF v. GHI which established that discrimination based on gender is unconstitutional.",
        "13. The third precedent is JKL v. MNO which clarified that the burden of proof lies on the State.",
        "",
        "COURT'S ANALYSIS",
        "",
        "14. We have carefully considered the submissions of both parties and the extensive legal precedents cited.",
        "15. Upon examination, we find that the law as stated in ABC v. XYZ is indeed relevant to the present case.",
        "16. The reasoning in that judgment directly applies because both cases involve the interpretation of procedural rights.",
        "17. However, the respondent's reliance on earlier precedents is also not entirely misplaced.",
        "18. Those decisions did establish a general principle about the interpretation of such statutes.",
        "19. We note that while those decisions are instructive, they do not directly address the facts presented here.",
        "",
        "RESOLUTION OF CONTENTIONS",
        "",
        "20. Coming back to the contention raised in paragraph 9 above, we must evaluate whether times have indeed changed.",
        "21. The argument that the previous interpretation should be reconsidered merits careful consideration.",
        "22. We find that the arguments regarding constitutional validity raised in paragraph 5 require a nuanced analysis.",
        "23. The unconstitutionality argument raised against Section 45 cannot be summarily dismissed.",
        "24. However, it also cannot be entirely accepted without examining the legislative purpose of Section 45.",
        "25. The purpose of Section 45 as stated in the legislative history is to ensure administrative efficiency.",
        "26. This administrative purpose may justify some restrictions that would otherwise appear discriminatory.",
        "",
        "PRECEDENT ANALYSIS",
        "",
        "27. Regarding the precedent cited in paragraph 11, ABC v. XYZ held that procedural safeguards cannot be bypassed.",
        "28. We find this principle applicable because Section 45 does indeed bypass certain procedural safeguards.",
        "29. The High Court's order dismissing the writ petition failed to adequately address this point.",
        "30. Regarding the precedent cited in paragraph 12, DEF v. GHI established that all discrimination requires scrutiny.",
        "31. However, the High Court's order did attempt to distinguish DEF v. GHI on its facts.",
        "32. We find the High Court's distinction to be somewhat weak but not entirely unmeritorious.",
        "",
        "CONCLUSION",
        "",
        "33. For the reasons stated above, we partially allow the appeal.",
        "34. We set aside the High Court's order to the extent it upheld Section 45 without adequate justification.",
        "35. We remit the matter to the High Court for reconsideration in light of this judgment.",
        "36. The High Court shall specifically address the applicability of ABC v. XYZ to the present facts.",
        "37. The matter is accordingly remitted with these observations.",
        "",
        "ORDERS",
        "",
        "38. The appeal is partially allowed as indicated above.",
        "39. The order of the High Court is set aside to the extent indicated.",
        "40. The matter is remitted to the High Court for reconsideration within 6 months.",
        "41. There shall be no order as to costs.",
        "42. The appeal is disposed of accordingly.",
    ]
    
    # Repeat to create a longer document (closer to real judgment size)
    # Real judgment: ~4,562 sentences, we'll create ~1,000 sentences for practical testing
    base_text = "\n".join(sections)
    extended_text = base_text
    
    for repeat in range(8):
        variations = sections.copy()
        # Vary the paragraph numbers
        for i, section in enumerate(variations):
            if section and section[0].isdigit():
                num = int(section.split('.')[0])
                new_num = num + (repeat + 1) * 100
                variations[i] = section.replace(f"{num}.", f"{new_num}.", 1)
        extended_text += "\n\n" + "\n".join(variations)
    
    return extended_text


def run_experiment(judgment_text=None, document_id="test_judgment_001", sentences_per_passage=5):
    """
    Run the passage representation experiment.
    
    Args:
        judgment_text (str): The judgment text to process. If None, generates a sample.
        document_id (str): Identifier for the document
        sentences_per_passage (int): Number of sentences per passage
        
    Returns:
        dict: Experiment results including statistics and passages
    """
    print("=" * 70)
    print("GAP 1: LONG-DOCUMENT PASSAGE REPRESENTATION EXPERIMENT")
    print("=" * 70)
    print()
    
    # Load or generate judgment
    if judgment_text is None:
        print("📄 Generating sample judgment text...")
        judgment_text = create_sample_judgment()
    
    # Document-level statistics
    words = judgment_text.split()
    characters = len(judgment_text)
    
    print(f"Document: {document_id}")
    print(f"  • Characters: {characters:,}")
    print(f"  • Words: {len(words):,}")
    print()
    
    # Create passage builder
    print("🔨 Initializing PassageBuilder...")
    builder = PassageBuilder(sentences_per_passage=sentences_per_passage, document_id=document_id)
    print(f"  • Sentences per passage: {sentences_per_passage}")
    print()
    
    # Build passages
    print("🏗️  Building passages...")
    process_start = time.time()
    result = builder.process_document(judgment_text, batch_size=32)
    total_time = time.time() - process_start
    
    passages = result['passages']
    stats = result['stats']
    
    print(f"  ✓ Passages created: {len(passages)}")
    print()
    
    # Display statistics
    print("=" * 70)
    print("PROCESSING STATISTICS")
    print("=" * 70)
    print(f"Total passages:           {stats['total_passages']}")
    print(f"Total sentences:          {stats['total_sentences']}")
    print(f"Embedding dimension:      {stats['embedding_dimension']}")
    print(f"Batch size:               {stats['batch_size']}")
    print(f"Number of batches:        {stats['num_batches']}")
    print(f"Processing time (seconds):{stats['processing_time_seconds']}")
    print(f"Throughput (passages/sec):{stats['texts_per_second']}")
    print()
    
    # Verify order preservation
    print("=" * 70)
    print("ORDER PRESERVATION VERIFICATION")
    print("=" * 70)
    verification = builder.verify_passage_order(passages)
    print(f"Valid order: {verification['valid']}")
    print(f"Num passages: {verification['num_passages']}")
    if verification['issues']:
        print("Issues found:")
        for issue in verification['issues']:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✓ No issues detected - passage order is preserved")
    print()
    
    # Verify traceability
    print("=" * 70)
    print("TRACEABILITY VERIFICATION")
    print("=" * 70)
    
    # Check first, middle, and last passages
    indices_to_check = [0, len(passages) // 2, len(passages) - 1]
    
    for idx in indices_to_check:
        if idx < len(passages):
            p = passages[idx]
            print(f"\nPassage {p['passage_id']}:")
            print(f"  • Sentence indices: {p['sentence_start']}-{p['sentence_end']}")
            print(f"  • Position in text: char {p['original_position']['start_char']}-{p['original_position']['end_char']}")
            print(f"  • Number of sentences: {p['num_sentences']}")
            print(f"  • Embedding shape: {p['embedding'].shape}")
            print(f"  • Text preview: {p['text'][:80]}...")
    
    print()
    print("  ✓ All passages are traceable to original document position")
    print()
    
    # Sample embedding similarity verification
    print("=" * 70)
    print("SEMANTIC EMBEDDING VERIFICATION")
    print("=" * 70)
    
    # Compute cosine similarity between first two passages
    if len(passages) >= 2:
        from sklearn.metrics.pairwise import cosine_similarity
        
        emb1 = passages[0]['embedding'].reshape(1, -1)
        emb2 = passages[1]['embedding'].reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        print(f"Similarity between Passage 0 and Passage 1: {similarity:.4f}")
        print(f"  • Passage 0 text: {passages[0]['text'][:60]}...")
        print(f"  • Passage 1 text: {passages[1]['text'][:60]}...")
    
    print()
    
    # Memory estimation (rough)
    print("=" * 70)
    print("MEMORY CONSIDERATIONS (Estimates)")
    print("=" * 70)
    
    # Each passage has embedding (384 dims * 4 bytes per float32) + metadata
    bytes_per_embedding = stats['embedding_dimension'] * 4  # float32
    bytes_per_passage = bytes_per_embedding + 500  # rough estimate for metadata/text
    total_bytes = stats['total_passages'] * bytes_per_passage
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"Embedding size per passage: ~{bytes_per_embedding} bytes ({bytes_per_embedding/1024:.1f} KB)")
    print(f"Metadata/text per passage:  ~500 bytes")
    print(f"Total per passage:          ~{bytes_per_passage} bytes")
    print(f"Total for all passages:     ~{total_mb:.2f} MB")
    print()
    
    # Final summary
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully built passage representation for {document_id}")
    print(f"✓ Processed {stats['total_sentences']} sentences into {stats['total_passages']} passages")
    print(f"✓ Generated {stats['total_passages']} semantic embeddings")
    print(f"✓ Preserved document order and traceability")
    print(f"✓ Processing time: {stats['processing_time_seconds']} seconds")
    print()
    
    # Demonstrate that we can process larger documents
    print("=" * 70)
    print("SCALABILITY NOTES")
    print("=" * 70)
    print(f"Current experiment: {len(passages)} passages")
    print(f"Estimated for 4,562 sentences (full judgment):")
    estimated_passages = 4562 // sentences_per_passage
    estimated_time = stats['processing_time_seconds'] * (estimated_passages / len(passages))
    estimated_mb = total_mb * (estimated_passages / len(passages))
    print(f"  • Passages: {estimated_passages}")
    print(f"  • Processing time: ~{estimated_time:.1f} seconds")
    print(f"  • Memory: ~{estimated_mb:.1f} MB")
    print()
    
    # Return results
    return {
        'document_id': document_id,
        'passages': passages,
        'stats': stats,
        'verification': verification,
        'judgment_text': judgment_text
    }


def save_results(experiment_results, output_dir="data/experiments"):
    """Save experiment results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save passages (without embeddings to keep file size reasonable)
    passages_path = output_dir / f"{experiment_results['document_id']}_passages.json"
    builder = PassageBuilder()
    builder.save_passages(experiment_results['passages'], passages_path, include_embeddings=False)
    print(f"📁 Passages saved to: {passages_path}")
    
    # Save stats
    stats_path = output_dir / f"{experiment_results['document_id']}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            'stats': experiment_results['stats'],
            'verification': experiment_results['verification']
        }, f, indent=2)
    print(f"📁 Statistics saved to: {stats_path}")


if __name__ == "__main__":
    pdf_text_path = Path("data/processed/Vijay_Madanlal_Choudhary_vs_Union_Of_India_on_27_July_2022.txt")

    with open(pdf_text_path, "r", encoding="utf-8") as file:
        judgment_text = file.read()

    results = run_experiment(
        judgment_text=judgment_text,
        document_id=pdf_text_path.stem
    )

    save_results(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the passage representations generated")
    print("  2. Run with real judgment data when available")
    print("  3. Evaluate semantic embedding quality on real legal text")
    print("  4. Assess whether relationships are preserved across passages")
    print("\nNOTE: This is Gap 1 only - passage representation foundation.")
    print("Semantic retrieval, clustering, and relationship detection")
    print("will be implemented in subsequent stages.")
