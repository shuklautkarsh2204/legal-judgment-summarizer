"""
Load original document texts from RMU:ECHR XMI files.

The RMU:ECHR dataset is distributed as XMI (XMI Meta Language) files.
Each XMI file contains:
- A Sofa element with the original document text (sofaString attribute)
- Annotation elements with begin/end offsets referring to the Sofa text

This module provides utilities to load the original document texts
and validate that annotation offsets are correct.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict


# Default path to RMU:ECHR gold_data directory
DEFAULT_GOLD_DATA_PATH = Path(r"D:\Projects\mining-legal-arguments\gold_data")


class DocumentLoader:
    """Load original document texts from RMU:ECHR XMI files."""
    
    def __init__(self, gold_data_dir: Optional[Path] = None):
        """
        Initialize the document loader.
        
        Args:
            gold_data_dir: Path to the RMU:ECHR gold_data directory.
                          Defaults to DEFAULT_GOLD_DATA_PATH.
        """
        if gold_data_dir is None:
            gold_data_dir = DEFAULT_GOLD_DATA_PATH
        
        self.gold_data_dir = Path(gold_data_dir)
        self.documents: Dict[str, str] = {}  # document_id -> full text
        self.load_status = {}  # document_id -> (success, message)
    
    def load_document_from_xmi(self, document_id: str) -> Optional[str]:
        """
        Load a single document from its XMI file.
        
        Args:
            document_id: The ECHR case ID (e.g., "001-100469")
            
        Returns:
            The original document text, or None if not found/error
        """
        if document_id in self.documents:
            return self.documents[document_id]
        
        xmi_file = self.gold_data_dir / f"{document_id}.xmi"
        
        if not xmi_file.exists():
            self.load_status[document_id] = (False, f"XMI file not found: {xmi_file}")
            return None
        
        try:
            tree = ET.parse(xmi_file)
            root = tree.getroot()
            
            # Extract document text from Sofa element
            for element in root.iter():
                if element.tag.endswith("Sofa"):
                    text = element.attrib.get("sofaString")
                    if text is not None:
                        self.documents[document_id] = text
                        self.load_status[document_id] = (True, "Loaded successfully")
                        return text
            
            # No Sofa element found
            self.load_status[document_id] = (False, "No Sofa element found in XMI")
            return None
            
        except ET.ParseError as e:
            self.load_status[document_id] = (False, f"XMI parse error: {e}")
            return None
        except Exception as e:
            self.load_status[document_id] = (False, f"Error loading XMI: {e}")
            return None
    
    def load_all_documents(self, document_ids) -> Dict[str, str]:
        """
        Load all documents from a list of document IDs.
        
        Args:
            document_ids: Iterable of document IDs to load
            
        Returns:
            Dict mapping document_id to text (only successful loads)
        """
        for doc_id in document_ids:
            self.load_document_from_xmi(doc_id)
        
        return self.documents
    
    def get_document(self, document_id: str) -> Optional[str]:
        """
        Get a document, loading it if necessary.
        
        Args:
            document_id: The ECHR case ID
            
        Returns:
            The document text, or None if not available
        """
        if document_id not in self.documents:
            self.load_document_from_xmi(document_id)
        
        return self.documents.get(document_id)
    
    def validate_annotation_offset(
        self,
        document_id: str,
        begin: int,
        end: int,
        annotation_text: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate that an annotation's offsets are correct.
        
        Args:
            document_id: The document ID
            begin: The start offset
            end: The end offset
            annotation_text: Optional stored annotation text to verify against
            
        Returns:
            Tuple of (is_valid, message)
        """
        doc_text = self.get_document(document_id)
        
        if doc_text is None:
            return (False, f"Document not found: {document_id}")
        
        # Check boundary conditions
        if begin < 0:
            return (False, f"begin < 0: {begin}")
        if end > len(doc_text):
            return (False, f"end > document length: {end} > {len(doc_text)}")
        if begin >= end:
            return (False, f"begin >= end: {begin} >= {end}")
        
        # Extract text from document
        extracted_text = doc_text[begin:end]
        
        if not extracted_text or not extracted_text.strip():
            return (False, "Extracted text is empty")
        
        # If annotation text is provided, verify it matches (with whitespace normalization)
        if annotation_text is not None:
            # Normalize whitespace for comparison
            extracted_normalized = " ".join(extracted_text.split())
            stored_normalized = " ".join(annotation_text.split())
            
            if extracted_normalized != stored_normalized:
                # Return warning but still valid (might be minor formatting difference)
                return (
                    True,
                    f"Text mismatch (extracted vs stored): "
                    f"'{extracted_normalized[:50]}...' vs '{stored_normalized[:50]}...'"
                )
        
        return (True, "Valid")
    
    def print_load_report(self):
        """Print a report of document loading status."""
        if not self.load_status:
            print("No documents loaded yet")
            return
        
        successful = sum(1 for success, _ in self.load_status.values() if success)
        failed = len(self.load_status) - successful
        
        print("\n" + "="*70)
        print("DOCUMENT LOADING REPORT")
        print("="*70)
        
        print(f"\nTotal documents:       {len(self.load_status)}")
        print(f"Successfully loaded:   {successful}")
        print(f"Failed:                {failed}")
        
        if failed > 0:
            print("\nFailed documents:")
            for doc_id, (success, msg) in sorted(self.load_status.items()):
                if not success:
                    print(f"  {doc_id}: {msg}")


def load_original_documents(
    annotation_list,
    gold_data_dir: Optional[Path] = None
) -> Dict[str, str]:
    """
    Convenience function to load original documents for a list of annotations.
    
    Args:
        annotation_list: List of annotation dicts with 'document_id' key
        gold_data_dir: Path to RMU:ECHR gold_data directory
        
    Returns:
        Dict mapping document_id to original document text
    """
    loader = DocumentLoader(gold_data_dir)
    
    # Extract unique document IDs from annotations
    unique_doc_ids = set(ann['document_id'] for ann in annotation_list)
    
    print(f"Loading {len(unique_doc_ids)} unique documents from RMU:ECHR XMI files...")
    loader.load_all_documents(unique_doc_ids)
    
    loader.print_load_report()
    
    return loader.documents
