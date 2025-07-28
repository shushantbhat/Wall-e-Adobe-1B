#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence using Ollama
Local LLM-powered document analysis for extracting relevant sections
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import logging
import time

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedSection:
    document: str
    page_number: int
    section_title: str
    importance_rank: int

@dataclass
class SubsectionAnalysis:
    document: str
    section_title: str
    refined_text: str
    page_number: int

@dataclass
class AnalysisResult:
    metadata: Dict[str, Any]
    extracted_sections: List[ExtractedSection]
    subsection_analysis: List[SubsectionAnalysis]

class DocumentAnalyzer:
    def __init__(self, model_name: str = "phi3:mini"):
        """Initialize local Ollama model and embedding model."""
        try:
            # Initialize Ollama with timeout and retry logic
            self.llm = Ollama(
                model=model_name, 
                temperature=0.1,
                timeout=30,
                num_ctx=4096  # Context window
            )
            
            # Test Ollama connection
            logger.info("Testing Ollama connection...")
            test_response = self.llm.invoke("Hello")
            logger.info(f"Ollama model '{model_name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.info("Falling back to basic text analysis...")
            self.llm = None
        
        # Initialize embeddings for semantic analysis
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", 
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            self.embeddings = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_documents(self, folder_path: str) -> List:
        """Load and split all PDFs in the given folder."""
        folder = Path(folder_path)
        pdfs = list(folder.glob("*.pdf"))
        
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {folder_path}")

        all_docs = []
        logger.info(f"Found {len(pdfs)} PDF files to process")
        
        for pdf in pdfs:
            try:
                logger.info(f"Loading: {pdf.name}")
                loader = PyPDFLoader(str(pdf))
                docs = loader.load()
                
                # Add source document metadata
                for doc in docs:
                    doc.metadata['source_document'] = pdf.name
                    # Clean up the content
                    doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
                
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf.name}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf.name}: {e}")
                continue

        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(all_docs)
        logger.info(f"Created {len(split_docs)} text chunks")
        
        return split_docs

    def detect_sections(self, documents) -> List[Dict]:
        """Detect section headers and organize content."""
        sections = []
        current_section = None
        
        for doc in documents:
            content = doc.page_content
            filename = doc.metadata.get('source_document', 'Unknown')
            page = doc.metadata.get('page', 1) + 1  # Convert to 1-based indexing
            
            # Try to detect section headers
            lines = content.split('\n')
            detected_title = None
            
            for line in lines[:5]:  # Check first few lines
                line = line.strip()
                if self._is_likely_header(line):
                    detected_title = line
                    break
            
            # If no clear header found, use first meaningful line
            if not detected_title:
                for line in lines:
                    line = line.strip()
                    if len(line) > 10 and len(line) < 100:
                        detected_title = line[:80] + "..." if len(line) > 80 else line
                        break
            
            if not detected_title:
                detected_title = f"Content from page {page}"
            
            sections.append({
                'document': filename,
                'section_title': detected_title,
                'content': content,
                'page_number': page,
                'word_count': len(content.split())
            })

        # Merge sections with similar titles from same document
        merged_sections = self._merge_similar_sections(sections)
        
        logger.info(f"Detected {len(merged_sections)} sections")
        return merged_sections

    def _is_likely_header(self, line: str) -> bool:
        """Check if a line is likely to be a section header."""
        if not line or len(line) < 3:
            return False
        
        # Check various header patterns
        header_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^[A-Z][a-z\s]+:$', # Title case ending with colon
            r'^(Chapter|Section|Part)\s+\d+',  # Chapter/Section/Part
            r'^(Abstract|Introduction|Conclusion|References)',  # Common sections
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional heuristics
        if (line.istitle() and len(line.split()) <= 8 and 
            not line.endswith('.') and len(line) < 80):
            return True
            
        return False

    def _merge_similar_sections(self, sections: List[Dict]) -> List[Dict]:
        """Merge sections with very similar titles."""
        merged = []
        used_indices = set()
        
        for i, section in enumerate(sections):
            if i in used_indices:
                continue
                
            # Find similar sections
            similar_sections = [section]
            for j, other_section in enumerate(sections[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if (section['document'] == other_section['document'] and
                    self._are_similar_titles(section['section_title'], other_section['section_title'])):
                    similar_sections.append(other_section)
                    used_indices.add(j)
            
            # Merge content if multiple similar sections found
            if len(similar_sections) > 1:
                merged_content = ' '.join([s['content'] for s in similar_sections])
                merged_section = similar_sections[0].copy()
                merged_section['content'] = merged_content
                merged_section['word_count'] = len(merged_content.split())
                merged.append(merged_section)
            else:
                merged.append(section)
            
            used_indices.add(i)
        
        return merged

    def _are_similar_titles(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar enough to merge."""
        # Simple similarity check
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.6

    def score_sections_with_llm(self, sections, persona: str, task: str) -> List[Dict]:
        """Score sections using Ollama LLM."""
        if not self.llm:
            return self.score_sections_fallback(sections, persona, task)
        
        scored = []
        
        # Limit to reasonable number for performance
        sections_to_score = sections[:20]
        
        for i, section in enumerate(sections_to_score):
            try:
                prompt = f"""Rate the relevance of this document section for a {persona} whose task is: {task}

Section Title: {section['section_title']}
Content Preview: {section['content'][:800]}...

Rate from 0-10 (10 = extremely relevant, 0 = not relevant).
Respond with only the number:"""

                logger.debug(f"Scoring section {i+1}/{len(sections_to_score)}")
                
                response = self.llm.invoke(prompt).strip()
                
                # Extract score from response
                score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
                if score_match:
                    score = float(score_match.group(1))
                    score = min(max(score, 0), 10) / 10  # Normalize to 0-1
                else:
                    score = 0.5  # Default score
                    
            except Exception as e:
                logger.warning(f"Error scoring section: {e}")
                score = 0.5
            
            section['relevance_score'] = score
            scored.append(section)
        
        # Add remaining sections with default score
        for section in sections[20:]:
            section['relevance_score'] = 0.3
            scored.append(section)
        
        return sorted(scored, key=lambda x: x['relevance_score'], reverse=True)

    def score_sections_fallback(self, sections, persona: str, task: str) -> List[Dict]:
        """Fallback scoring method using keyword matching."""
        logger.info("Using fallback scoring method")
        
        # Extract keywords from persona and task
        persona_keywords = set(re.findall(r'\w+', persona.lower()))
        task_keywords = set(re.findall(r'\w+', task.lower()))
        all_keywords = persona_keywords.union(task_keywords)
        
        for section in sections:
            content_words = set(re.findall(r'\w+', section['content'].lower()))
            title_words = set(re.findall(r'\w+', section['section_title'].lower()))
            
            # Calculate keyword overlap
            content_overlap = len(all_keywords.intersection(content_words))
            title_overlap = len(all_keywords.intersection(title_words))
            
            # Score based on overlap and content length
            base_score = (content_overlap + title_overlap * 2) / max(len(all_keywords), 1)
            length_bonus = min(section['word_count'] / 200, 0.3)
            
            section['relevance_score'] = min(base_score + length_bonus, 1.0)
        
        return sorted(sections, key=lambda x: x['relevance_score'], reverse=True)

    def refine_section_with_llm(self, section, persona: str, task: str) -> str:
        """Generate refined summary using Ollama."""
        if not self.llm:
            return self._fallback_summary(section)
        
        try:
            prompt = f"""Summarize this content for a {persona} working on: {task}

Section: {section['section_title']}
Content: {section['content'][:1200]}

Provide a concise, actionable summary (2-3 sentences):"""

            response = self.llm.invoke(prompt).strip()
            
            # Clean up the response
            if len(response) > 500:
                response = response[:500] + "..."
            
            return response if response else self._fallback_summary(section)
            
        except Exception as e:
            logger.warning(f"Error refining section: {e}")
            return self._fallback_summary(section)

    def _fallback_summary(self, section) -> str:
        """Fallback summary method."""
        content = section['content']
        sentences = re.split(r'[.!?]+', content)
        
        # Take first 2-3 meaningful sentences
        summary_sentences = []
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                summary_sentences.append(sentence)
                if len(summary_sentences) >= 2:
                    break
        
        summary = '. '.join(summary_sentences)
        return summary[:400] + "..." if len(summary) > 400 else summary

    def analyze(self, persona: str, task: str, folder: str) -> AnalysisResult:
        """Main analysis function."""
        start_time = time.time()
        logger.info(f"Starting analysis for persona: {persona}")
        logger.info(f"Task: {task}")
        
        # Load and process documents
        documents = self.load_documents(folder)
        sections = self.detect_sections(documents)
        
        # Score sections for relevance
        ranked_sections = self.score_sections_with_llm(sections, persona, task)
        
        # Extract top sections
        top_sections = ranked_sections[:10]
        
        extracted_sections = [
            ExtractedSection(
                document=section['document'],
                page_number=section['page_number'],
                section_title=section['section_title'],
                importance_rank=i + 1
            )
            for i, section in enumerate(top_sections)
        ]
        
        # Generate refined summaries for top sections
        subsection_analysis = []
        for section in top_sections[:5]:  # Limit refinement to top 5 for performance
            refined_text = self.refine_section_with_llm(section, persona, task)
            
            subsection_analysis.append(
                SubsectionAnalysis(
                    document=section['document'],
                    section_title=section['section_title'],
                    refined_text=refined_text,
                    page_number=section['page_number']
                )
            )
        
        # Create metadata
        unique_documents = list(set(doc.metadata['source_document'] for doc in documents))
        
        metadata = {
            "input_documents": unique_documents,
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now().isoformat(),
            "model_used": "ollama" if self.llm else "fallback",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        result = AnalysisResult(metadata, extracted_sections, subsection_analysis)
        
        logger.info(f"Analysis completed in {metadata['processing_time_seconds']} seconds")
        logger.info(f"Found {len(extracted_sections)} relevant sections")
        
        return result

    def save_result(self, result: AnalysisResult, output_path: str):
        """Save analysis result to JSON file."""
        output_data = {
            "metadata": result.metadata,
            "extracted_sections": [asdict(section) for section in result.extracted_sections],
            "subsection_analysis": [asdict(analysis) for analysis in result.subsection_analysis]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")

def main():
    """Main entry point for Docker container."""
    # Setup paths
    input_file = Path("/app/input/input.json")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        logger.error("Input file not found: /app/input/input.json")
        return
    
    try:
        # Load input configuration
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Extract persona and job_to_be_done, handling both string and object formats
        persona_data = input_data.get('persona', '')
        job_data = input_data.get('job_to_be_done', '')
        
        # Handle case where persona is an object with 'role' field
        if isinstance(persona_data, dict):
            persona = persona_data.get('role', '')
        else:
            persona = str(persona_data)
            
        # Handle case where job_to_be_done is an object with 'task' field
        if isinstance(job_data, dict):
            job_to_be_done = job_data.get('task', '')
        else:
            job_to_be_done = str(job_data)
        
        if not persona or not job_to_be_done:
            logger.error("Missing persona or job_to_be_done in input.json")
            return
        
        # Initialize analyzer
        analyzer = DocumentAnalyzer(model_name="phi3:mini")
        
        # Run analysis
        result = analyzer.analyze(persona, job_to_be_done, "/app/input")
        
        # Save results
        output_file = output_dir / "analysis_result.json"
        analyzer.save_result(result, str(output_file))
        
        logger.info("✅ Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        
        # Create error output
        error_result = {
            "metadata": {
                "input_documents": [],
                "persona": "Error",
                "job_to_be_done": "Processing failed",
                "processing_timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        output_file = output_dir / "analysis_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2)

if __name__ == "__main__":
    main()