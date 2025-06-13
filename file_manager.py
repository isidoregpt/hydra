"""
File upload and processing system for Hydra v3
"""
import os
import tempfile
import shutil
import zipfile
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import streamlit as st

class FileManager:
    """Manages file uploads, processing, and storage for Hydra v3"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.upload_dir = self._create_session_directory()
        self.processed_files = []
        
    def _create_session_directory(self) -> str:
        """Create a unique directory for this session's uploads"""
        base_dir = tempfile.gettempdir()
        session_dir = os.path.join(base_dir, f"hydra_session_{self.session_id}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def process_uploaded_files(self, uploaded_files) -> Dict[str, Any]:
        """
        Process uploaded files from Streamlit file uploader
        
        Args:
            uploaded_files: Files from st.file_uploader
            
        Returns:
            Dict with processing results and file paths
        """
        if not uploaded_files:
            return {"files": [], "total_files": 0, "errors": []}
        
        results = {
            "files": [],
            "total_files": 0,
            "errors": [],
            "file_types": {},
            "total_size": 0
        }
        
        for uploaded_file in uploaded_files:
            try:
                file_result = self._process_single_file(uploaded_file)
                results["files"].append(file_result)
                results["total_files"] += file_result.get("file_count", 1)
                results["total_size"] += file_result.get("size", 0)
                
                # Track file types
                file_type = file_result.get("type", "unknown")
                results["file_types"][file_type] = results["file_types"].get(file_type, 0) + 1
                
            except Exception as e:
                results["errors"].append(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return results
    
    def _process_single_file(self, uploaded_file) -> Dict[str, Any]:
        """Process a single uploaded file"""
        file_name = uploaded_file.name
        file_size = len(uploaded_file.getvalue())
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(file_name)
        file_extension = Path(file_name).suffix.lower()
        
        # Save the file
        file_path = os.path.join(self.upload_dir, file_name)
        
        result = {
            "name": file_name,
            "path": file_path,
            "size": file_size,
            "type": self._categorize_file_type(file_extension, mime_type),
            "extension": file_extension,
            "mime_type": mime_type,
            "file_count": 1
        }
        
        # Handle different file types
        if file_extension == '.zip':
            return self._process_zip_file(uploaded_file, file_path, result)
        else:
            # Regular file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            result["absolute_path"] = os.path.abspath(file_path)
            self.processed_files.append(result["absolute_path"])
            
            return result
    
    def _process_zip_file(self, uploaded_file, file_path: str, result: Dict) -> Dict[str, Any]:
        """Process uploaded ZIP file by extracting contents"""
        
        # Save the ZIP file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Extract ZIP contents
        extract_dir = os.path.join(self.upload_dir, f"{Path(uploaded_file.name).stem}_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Scan extracted files
            extracted_files = []
            total_size = 0
            
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if not file.startswith('.'):  # Skip hidden files
                        full_path = os.path.join(root, file)
                        abs_path = os.path.abspath(full_path)
                        extracted_files.append(abs_path)
                        self.processed_files.append(abs_path)
                        total_size += os.path.getsize(full_path)
            
            result.update({
                "type": "archive",
                "extracted_to": extract_dir,
                "extracted_files": extracted_files,
                "file_count": len(extracted_files),
                "total_extracted_size": total_size,
                "absolute_path": os.path.abspath(extract_dir)
            })
            
            return result
            
        except zipfile.BadZipFile:
            result["error"] = "Invalid ZIP file"
            return result
    
    def _categorize_file_type(self, extension: str, mime_type: str) -> str:
        """Categorize file types for better organization"""
        
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.css', '.html', '.php', '.rb', '.go', '.rs', '.swift', '.kt'
        }
        
        document_extensions = {
            '.txt', '.md', '.pdf', '.doc', '.docx', '.rtf'
        }
        
        data_extensions = {
            '.csv', '.json', '.xml', '.yaml', '.yml', '.sql'
        }
        
        config_extensions = {
            '.env', '.ini', '.conf', '.config', '.toml'
        }
        
        if extension in code_extensions:
            return "code"
        elif extension in document_extensions:
            return "document"
        elif extension in data_extensions:
            return "data"
        elif extension in config_extensions:
            return "config"
        elif extension == '.zip':
            return "archive"
        else:
            return "other"
    
    def get_file_summary(self) -> Dict[str, Any]:
        """Get a summary of all processed files"""
        if not self.processed_files:
            return {"total_files": 0, "message": "No files uploaded"}
        
        file_types = {}
        total_size = 0
        
        for file_path in self.processed_files:
            if os.path.exists(file_path):
                extension = Path(file_path).suffix.lower()
                file_type = self._categorize_file_type(extension, None)
                file_types[file_type] = file_types.get(file_type, 0) + 1
                total_size += os.path.getsize(file_path)
        
        return {
            "total_files": len(self.processed_files),
            "file_types": file_types,
            "total_size": total_size,
            "paths": self.processed_files
        }
    
    def get_analysis_ready_paths(self) -> List[str]:
        """Get list of file paths ready for analysis"""
        return [path for path in self.processed_files if os.path.exists(path)]
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.upload_dir):
                shutil.rmtree(self.upload_dir)
        except Exception as e:
            st.warning(f"Could not clean up temporary files: {e}")
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"


class FileAnalysisInterface:
    """Interface for file analysis within Hydra v3"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    def create_file_context_prompt(self, user_query: str) -> str:
        """Create a prompt that includes file context"""
        
        file_summary = self.file_manager.get_file_summary()
        
        if file_summary["total_files"] == 0:
            return user_query
        
        # Build file context
        context_parts = [
            f"=== UPLOADED FILES CONTEXT ===",
            f"Total files: {file_summary['total_files']}",
            f"Total size: {FileManager.format_file_size(file_summary['total_size'])}",
        ]
        
        # Add file type breakdown
        if file_summary.get("file_types"):
            context_parts.append("File types:")
            for file_type, count in file_summary["file_types"].items():
                context_parts.append(f"  - {file_type}: {count} files")
        
        context_parts.extend([
            "",
            f"Files available for analysis at these paths:",
        ])
        
        # Add file paths
        for path in file_summary["paths"]:
            context_parts.append(f"  - {path}")
        
        context_parts.extend([
            "=== END FILES CONTEXT ===",
            "",
            f"User Query: {user_query}",
            "",
            "Please analyze the uploaded files in relation to the user's query. "
            "You can reference specific files by their paths and provide insights "
            "based on the file contents."
        ])
        
        return "\n".join(context_parts)
    
    def suggest_analysis_approaches(self, file_summary: Dict) -> List[str]:
        """Suggest analysis approaches based on uploaded files"""
        
        suggestions = []
        file_types = file_summary.get("file_types", {})
        
        if "code" in file_types:
            suggestions.extend([
                "🔍 Code Analysis: Review code quality, security, and architecture",
                "🐛 Debug Analysis: Identify potential bugs and issues",
                "📊 Code Review: Professional code review with recommendations"
            ])
        
        if "data" in file_types:
            suggestions.extend([
                "📈 Data Analysis: Analyze patterns and insights in your data",
                "🔍 Data Structure: Understand data schemas and relationships",
                "📊 Data Quality: Assess data completeness and consistency"
            ])
        
        if "document" in file_types:
            suggestions.extend([
                "📚 Document Analysis: Summarize and extract key information", 
                "🔍 Content Review: Analyze document structure and content",
                "📝 Documentation Quality: Review clarity and completeness"
            ])
        
        if "config" in file_types:
            suggestions.extend([
                "⚙️ Configuration Review: Analyze settings and configurations",
                "🔒 Security Analysis: Review configuration security",
                "🔧 Setup Analysis: Understand system configuration"
            ])
        
        # General suggestions
        suggestions.extend([
            "🤔 Ask Questions: Ask specific questions about your files",
            "🎯 Custom Analysis: Request specific analysis based on your needs"
        ])
        
        return suggestions
