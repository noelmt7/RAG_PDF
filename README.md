# PDF Text Extraction and Querying App

## Overview
The PDF Text Extraction and Querying App is designed to process multiple PDF files, extract text content, generate embeddings, and provide context-aware answers to user queries using a retrieval-augmented generation approach with the Groq language model. The project also incorporates a fine-tuned model based on the TinyLlama-1.1B-Chat-v1.0 model from Hugging Face Transformers.

## System Architecture
The architecture consists of several key components working together to facilitate the extraction and querying process:

```plaintext
+----------------------------------+
|      User Interface (Streamlit)  |
|          (Frontend)               |
+----------------------------------+
                |
                |
                v
+----------------------------------+
|      PDF Processing Module        |
| (Extracts text and applies OCR)  |
+----------------------------------+
                |
                |
                v
+----------------------------------+
|      Embedding Generation Module   |
|    (SentenceTransformer Model)    |
+----------------------------------+
                |
                |
                v
+----------------------------------+
|        FAISS Index                |
|     (Similarity Search Engine)    |
+----------------------------------+
                |
                |
                v
+----------------------------------+
|     Groq Language Model           |
|       (Response Generation)       |
+----------------------------------+
                |
                |
                v
+----------------------------------+
|    Fine-Tuned Language Model      |
|   (TinyLlama-1.1B-Chat-v1.0)     |
+----------------------------------+
```

### Component Descriptions

1. **User Interface (Streamlit)**:
   - **Function**: Acts as the frontend for user interaction. Users can upload PDF files and enter queries to extract relevant information.
   - **Technologies**: Streamlit framework.

2. **PDF Processing Module**:
   - **Function**: Responsible for extracting text from uploaded PDF files using PyMuPDF. If text extraction fails, it applies Optical Character Recognition (OCR) using Tesseract.
   - **Key Libraries**: 
     - `PyMuPDF` for PDF text extraction.
     - `Pytesseract` for OCR processing.

3. **Embedding Generation Module**:
   - **Function**: Generates embeddings from the extracted text using the SentenceTransformer model. Normalizes embeddings for better similarity search performance.
   - **Key Libraries**: 
     - `SentenceTransformer` from Hugging Face for generating embeddings.
   
4. **FAISS Index**:
   - **Function**: Stores the generated embeddings for efficient similarity search. It utilizes the FAISS library for high-speed vector search operations.
   - **Key Libraries**: 
     - `FAISS` for managing and querying the vector space of embeddings.

5. **Groq Language Model**:
   - **Function**: Takes user queries and relevant extracted text to generate context-aware responses using the Groq model.
   - **Key Libraries**: 
     - Groq API for chat completions and response generation.

6. **Fine-Tuned Language Model**:
   - **Function**: A fine-tuned variant of the TinyLlama-1.1B-Chat model is employed to enhance response quality and relevance.
   - **Key Libraries**: 
     - `Transformers` library from Hugging Face for fine-tuning and generating responses.

## Workflow
1. **PDF Upload**: Users upload one or more PDF files via the Streamlit interface.
2. **Text Extraction**: The application processes each PDF, extracting text or applying OCR if necessary.
3. **Embedding Creation**: Extracted text is encoded into embeddings.
4. **Indexing**: Embeddings are indexed using FAISS for rapid similarity searching.
5. **User Query**: Users input their queries, and the application retrieves the most relevant text segments.
6. **Response Generation**: The retrieved text and user query are sent to the Groq model and the fine-tuned TinyLlama model to generate coherent responses, which are displayed to the user.

## Installation and Setup

### Prerequisites
- Python 3.7+
- Install required libraries:
```bash
pip install streamlit pymupdf pytesseract sentence-transformers faiss-cpu pinecone dotenv groq
```

### Environment Variables
Create a `.env` file in the project root with the following variables:
```
PINE_API=<your_pinecone_api_key>
GROQ_API_KEY=<your_groq_api_key>
```

### Running the App
To run the application, execute the following command in your terminal:
```bash
streamlit run app.py
```

## Conclusion
This application provides an efficient solution for extracting information from PDF files and generating contextually relevant responses to user queries. The use of advanced NLP models and efficient indexing techniques ensures quick and accurate results. The integration of a fine-tuned model enhances the overall quality of responses.

## Future Work
Future enhancements may include:
- Support for additional file formats (e.g., DOCX, TXT).
- Improved OCR capabilities for better text recognition.
- Integration with more advanced models for enhanced response generation.
