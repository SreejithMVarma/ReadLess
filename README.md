# ReadLess - Chat-based URL Research Tool

A powerful Streamlit application that simplifies research by transforming URLs into insightful and contextual chat interactions. This tool leverages LangChain, FAISS, Hugging Face embeddings, and Google Generative AI for efficient data retrieval and conversational outputs.

## Features

- **Multi-URL Input**: Add up to three URLs for processing simultaneously.
- **Unstructured Data Parsing**: Handles raw web content and converts it into meaningful chunks.
- **Embeddings with FAISS**: Efficiently indexes document embeddings for fast retrieval.
- **Chat Interface**: Query processed data through an intuitive chat interface.
- **Contextual AI Responses**: Powered by Google's Gemini for relevant and accurate answers.
- **Source Attribution**: Provides sources for retrieved answers to ensure reliability.

---

## Getting Started

### Prerequisites

- Python 3.8+
- A Google Generative AI API key
- Required Python dependencies (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SreejithMVarma/ReadLess.git
   cd ReadLess


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Create a `.env` file in the project root.
   - Add your **Google Generative AI API key**:
     ```env
     GOOGLE_API_KEY=your_api_key
     ```

### Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser (default: `http://localhost:8501`).

3. Input URLs in the sidebar and click **Process URLs** to extract data.

4. Start chatting in the interface by asking questions related to the processed content.

---

## File Structure

```plaintext
📂 ReadLess/
├── core.py                # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # environment variables
├── vector_index.pkl      # Saved vector index (auto-generated)
└── README.md             # Project documentation
```

---

## Technologies Used

- **Streamlit**: For interactive web applications
- **LangChain**: Building conversational AI workflows
- **FAISS**: Fast and efficient similarity search
- **Hugging Face Embeddings**: Semantic understanding of text
- **Google Generative AI**: State-of-the-art conversational AI
- **Python**: Core programming language

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## Acknowledgements

Special thanks to:

- [LangChain Community](https://langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [Google AI](https://ai.google/)

---

## Author

**Sreejith M Varma**  
- [Website](https://sreejithmvarma.in)  
- [LinkedIn](https://www.linkedin.com/in/sreejithmvarma)  
- [GitHub](https://github.com/SreejithMVarma)

---

### Happy Researching! 🚀

---
