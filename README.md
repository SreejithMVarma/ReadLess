
# _READ LESS_ - chat, search, and save time - no more endless reading!

A powerful Streamlit application designed to simplify research by transforming URLs into insightful and contextual chat interactions. This tool leverages LangChain, FAISS, Hugging Face embeddings, Google Generative AI (Gemini 1.5 Flash), and other technologies for efficient data retrieval and conversational outputs.

## Features

- **Multi-URL Input**: Input up to three URLs for processing simultaneously.
- **Unstructured Data Parsing**: Extracts raw web content and converts it into meaningful chunks.
- **Embeddings with FAISS**: Efficiently indexes document embeddings for fast retrieval.
- **Chat Interface**: Engage with processed data through an intuitive chat interface.
- **Contextual AI Responses**: Powered by Google's Gemini 1.5 Flash for relevant and accurate answers.
- **Source Attribution**: Provides sources for the answers to ensure reliability and transparency.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python dependencies (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SreejithMVarma/ReadLess.git
   cd ReadLess
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Google API Key (Gemini 1.5 Flash)

1. You must generate a **Google API Key** for Gemini 1.5 Flash.
2. Once generated, paste your **Google API Key** in the input box provided in the **sidebar** of the application.

### Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser (default: `http://localhost:8501`).

3. In the sidebar, enter your **Google Generative AI API key** for Gemini 1.5 Flash directly (input box available at the top).

4. Input URLs in the sidebar and click **Process URLs** to extract and process data.

5. Start interacting with the processed content by typing your questions into the chat interface.

---

## File Structure

```plaintext
📂 ReadLess/
├── core.py               # Core functions and logic
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Technologies Used

- **Streamlit**: For building interactive web applications
- **LangChain**: For conversational AI workflows
- **FAISS**: For fast and efficient similarity search
- **Hugging Face Embeddings**: For semantic understanding of text
- **Google Generative AI (Gemini 1.5 Flash)**: For advanced AI-driven responses
- **Python**: Core programming language

---

## Contributing

Contributions are welcome! To contribute, follow these steps:

1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push your branch: `git push origin feature-name`.
5. Submit a pull request.

---

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