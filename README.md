# StoryBuddy
Prototype for kid-safe interactive stories with choices, TTS, and simple illustrations, plus a minimal parent view.

# Interactive Story App (Streamlit + OpenAI)

This is a **Streamlit-based application** that generates kid-friendly interactive stories with text-to-speech and AI-generated images using the OpenAI API.  

## 🚀 Features
- Create interactive stories with branching choices  
- Text-to-Speech (TTS) using `gTTS`  
- Image generation via OpenAI API  
- Runs locally with Streamlit  

---

## 🛠️ Installation & Setup

Clone this repository and navigate into the project folder:  

```bash
git clone https://github.com/lokesh97jain/StoryBuddy.git
cd StoryBuddy
```

### 1. Install dependencies  
Install the required Python packages:  

#### Windows (CMD)
```cmd
python installation.py
```

#### Windows (PowerShell)
```powershell
python installation.py
```

#### macOS / Linux (bash/zsh)
```bash
python3 installation.py
```

---

### 2. Set your **OpenAI API Key**

You’ll need an OpenAI API key. Set it as an environment variable:  

#### Windows (CMD)
```cmd
set OPENAI_API_KEY=sk-...
```

#### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="sk-..."
```

#### macOS / Linux (bash/zsh)
```bash
export OPENAI_API_KEY="sk-..."
```

---

### 3. Run the Streamlit app

```bash
streamlit run Code.py
```

Once started, Streamlit will provide a **local URL** (e.g., `http://localhost:8501`) where you can interact with the app.  

---

## 📦 Project Structure
```
├── Code.py              # Main Streamlit app
├── requirements.txt     # Python dependencies
├── installation.py     # Python Installation file 
├── README.md            # Project documentation
```

---

## ✅ Notes
- Ensure you have Python 3.8+ installed  
- Use a virtual environment for cleaner dependency management  
- Replace `sk-...` with your actual OpenAI API key  

