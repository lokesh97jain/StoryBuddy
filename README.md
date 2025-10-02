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
python -m streamlit run ainia_demo.py
```

Once started, Streamlit will provide a **local URL** (e.g., `http://localhost:8501`) where you can interact with the app.  

---

## 📦 Project Structure
```
├── ainia_demo.py              # Main app
├── requirements.txt     # Python dependencies
├── installation.py     # Python Installation file 
├── README.md            # Project documentation
```

---

## ✅ Notes
- Ensure you have Python 3.8+ installed  
- Use a virtual environment for cleaner dependency management  
- Replace `sk-...` with your actual OpenAI API key

---
## 📸 Demo Screenshots

### 👨‍👩‍👧 Parental View
<p align="center">
  <div style="display:inline-block; margin:15px; text-align:center; border:1px solid #ccc; border-radius:8px; padding:10px;">
    <b>Parental View – Image 1</b><br/>
    <img src="https://github.com/user-attachments/assets/f51e98fe-70ff-46df-96f3-50fb48c13e0b" alt="Parental Dashboard 1" width="400"/>
  </div>
  <div style="display:inline-block; margin:15px; text-align:center; border:1px solid #ccc; border-radius:8px; padding:10px;">
    <b>Parental View – Image 2</b><br/>
    <img src="https://github.com/user-attachments/assets/de23e664-8f0f-4814-ad00-00c47d907cc5" alt="Parental Dashboard 2" width="400"/>
  </div>
</p>

---

### 🧒 Child View
<p align="center">
  <div style="display:inline-block; margin:15px; text-align:center; border:1px solid #ccc; border-radius:8px; padding:10px;">
    <b>Child View – Image 1</b><br/>
    <img src="https://github.com/user-attachments/assets/e6772df4-eef5-4bd0-802d-c4e2f4951a62" alt="Child Story View 1" width="350"/>
  </div>
  <div style="display:inline-block; margin:15px; text-align:center; border:1px solid #ccc; border-radius:8px; padding:10px;">
    <b>Child View – Image 2</b><br/>
    <img src="https://github.com/user-attachments/assets/ed22c2bd-38bd-46b6-8a55-e1a37f661bd0" alt="Child Story View 2" width="350"/>
  </div>
</p>

<p align="center">
  <div style="display:inline-block; margin:15px; text-align:center; border:1px solid #ccc; border-radius:8px; padding:10px;">
    <b>Child View – Image 3</b><br/>
    <img src="https://github.com/user-attachments/assets/4f9661ca-9f5d-4721-933b-0f4d4e20cf03" alt="Child Story View 3" width="350"/>
  </div>
  <div style="display:inline-block; margin:15px; text-align:center; border:1px solid #ccc; border-radius:8px; padding:10px;">
    <b>Child View – Image 4</b><br/>
    <img src="https://github.com/user-attachments/assets/6a6b476d-0b76-4d5e-9989-7b4ab632b747" alt="Child Story View 4" width="350"/>
  </div>
</p>






<!--
## DEMO SCREENSHOTS
### Parental View - 
<img width="817" height="939" alt="image" src="https://github.com/user-attachments/assets/f51e98fe-70ff-46df-96f3-50fb48c13e0b" /> 
<img width="1034" height="805" alt="image" src="https://github.com/user-attachments/assets/de23e664-8f0f-4814-ad00-00c47d907cc5" />

### Child View - 

<img width="747" height="724" alt="image" src="https://github.com/user-attachments/assets/e6772df4-eef5-4bd0-802d-c4e2f4951a62" />
<img width="423" height="811" alt="image" src="https://github.com/user-attachments/assets/ed22c2bd-38bd-46b6-8a55-e1a37f661bd0" />
<img width="775" height="877" alt="image" src="https://github.com/user-attachments/assets/4f9661ca-9f5d-4721-933b-0f4d4e20cf03" />
<img width="379" height="947" alt="image" src="https://github.com/user-attachments/assets/6a6b476d-0b76-4d5e-9989-7b4ab632b747" />

 -->

