# ============================================================
# 1️⃣ Setup: Libraries & Hardware
# ============================================================
!pip install sentence-transformers datasets scikit-learn torch psutil matplotlib seaborn transformers accelerate pynvml -q

import numpy as np, torch, time, math, subprocess, pynvml, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, models
from huggingface_hub import login

# Hardware Check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {device.upper()}")
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(f"✅ GPU Detected: {pynvml.nvmlDeviceGetName(handle).decode('utf-8')}")
except:
    print("⚠️ No NVIDIA GPU detected for power profiling.")

def gpu_power(): 
    try: return pynvml.nvmlDeviceGetPowerUsage(handle)/1000
    except: return 0

# Auth (Remplacez par votre token ou commentez si public)
# login(token="hf_xxxxxxxxxxxxxxxxxxxx")