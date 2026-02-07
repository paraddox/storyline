# **The Dungeon Master’s Silicon Scribe: Architecting a Self-Hosted, AI-Driven Campaign Archive on Consumer Hardware**

## **1\. Introduction: The Ephemeral Nature of Tabletop Narratives**

In the realm of tabletop role-playing games (TTRPGs) such as Dungeons & Dragons (D\&D), the narrative is a collaborative, ephemeral construct. It exists primarily in the shared acoustic space of the players and the Dungeon Master (DM). Unlike video games, where logs are automatically generated, or novels, where the text is fixed, a D\&D session is a fluid improvisation of dialogue, mechanics, and "theater of the mind." This transience presents a significant challenge for long-running campaigns: the degradation of memory. Critical plot points, obscure non-player character (NPC) names, and improvised lore details often vanish into the ether the moment they are spoken, preserved only in the fragmented, subjective, and often incomplete handwritten notes of the participants.

The advent of large language models (LLMs) and advanced automatic speech recognition (ASR) systems offers a transformative solution to this "lore entropy." By creating a self-hosted, automated pipeline, it is now possible to capture the raw audio-visual data of a session, transmute it into highly accurate, speaker-attributed text, and index it within a semantic database. This allows the DM to query the campaign’s history as one would a search engine, asking complex questions like "What did the rogue promise the goblin king in session three?" and receiving an answer grounded in the actual recording.

This report outlines a comprehensive technical architecture for building such a system. It is specifically tailored for deployment on high-end consumer hardware, centered around the NVIDIA RTX 3090 GPU. The analysis prioritizes data sovereignty (self-hosting), cost-efficiency (open-source software), and the specific, nuanced requirements of transcribing multi-speaker, fantasy-laden dialogue.

## **2\. Computational Infrastructure and Hardware Constraints**

The foundation of any local AI pipeline is the hardware accelerator. The NVIDIA GeForce RTX 3090, utilized in this proposed architecture, represents a unique inflection point in consumer GPU history. Its 24 gigabytes of GDDR6X video memory (VRAM) places it in a rarefied tier of hardware capable of running "datacenter-class" workloads without the prohibitive cost of enterprise cards like the A100 or H100. Understanding the memory hierarchy and computational limits of this card is essential for orchestrating the competing demands of transcription, diarization, and LLM inference.

### **2.1 The VRAM Imperative in AI Workloads**

In deep learning inference, VRAM is the primary bottleneck. The model weights—the parameters that define the neural network—must be loaded into VRAM for the CUDA cores to perform matrix multiplications efficiently. If a model exceeds the available VRAM, the system must offload layers to the system RAM (CPU memory) via the PCIe bus. This offloading results in a catastrophic drop in performance, often slowing generation speeds from 50+ tokens per second (t/s) to fewer than 3 t/s.1

For a D\&D archival project, the RTX 3090’s 24GB buffer allows for a "Goldilocks" configuration. It is sufficient to run state-of-the-art transcription models (like Whisper Large-v3) or substantial LLMs (like Qwen-2.5-32B or Llama-3-8B) entirely on the GPU. However, it is *not* sufficient to run both the largest transcription models and the largest LLMs simultaneously at full precision. This necessitates a sequential pipeline architecture, where resources are dynamically allocated: first to the transcription engine (Ingestion Mode), and then to the RAG system (Query Mode).3

### **2.2 Precision and Quantization**

To maximize the utility of the 24GB VRAM, modern AI workflows utilize quantization—reducing the precision of the model weights from 16-bit floating-point (FP16) to 8-bit, 6-bit, or even 4-bit integers.

* **FP16 (Half Precision):** Standard for inference. A 32-billion parameter model requires approximately 64GB of VRAM, far exceeding the 3090's capacity.  
* **Q4\_K\_M (4-bit Quantization):** This compression technique reduces the memory footprint significantly. A 32B model at 4-bit quantization occupies roughly 18-20 GB of VRAM. This fits comfortably on an RTX 3090, leaving \~4GB for the context window (KV cache), which stores the conversation history.2

The implications for this project are clear: while transcription requires high precision to capture the nuances of speech, the retrieval and summarization (LLM) phase must leverage quantization to fit a sufficiently "smart" model into memory alongside the operating system overhead.

### **2.3 System Integration and Software Stack**

The hardware must be exposed to the software layer through a robust driver stack. The recommended environment is a Linux-based operating system (e.g., Ubuntu 22.04 LTS) or Windows Subsystem for Linux 2 (WSL2), running the NVIDIA Container Toolkit. This allows Docker containers to access the GPU directly.

**Critical Software Components:**

* **NVIDIA Drivers (535+):** Essential for CUDA 12.x compatibility.  
* **CUDA Toolkit:** Provides the parallel computing platform.  
* **Docker & Docker Compose:** Facilitates the modular deployment of the transcription service, database, and web UI without "dependency hell".5

## **3\. The Acoustic Pipeline: From Signal to Semantics**

The first stage of the pipeline involves converting the raw recording into text. D\&D sessions pose unique acoustic challenges: they involve multiple speakers (often 4-7), overlapping speech (crosstalk), wide dynamic ranges (shouting/whispering), and non-standard vocabulary (fantasy proper nouns).

### **3.1 Audio Extraction and Pre-processing**

The user query specifies "audio video" recordings. Before any AI processing can occur, the audio track must be extracted and normalized. Video containers (MP4, MKV) often contain high-bitrate AAC or PCM audio.

**The FFMPEG Normalization Layer:**

Raw audio often suffers from uneven levels—the DM might be close to the mic, while a player across the table is faint. A pre-processing script using ffmpeg is required to:

1. **Extract:** Strip the audio stream from the video file.  
2. **Downmix:** Convert multi-channel audio to mono (unless the recording setup used a multi-track interface, which is rare for casual setups).  
3. **Resample:** Convert to 16kHz, the native sampling rate for the Whisper model.  
4. **Normalize:** Apply dynamic range compression to boost quiet voices and attenuate loud outbursts, ensuring the ASR model receives a consistent signal.7

### **3.2 The Automatic Speech Recognition (ASR) Engine: WhisperX**

OpenAI's Whisper model is the industry standard for open-source ASR. However, the vanilla Whisper implementation suffers from two critical flaws for long-form transcription: hallucination during silence and imprecise timestamping. For this project, **WhisperX** is the superior architectural choice.5

#### **3.2.1 Why WhisperX?**

WhisperX enhances the standard Whisper model through three mechanisms:

1. **Voice Activity Detection (VAD):** It uses a separate model (typically pyannote/segmentation) to detect the exact start and end times of speech. It only feeds active speech segments to the Whisper model. This eliminates the "hallucination loops" where Whisper generates repetitive garbage text during long pauses—a frequent occurrence in D\&D sessions when players are thinking or rolling dice.10  
2. **Batch Processing:** Because the audio is pre-segmented by the VAD, WhisperX can batch these segments and feed them to the GPU in parallel. On an RTX 3090, this results in transcription speeds up to 70x real-time (transcribing a 4-hour session in under 5 minutes).11  
3. **Forced Alignment:** Standard Whisper outputs timestamps at the "utterance" level, which can drift by seconds. WhisperX utilizes a secondary phoneme alignment model (Wav2Vec2) to force-align the transcript with the audio waveform at the word level. This precision is non-negotiable for accurate speaker attribution.11

#### **3.2.2 Model Selection**

For a self-hosted setup on an RTX 3090, **Whisper Large-v3** is the recommended model.

* **Accuracy:** It offers the lowest Word Error Rate (WER) on multilingual and accented speech.  
* **VRAM Usage:** Large-v3 requires approximately 10-12 GB of VRAM. This fits easily within the 3090's 24GB budget during the ingestion phase.5  
* **Fantasy Terminology:** The larger model has a broader world-knowledge base, making it slightly more likely to correctly phonetically approximate fantasy terms, though it still requires prompting assistance (discussed in Section 3.4).

### **3.3 The Speaker Diarization Challenge**

Transcribing *what* was said is only half the battle; identifying *who* said it is crucial for a D\&D transcript. A transcript that reads "I cast Fireball" without knowing whether it was the Wizard or the Fighter is useless for campaign continuity.

#### **3.3.1 Pyannote.audio Integration**

The diarization engine of choice is **Pyannote.audio 3.1**, widely regarded as the state-of-the-art open-source solution.15 WhisperX integrates Pyannote directly into its pipeline.

The Pyannote workflow involves:

1. **Speaker Segmentation:** Detecting instantaneous speaker changes.  
2. **Embedding:** Converting audio segments into high-dimensional vectors that represent the acoustic characteristics of the voice (a "voice fingerprint").  
3. **Clustering:** Grouping these vectors to determine that "Speaker A" at timestamp 00:10 is the same person as "Speaker A" at 01:45.16

#### **3.3.2 Optimizing for D\&D Groups**

Diarization models often struggle with the "cocktail party problem"—distinguishing voices when multiple people laugh or shout over each other. However, a D\&D group has a stable number of participants. We can significantly improve accuracy by providing the model with a "hint."

* **Constraint Injection:** By passing the arguments \--min\_speakers X and \--max\_speakers Y (e.g., min 4, max 5\) to the WhisperX/Pyannote pipeline, we constrain the clustering algorithm. This prevents the model from hallucinating a 6th speaker due to a weird microphone artifact or a funny voice done by the DM.11

### **3.4 Addressing the "Fantasy Vocabulary" Gap**

Standard ASR models are trained on varied datasets, but rarely on the specific proper nouns of a user's homebrew campaign. "Phandalin" might become "Fan Dalin," and "Tiamat" might become "Tea Mat."

#### **3.4.1 Prompt Engineering for ASR**

Whisper supports an initial\_prompt parameter. This is a mechanism to bias the model's decoder towards specific tokens.

* **Implementation:** The pipeline should construct a prompt string containing a list of player names, character names, key locations, and deities relevant to the campaign.  
* **Example Prompt:** *"Transcript of a D\&D session featuring characters Grom, Elara, and Zyloph exploring the Underdark, fighting Mind Flayers, and visiting the city of Menzoberranzan."* This simple inclusion drastically reduces phonetic errors for campaign-specific proper nouns.17

## **4\. Semantic Storage: Vector Databases and Chunking**

Once the audio is transcribed and diarized, the resulting text is a massive, unstructured JSON or text blob. To make this data retrievable by an LLM ("What did we do last session?"), it must be indexed in a vector database.

### **4.1 The Vector Database Strategy**

A vector database stores text not as strings, but as numerical embeddings—vectors of numbers that represent the *semantic meaning* of the text. This allows for "semantic search," where the query "Who is the villain?" can retrieve chunks discussing "Strahd" even if the word "villain" isn't explicitly used.

For a self-hosted RTX 3090 setup, **ChromaDB** is the optimal choice.

* **Architecture:** It works as an embedded database or a lightweight server container. It does not require the heavy infrastructure overhead of alternatives like Milvus or Weaviate.  
* **Integration:** It is the default vector store for Open WebUI, the recommended interface for this project.6

### **4.2 The "Speaker-Aware" Chunking Problem**

Standard RAG pipelines chop text into fixed-length chunks (e.g., 500 characters). For a play script, this is disastrous. It might split a sentence in half, or worse, separate a speaker label from their dialogue.

* *Bad Chunk:* ...and then I said, "You shall not (Speaker label lost in previous chunk).  
* *Result:* The LLM retrieves the text but doesn't know who said it.

**Solution: Semantic Dialogue Chunking**

The pipeline requires a custom preprocessing script that respects the logical boundaries of the conversation.

1. **Turn Aggregation:** Group consecutive sentences spoken by the same speaker into a single block.  
2. **Metadata Injection:** Crucially, the speaker's name must be injected *into the text content* of the chunk, not just stored as hidden metadata.  
   * *Format:* You see a dark cave. I check for traps. This ensures that the embedding model associates the semantic action ("checking for traps") with the entity ("Rogue") within the vector space itself.21

## **5\. The Retrieval and Interface Layer: Open WebUI**

The final component is the user interface where the DM interacts with their data. **Open WebUI** (formerly Ollama WebUI) provides a polished, ChatGPT-like experience that runs entirely locally.

### **5.1 Why Open WebUI?**

* **Native RAG:** It has a built-in RAG pipeline that connects directly to Ollama and ChromaDB. You can upload documents (or in this case, transcripts) and immediately query them using \# commands.24  
* **Ollama Orchestration:** It acts as a frontend for **Ollama**, the backend service that will manage the LLM on the RTX 3090\.  
* **Citations:** When the LLM answers a question, Open WebUI provides citations. Clicking a citation highlights the exact chunk of the transcript used to generate the answer. This is vital for a DM to verify facts—ensuring the AI isn't hallucinating a plot point.24

### **5.2 LLM Selection for the RTX 3090**

With 24GB of VRAM, the system can run powerful quantized models.

* **The Contender: Llama-3.1 8B (Q8):** Very fast, low memory usage (\~10GB). Good for simple queries.  
* **The Champion: Qwen-2.5 32B (Q4\_K\_M):** This model represents the current state-of-the-art for mid-sized models. At 4-bit quantization, it fits into roughly 19-20GB of VRAM. It possesses superior reasoning capabilities and a massive context window (up to 128k tokens supported, though limited by VRAM), making it excellent for synthesizing complex campaign lore across multiple sessions.4

**Comparison Table: Models on RTX 3090**

| Model | Size (Quant) | VRAM Usage | Capabilities | Recommendation |
| :---- | :---- | :---- | :---- | :---- |
| **Llama-3.1 8B** | Q8 | \~10 GB | Fast, good reasoning, allows concurrent Whisper use. | Backup |
| **Mistral Small 24B** | Q4 | \~14 GB | Strong logic, restricted context. | Alternative |
| **Qwen-2.5 32B** | Q4 | \~19 GB | Near-GPT4 reasoning, extensive knowledge retention. | **Primary** |
| **Llama-3.1 70B** | EXL2 (2.4bpw) | \~22 GB | Very high perplexity (dumb) at this compression. | Avoid |

4

## **6\. Implementation Architecture**

This section details the implementation of the "Archivist" pipeline using Docker Compose.

### **6.1 Architecture Diagram (Conceptual)**

The system consists of three primary containers:

1. **whisperx-service**: An API wrapper around WhisperX/Pyannote. It mounts the GPU to perform batch transcription.  
2. **ollama**: The inference server for the LLM (Qwen/Llama). It mounts the GPU for query generation.  
3. **open-webui**: The frontend and RAG orchestrator. It connects to Ollama and manages the Vector DB.

**Resource Conflict Management:** Since both WhisperX and Ollama need the GPU, they cannot run heavy workloads simultaneously without OOM errors. The workflow is designed to be **asynchronous**:

* *Post-Session:* The DM uploads the audio. The whisperx-service grabs the GPU resources. Ollama sits idle or is temporarily stopped.  
* *Prep-Time:* Transcription finishes. whisperx-service releases VRAM. Ollama loads the LLM for RAG queries.

### **6.2 Docker Compose Configuration**

The following docker-compose.yml structure orchestrates the environment. Note the use of NVIDIA runtime and shared volumes.

YAML

version: '3.8'

services:  
  \# Service 1: The Transcription Engine  
  whisperx:  
    image: murtaza-nasir/whisperx-asr-service:latest-gpu  
    container\_name: whisperx\_api  
    environment:  
      \- HF\_TOKEN=${HF\_TOKEN} \# Hugging Face token for Pyannote  
      \- WHISPER\_MODEL=large-v3  
      \- LANG=en  
    volumes:  
      \-./input\_audio:/app/audio  
      \-./output\_transcripts:/app/output  
    deploy:  
      resources:  
        reservations:  
          devices:  
            \- driver: nvidia  
              count: 1  
              capabilities: \[gpu\]  
    ports:  
      \- "8000:8000"  
    restart: unless-stopped

  \# Service 2: The LLM Backend  
  ollama:  
    image: ollama/ollama:latest  
    container\_name: ollama  
    volumes:  
      \- ollama\_storage:/root/.ollama  
    deploy:  
      resources:  
        reservations:  
          devices:  
            \- driver: nvidia  
              count: 1  
              capabilities: \[gpu\]  
    ports:  
      \- "11434:11434"  
    restart: always

  \# Service 3: The RAG Frontend  
  open-webui:  
    image: ghcr.io/open-webui/open-webui:main  
    container\_name: open-webui  
    environment:  
      \- OLLAMA\_BASE\_URL=http://ollama:11434  
      \- ENABLE\_RAG\_WEB\_SEARCH=False  
      \- RAG\_EMBEDDING\_ENGINE=ollama \# Use Ollama for embeddings too  
      \- RAG\_EMBEDDING\_MODEL=nomic-embed-text  
    volumes:  
      \- openwebui\_data:/app/backend/data  
    ports:  
      \- "3000:8080"  
    depends\_on:  
      \- ollama  
    restart: always

volumes:  
  ollama\_storage:  
  openwebui\_data:

5

### **6.3 The Automation Glue Script**

To bridge WhisperX output with Open WebUI, a Python script is necessary. This script performs the "ETL" (Extract, Transform, Load) process.

**Script Logic:**

1. **Extract:** Monitor the ./input\_audio folder. When a file (e.g., session\_42.mp3) appears, send a POST request to localhost:8000/transcribe with diarization enabled.  
2. **Transform:** Receive the JSON response.  
   * *Speaker Mapping:* Use a configuration file players.json ({"SPEAKER\_00": "DM", "SPEAKER\_01": "Grom"}) to rename speakers.  
   * *Formatting:* Convert the JSON into a clean Markdown file.  
   * *Header:* Add metadata: \# Session 42 \-.  
   * *Body:* Format dialogue as \*\*Grom:\*\* I attack the darkness.  
3. **Load:** Use the Open WebUI API to upload the Markdown file into a specific Knowledge Collection (e.g., "Campaign 1").  
   * API Endpoint: POST /api/v1/files/ (Upload)  
   * API Endpoint: POST /api/v1/knowledge/{id}/file/add (Index).30

## **7\. Advanced Workflow: From Audio to Wiki**

This section describes the practical workflow for the user.

### **7.1 "Session Zero" Configuration**

Before the first recording, the DM must configure the system.

1. **Hugging Face Auth:** Pyannote requires accepting user agreements on Hugging Face. The API token must be generated and added to the .env file for the Docker container.11  
2. **Speaker Profiles:** To map SPEAKER\_00 to "Alice" consistently, the user needs a reference. The system output will initially be anonymous. The user must manually review the first transcript or listen to audio snippets to map IDs to names in the script's players.json file.  
   * *Note on Automation:* While advanced systems use voice embeddings to auto-recognize speakers across sessions, Pyannote 3.1 generally treats each file as a new session. Simple manual mapping (Speaker 00 \= Alice) per session is often more reliable and takes only seconds.33

### **7.2 Post-Session Routine**

1. **Ingest:** Drop session.mp3 into the input folder.  
2. **Process:** The script triggers WhisperX. On an RTX 3090, a 4-hour session will take approximately 10-15 minutes to process (transcription \+ alignment \+ diarization).11  
3. **Verify:** The user opens the generated Markdown text file. They perform a quick "find and replace" if a speaker was misidentified (e.g., if the DM was labeled SPEAKER\_02 instead of SPEAKER\_00).  
4. **Index:** The script uploads the verified file to Open WebUI.  
5. **Query:** The DM opens Open WebUI, selects the "Campaign" model, and asks: "Summarize the events involving the cultists in the last session." The RAG system retrieves the relevant dialogue chunks, and the LLM generates a summary.

### **7.3 Maintaining Consistency**

Over time, the database grows.

* **Hybrid Search:** Pure vector search can miss specific names (e.g., "K'r'shh"). Open WebUI supports Hybrid Search (BM25 \+ Vector). Enabling this is critical for D\&D to find proper nouns accurately.36  
* **Summarization Hierarchies:** To improve retrieval, it is best practice to store *both* the raw transcript *and* an AI-generated summary of the session. When asking high-level questions ("What is the main plot?"), the summary chunks are retrieved. When asking specific questions ("What exact words did the villain use?"), the transcript chunks are retrieved.37

## **8\. Conclusion: The "Second Brain" for Dungeon Masters**

The architecture described herein leverages the massive parallel processing power of the RTX 3090 to solve the age-old problem of campaign memory. By chaining **WhisperX** for high-fidelity transcription, **Pyannote** for speaker differentiation, and **Open WebUI** for semantic retrieval, a Dungeon Master can build a self-hosted, private, and indefinitely scalable archive of their game.

This system moves beyond simple "transcription" to true "knowledge management." It respects the privacy of the players by keeping voice data local, avoids the recurring costs of cloud APIs (which can run $5-$10 per session for this duration), and utilizes the specific hardware advantage of the 24GB VRAM to run uncompromised, professional-grade models. The result is a living wiki that listens, learns, and recalls every roll of the dice.

### ---

**Appendix: Summary of Tools**

| Component | Tool Choice | Role | Justification |
| :---- | :---- | :---- | :---- |
| **GPU** | RTX 3090 | Hardware | 24GB VRAM allows Large-v3 Whisper \+ 32B LLM (sequentially). |
| **ASR** | WhisperX | Transcription | Superior timestamp accuracy (word-level) and hallucination control vs. vanilla Whisper. |
| **Diarization** | Pyannote 3.1 | Speaker ID | State-of-the-art accuracy; integrated into WhisperX. |
| **Vector DB** | ChromaDB | Storage | Lightweight, embedded, native support in Open WebUI. |
| **Interface** | Open WebUI | Frontend | Native RAG pipeline, citation support, Ollama integration. |
| **Inference** | Ollama | LLM Runner | Easy model management, simple API, highly optimized. |
| **Model** | Qwen-2.5 32B | Intelligence | Best balance of reasoning/size for 24GB cards; beats Llama-3 8B. |

## **9\. References and Citations**

The architectural decisions in this report are supported by the following analysis of current technology standards as of late 2025/early 2026:

* **WhisperX capabilities and VAD integration:**.5  
* **Pyannote diarization performance and integration:**.11  
* **RTX 3090 VRAM constraints and quantization:**.1  
* **Open WebUI RAG features and API:**.24  
* **D\&D specific workflow (TASMAS, Craig, etc.):**.37  
* **Vector Database selection (Chroma vs others):**.6

#### **Works cited**

1. Ollama VRAM Requirements: Complete 2026 Guide to GPU Memory for Local LLMs, accessed February 7, 2026, [https://localllm.in/blog/ollama-vram-requirements-for-local-llms](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)  
2. Context Kills VRAM: How to Run LLMs on consumer GPUs | by Lyx | Medium, accessed February 7, 2026, [https://medium.com/@lyx\_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632](https://medium.com/@lyx_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632)  
3. Is RTX 3090 still the only king of price/performance for running local LLMs and diffusion models? (plus some rant) \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1j6vmke/is\_rtx\_3090\_still\_the\_only\_king\_of/](https://www.reddit.com/r/LocalLLaMA/comments/1j6vmke/is_rtx_3090_still_the_only_king_of/)  
4. The Best GPUs for Local LLM Inference in 2025, accessed February 7, 2026, [https://localllm.in/blog/best-gpus-llm-inference-2025](https://localllm.in/blog/best-gpus-llm-inference-2025)  
5. murtaza-nasir/whisperx-asr-service \- GitHub, accessed February 7, 2026, [https://github.com/murtaza-nasir/whisperx-asr-service](https://github.com/murtaza-nasir/whisperx-asr-service)  
6. How to Build a Local RAG App with Ollama and ChromaDB in the R Programming Language \- freeCodeCamp, accessed February 7, 2026, [https://www.freecodecamp.org/news/build-a-local-rag-app-with-ollama-and-chromadb-in-r/](https://www.freecodecamp.org/news/build-a-local-rag-app-with-ollama-and-chromadb-in-r/)  
7. naveedn/audio-transcriber: Seamlessly merge multi-track audio into a single unified transcript \- perfect with Craig.chat \- GitHub, accessed February 7, 2026, [https://github.com/naveedn/audio-transcriber](https://github.com/naveedn/audio-transcriber)  
8. Choosing between Whisper variants: faster-whisper, insanely-fast-whisper, WhisperX, accessed February 7, 2026, [https://modal.com/blog/choosing-whisper-variants](https://modal.com/blog/choosing-whisper-variants)  
9. I compared the different open source whisper packages for long-form transcription \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1brqwun/i\_compared\_the\_different\_open\_source\_whisper/](https://www.reddit.com/r/LocalLLaMA/comments/1brqwun/i_compared_the_different_open_source_whisper/)  
10. Unlock the Power of Audio Data: Advanced Transcription and Diarization with Whisper, WhisperX, and... | Towards Data Science, accessed February 7, 2026, [https://towardsdatascience.com/unlock-the-power-of-audio-data-advanced-transcription-and-diarization-with-whisper-whisperx-and-ed9424307281/](https://towardsdatascience.com/unlock-the-power-of-audio-data-advanced-transcription-and-diarization-with-whisper-whisperx-and-ed9424307281/)  
11. WhisperX: Automatic Speech Recognition with Word-level Timestamps (& Diarization) \- GitHub, accessed February 7, 2026, [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)  
12. Comparing WhisperX and Faster-Whisper on RunPod: Speed, Accuracy, and Optimization · Issue \#1066 \- GitHub, accessed February 7, 2026, [https://github.com/m-bain/whisperX/issues/1066](https://github.com/m-bain/whisperX/issues/1066)  
13. davidamacey/OpenTranscribe: Self-hosted AI-powered transcription platform with speaker diarization, search, and collaboration features. Built with Svelte, FastAPI, and Docker for easy deployment. \- GitHub, accessed February 7, 2026, [https://github.com/davidamacey/OpenTranscribe](https://github.com/davidamacey/OpenTranscribe)  
14. WhisperX vs Competitors: AI Transcription Comparison 2026 \- BrassTranscripts, accessed February 7, 2026, [https://brasstranscripts.com/blog/whisperx-vs-competitors-accuracy-benchmark](https://brasstranscripts.com/blog/whisperx-vs-competitors-accuracy-benchmark)  
15. Best Speaker Diarization Models Compared \[2026\] \- BrassTranscripts, accessed February 7, 2026, [https://brasstranscripts.com/blog/speaker-diarization-models-comparison](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)  
16. Speaker diarization using Whisper ASR and Pyannote | by Ritesh \- Medium, accessed February 7, 2026, [https://medium.com/@xriteshsharmax/speaker-diarization-using-whisper-asr-and-pyannote-f0141c85d59a](https://medium.com/@xriteshsharmax/speaker-diarization-using-whisper-asr-and-pyannote-f0141c85d59a)  
17. WhisperX is only accurate on the first 10 words. Any Tips? : r/speechtech \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/speechtech/comments/1q09mhw/whisperx\_is\_only\_accurate\_on\_the\_first\_10\_words/](https://www.reddit.com/r/speechtech/comments/1q09mhw/whisperx_is_only_accurate_on_the_first_10_words/)  
18. Accuracy of Transcription: Filler words · Issue \#293 · m-bain/whisperX \- GitHub, accessed February 7, 2026, [https://github.com/m-bain/whisperX/issues/293](https://github.com/m-bain/whisperX/issues/293)  
19. Whisper prompting guide \- OpenAI for developers, accessed February 7, 2026, [https://developers.openai.com/cookbook/examples/whisper\_prompting\_guide/](https://developers.openai.com/cookbook/examples/whisper_prompting_guide/)  
20. From Milvus to Qdrant: The Ultimate Guide to the Top 10 Open-Source Vector Databases | by TechLatest.Net \- Medium, accessed February 7, 2026, [https://medium.com/@techlatest.net/from-milvus-to-qdrant-the-ultimate-guide-to-the-top-10-open-source-vector-databases-7d2805ed8970](https://medium.com/@techlatest.net/from-milvus-to-qdrant-the-ultimate-guide-to-the-top-10-open-source-vector-databases-7d2805ed8970)  
21. How to Store and Query Transcription with Speaker Labels in a Vector Database for Q\&A?, accessed February 7, 2026, [https://www.reddit.com/r/vectordatabase/comments/1h2jcla/how\_to\_store\_and\_query\_transcription\_with\_speaker/](https://www.reddit.com/r/vectordatabase/comments/1h2jcla/how_to_store_and_query_transcription_with_speaker/)  
22. Chunk Twice, Retrieve Once: RAG Chunking Strategies Optimized for Different Content Types | Dell Technologies Info Hub, accessed February 7, 2026, [https://infohub.delltechnologies.com/en-uk/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/](https://infohub.delltechnologies.com/en-uk/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/)  
23. Effective Chunking Strategies for RAG \- Cohere Documentation, accessed February 7, 2026, [https://docs.cohere.com/page/chunking-strategies](https://docs.cohere.com/page/chunking-strategies)  
24. Features | Open WebUI, accessed February 7, 2026, [https://docs.openwebui.com/features/](https://docs.openwebui.com/features/)  
25. Retrieval Augmented Generation (RAG) \- Open WebUI, accessed February 7, 2026, [https://docs.openwebui.com/features/rag/](https://docs.openwebui.com/features/rag/)  
26. Local web UI with actually decent RAG? : r/LocalLLaMA \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1cm6u9f/local\_web\_ui\_with\_actually\_decent\_rag/](https://www.reddit.com/r/LocalLLaMA/comments/1cm6u9f/local_web_ui_with_actually_decent_rag/)  
27. Best Overall LLM for 24GB VRAM : r/LocalLLaMA \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1e6yad6/best\_overall\_llm\_for\_24gb\_vram/](https://www.reddit.com/r/LocalLLaMA/comments/1e6yad6/best_overall_llm_for_24gb_vram/)  
28. RAG vs. long-context LLMs: A side-by-side comparison \- Meilisearch, accessed February 7, 2026, [https://www.meilisearch.com/blog/rag-vs-long-context-llms](https://www.meilisearch.com/blog/rag-vs-long-context-llms)  
29. How to Use Ollama with Docker \- OneUptime, accessed February 7, 2026, [https://oneuptime.com/blog/post/2026-01-28-ollama-docker/view](https://oneuptime.com/blog/post/2026-01-28-ollama-docker/view)  
30. API Endpoints \- Open WebUI, accessed February 7, 2026, [https://docs.openwebui.com/getting-started/api-endpoints/](https://docs.openwebui.com/getting-started/api-endpoints/)  
31. Using API to add document to Knowledge? \- OpenWebUI \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/OpenWebUI/comments/1ka4gmp/using\_api\_to\_add\_document\_to\_knowledge/](https://www.reddit.com/r/OpenWebUI/comments/1ka4gmp/using_api_to_add_document_to_knowledge/)  
32. Whisper Speaker Diarization: Python Tutorial 2026 \- BrassTranscripts, accessed February 7, 2026, [https://brasstranscripts.com/blog/whisper-speaker-diarization-guide](https://brasstranscripts.com/blog/whisper-speaker-diarization-guide)  
33. Automatic Speaker Recognition in MacWhisper, accessed February 7, 2026, [https://macwhisper.helpscoutdocs.com/article/32-automatic-speaker-recognition-in-macwhisper](https://macwhisper.helpscoutdocs.com/article/32-automatic-speaker-recognition-in-macwhisper)  
34. Feature suggestion: editing speaker names in the body of the transcript : r/MacWhisper, accessed February 7, 2026, [https://www.reddit.com/r/MacWhisper/comments/1pd2ws9/feature\_suggestion\_editing\_speaker\_names\_in\_the/](https://www.reddit.com/r/MacWhisper/comments/1pd2ws9/feature_suggestion_editing_speaker_names_in_the/)  
35. Creating Very High-Quality Transcripts with Open-Source Tools: An 100% automated workflow guide : r/LocalLLaMA \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1g2vhy3/creating\_very\_highquality\_transcripts\_with/](https://www.reddit.com/r/LocalLLaMA/comments/1g2vhy3/creating_very_highquality_transcripts_with/)  
36. Part 2: Getting Started with Local AI \- Open WebUI Documents and Tools | by John Wong, accessed February 7, 2026, [https://medium.com/@able\_wong/getting-started-with-local-ai-open-webui-documents-and-tools-part-2-5f8f9c67a414](https://medium.com/@able_wong/getting-started-with-local-ai-open-webui-documents-and-tools-part-2-5f8f9c67a414)  
37. KaddaOK/TASMAS: Free open-source transcriber and summarizer for file-per-speaker recordings, such as Discord calls recorded by the Craig bot \- GitHub, accessed February 7, 2026, [https://github.com/KaddaOK/TASMAS](https://github.com/KaddaOK/TASMAS)  
38. I record our sessions and use AI to make a summary : r/DMAcademy \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/DMAcademy/comments/1jvej7n/i\_record\_our\_sessions\_and\_use\_ai\_to\_make\_a\_summary/](https://www.reddit.com/r/DMAcademy/comments/1jvej7n/i_record_our_sessions_and_use_ai_to_make_a_summary/)  
39. Level Up Your D\&D Session Notes: Useful AI Prompts & Cleanup Scripts \- Medium, accessed February 7, 2026, [https://medium.com/@brandonharris\_12357/level-up-your-d-d-session-notes-useful-ai-prompts-cleanup-scripts-ca959de9a541](https://medium.com/@brandonharris_12357/level-up-your-d-d-session-notes-useful-ai-prompts-cleanup-scripts-ca959de9a541)  
40. The Top 7 Vector Databases in 2026 \- DataCamp, accessed February 7, 2026, [https://www.datacamp.com/blog/the-top-5-vector-databases](https://www.datacamp.com/blog/the-top-5-vector-databases)