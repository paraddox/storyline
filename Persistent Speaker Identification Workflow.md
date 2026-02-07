# **Persistent Speaker Identity in Long-Form Audio Archives: A Comprehensive Implementation Framework Using WhisperX and Pyannote.audio**

## **1\. Executive Summary**

The proliferation of high-fidelity digital audio recording has created a paradox in information management: while capturing spoken content is trivial, retrieving specific information attributed to specific individuals remains a complex computational challenge. In domains ranging from legal depositions and corporate board meetings to the increasingly popular archival of tabletop role-playing game (RPG) sessions, the core utility of an audio archive depends on the ability to answer two fundamental questions: "What was said?" and "Who said it?".

Modern Automatic Speech Recognition (ASR) systems, epitomized by OpenAI’s Whisper architecture, have largely solved the first question, achieving near-human parity in transcription accuracy across diverse languages and acoustic environments. However, the second question—persistent speaker identification across multiple distinct recording sessions—remains a significant hurdle. Standard diarization pipelines can distinguish between *Speaker A* and *Speaker B* within a single file but lack the inherent memory to recognize that *Speaker A* in Session 1 is the same individual as *Speaker B* in Session 10\. This "amnesia" necessitates a sophisticated architectural intervention that bridges the gap between ephemeral diarization labels and persistent identity verification.

This report presents an exhaustive technical framework for implementing a persistent speaker recognition system utilizing **WhisperX** for transcription and forced alignment, and **pyannote.audio** for speaker diarization and embedding extraction. Designed for deployment on high-performance consumer hardware—specifically the NVIDIA RTX 3090—this architecture establishes a local, privacy-centric pipeline capable of transforming raw audio into structured, queryable knowledge bases.

The analysis proceeds through a rigorous examination of the underlying neural architectures, detailing the mathematical foundations of speaker embeddings (ECAPA-TDNN) and the algorithmic logic of cosine similarity matching. It further explores the integration of this structured data into **Vector Databases** (such as ChromaDB) to enable Retrieval-Augmented Generation (RAG), allowing for context-aware semantic search. Special attention is given to the unique challenges of the "Dungeons & Dragons" (D\&D) use case, where fantasy vocabulary, overlapping speech, and character-switching present stress tests for standard ASR models. By synthesizing insights from recent developments in self-hosted AI, this report provides a blueprint for constructing a "digital scribe" capable of preserving the narrative continuity of long-form oral histories.1

## ---

**2\. Theoretical Framework: The Evolution of Neural Audio Processing**

To understand the architectural decisions behind the proposed pipeline, one must first appreciate the evolution of the underlying technologies. The shift from statistical modeling to deep learning has fundamentally altered the landscape of speech processing, moving from brittle, hand-crafted feature extraction to robust, end-to-end neural representations.

### **2.1 From HMMs to End-to-End Transformers**

For decades, ASR was dominated by Hidden Markov Models (HMMs) and Gaussian Mixture Models (GMMs). These systems relied on complex phonetic dictionaries and language models to probability-match acoustic signals to text. They were computationally efficient but struggled with noise, accents, and diverse vocabularies.

The advent of the **Transformer** architecture, initially designed for Natural Language Processing (NLP), revolutionized ASR. OpenAI’s **Whisper** model represents the pinnacle of this approach. Trained on 680,000 hours of multilingual, multitask supervision, Whisper learns to map raw audio directly to a sequence of text tokens.4 Unlike its predecessors, Whisper does not require a separate phoneme alignment step during training; it learns a robust mapping between the audio spectrogram and the text, effectively bypassing the need for traditional acoustic modeling.

However, Whisper’s architecture is fundamentally an **encoder-decoder** model designed for sequence-to-sequence tasks. It excels at predicting the next token in a transcript but has no inherent mechanism for understanding *who* is speaking. The model "hears" the audio as a single stream of information, occasionally hallucinating speaker turns or failing to distinguish between overlapping voices.2 This limitation necessitates the integration of a specialized secondary system for speaker analysis.

### **2.2 The "Who Spoke When" Problem: Diarization**

Speaker diarization is the process of partitioning an audio stream into homogeneous segments according to the speaker identity. It answers the question, "who spoke when?" without necessarily knowing the real-world identity of the speakers.2

Modern diarization pipelines, such as **pyannote.audio**, utilize a modular approach:

1. **Voice Activity Detection (VAD):** Separating speech from non-speech (silence, noise).  
2. **Segmentation:** Detecting instantaneous changes in speaker identity.  
3. **Embedding:** Mapping audio segments to a vector space where distance corresponds to speaker similarity.  
4. **Clustering:** Grouping these vectors into clusters, where each cluster represents a unique (but anonymous) speaker.4

The critical distinction in this report is between **Diarization** and **Identification**. Diarization is *relative*; it knows that Segment A and Segment B were spoken by the same person. Identification is *absolute*; it knows that Segment A was spoken by "The Dungeon Master." Bridging this gap requires the persistence of speaker embeddings across sessions.2

### **2.3 Mathematical Foundations of Speaker Embeddings**

The core of persistent identification lies in the **Speaker Embedding**. An embedding is a high-dimensional vector (typically 192 or 512 dimensions) that encapsulates the unique acoustic characteristics of a voice—pitch, timbre, cadence, and formant structure—while abstracting away the linguistic content (what is being said).4

**ECAPA-TDNN Architecture:**

Pyannote often relies on the **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) architecture. This model is designed to capture temporal context at multiple scales.

* **Time Delay Neural Networks (TDNN):** Unlike standard Recurrent Neural Networks (RNNs) that process data sequentially, TDNNs look at a fixed window of past and future frames, allowing them to capture local acoustic dependencies efficiently.  
* **Channel Attention:** This mechanism allows the network to focus on the most informative frequency channels, enhancing robustness against noise.9

The output of this network is a vector ![][image1]. If the network is trained correctly (usually on massive datasets like VoxCeleb), the Euclidean distance or Cosine similarity between two vectors ![][image2] and ![][image3] should directly correlate with the likelihood that they belong to the same speaker.

![][image4]  
This mathematical property allows us to treat speaker recognition as a **Nearest Neighbor** search problem in a vector space, a concept central to the implementation strategy detailed in Section 6\.7

## ---

**3\. Hardware Infrastructure: The Self-Hosted Advantage**

While cloud-based APIs (Google Speech-to-Text, AWS Transcribe) offer convenience, they impose significant costs and privacy concerns, particularly for long-duration recordings like D\&D sessions which can run for 4-6 hours. A self-hosted approach using consumer high-end hardware offers a compelling alternative, providing data sovereignty and zero marginal cost per hour of audio processed.

### **3.1 The GPU Substrate: NVIDIA RTX 3090**

The implementation described in this report is optimized for the **NVIDIA RTX 3090**. With 24GB of GDDR6X VRAM and 10,496 CUDA cores, this card sits in the "sweet spot" for local AI workloads.

* **VRAM Capacity:** The 24GB buffer is critical. Audio transformers are memory-intensive. Loading Whisper Large-v3 requires approximately 10-12GB of VRAM in FP16 precision. A typical diarization pipeline requires an additional 2-4GB. Concurrent execution of these models alongside an LLM for summarization (e.g., Llama-3-8B) is possible but requires careful memory management.12  
* **Memory Bandwidth:** The RTX 3090's 936 GB/s memory bandwidth ensures that the massive matrices involved in Transformer attention mechanisms can be moved in and out of the compute units rapidly, which is essential for batch processing audio segments.14

### **3.2 CUDA and Precision Strategy**

To maximize the throughput on the RTX 3090, the pipeline leverages **Mixed Precision Training/Inference**.

* **FP16 (Half Precision):** Most inference operations should be conducted in FP16. This effectively doubles the throughput compared to FP32 and halves the VRAM footprint with negligible loss in accuracy for audio tasks.15  
* **Int8 Quantization:** For even greater efficiency, specifically with WhisperX, using 8-bit quantization (int8) can reduce the model size further, allowing for larger batch sizes. This is particularly useful if the system is also serving a Vector Database or LLM simultaneously.16

### **3.3 Storage Requirements**

Audio processing is I/O intensive.

* **NVMe SSDs:** Pre-processing steps like converting MP3 to WAV (16kHz PCM) generate large temporary files. A 4-hour D\&D session in uncompressed WAV is approximately 2.5GB. Fast NVMe storage prevents the GPU from stalling while waiting for data.17  
* **Database Storage:** Storing embeddings is cheap (a 192-dim vector is negligible), but the associated metadata and transcripts for RAG can grow. A dedicated SSD for the Vector Database (e.g., ChromaDB persistence directory) ensures low-latency retrieval.18

## ---

**4\. Architectural Design: The Offline Pipeline**

The proposed solution is not a monolithic application but a linear Directed Acyclic Graph (DAG) pipeline. This modularity allows for easier debugging and component upgrades (e.g., swapping the embedding model without breaking the transcriber).

### **4.1 The Pipeline Stages**

1. **Ingestion:** Raw audio files (MP3, M4A, FLAC) are normalized to the format expected by the models (16kHz Mono WAV).  
2. **Voice Activity Detection (VAD):** Silence is removed to optimize processing time.  
3. **Transcription (ASR):** Speech is converted to text.  
4. **Forced Alignment:** Text is aligned to audio timestamps.  
5. **Diarization:** Speakers are segmented and clustered locally.  
6. **Embedding Extraction:** Audio segments are converted to vectors.  
7. **Identification:** Vectors are matched against the "Voice Bank."  
8. **Output Generation:** Structured data (JSON/Database) is produced.

### **4.2 Why WhisperX?**

The original Whisper implementation processes audio in sequential 30-second windows. This has two major drawbacks:

1. **Slowness:** It cannot fully utilize the massive parallelism of the RTX 3090\.  
2. **Hallucinations:** In long periods of silence or background noise (common in D\&D sessions), the decoder can get "stuck," repeating phrases or hallucinating text.

**WhisperX** solves this by introducing a pre-processing VAD step. It cuts the audio into valid speech segments *first*, then batches these segments and feeds them to Whisper in parallel. This **Batched Inference** can achieve speeds of 60x-70x real-time on an RTX 3090 (processing an hour of audio in under a minute).19 Furthermore, WhisperX includes a post-processing **Phoneme Alignment** step using Wav2Vec2, which corrects the sometimes loose timestamps of the original Whisper model, providing the precision necessary for accurate speaker attribution.19

## ---

**5\. Implementation Phase 1: The Transcription Engine**

The first operational phase involves setting up the transcription engine. This section details the configuration of WhisperX for maximum accuracy and speed.

### **5.1 Environmental Setup**

The environment relies on Python 3.10+ and PyTorch with CUDA support.

* **Dependencies:** torch, torchaudio, whisperx.  
* **FFmpeg:** Essential for the audio decoding and conversion.

Bash

pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu118  
pip install git+https://github.com/m-bain/whisperX.git

Note: Installing directly from GitHub is often recommended for WhisperX to get the latest VAD fixes.19

### **5.2 Optimizing the Transcription Call**

The core transcription script must handle memory management explicitly to avoid VRAM fragmentation.

**Batch Size:** On an RTX 3090, a batch size of 16 or 32 is optimal. Increasing it further yields diminishing returns and risks Out-Of-Memory (OOM) errors.

**Compute Type:** float16 is the standard. If VRAM is tight (e.g., if running a local LLM simultaneously), int8 can be used.

Python

import whisperx  
import gc  
import torch

device \= "cuda"  
audio\_file \= "dnd\_session\_01.wav"  
batch\_size \= 16   
compute\_type \= "float16" \# Change to "int8" if VRAM is constrained

\# 1\. Transcribe with original whisper (batched)  
model \= whisperx.load\_model("large-v2", device, compute\_type=compute\_type)  
audio \= whisperx.load\_audio(audio\_file)  
result \= model.transcribe(audio, batch\_size=batch\_size)

\# Free GPU resources immediately after transcription  
del model  
gc.collect()  
torch.cuda.empty\_cache()

The explicit deletion of the model and clearing of the CUDA cache is a critical pattern for the RTX 3090\. It ensures that the full 24GB is available for the subsequent alignment and diarization models, preventing the "CUDA out of memory" crashes common in multi-model pipelines.15

### **5.3 Handling Fantasy Vocabulary (Prompt Engineering)**

A major challenge in D\&D transcription is the prevalence of invented proper nouns (e.g., "Mordenkainen," "Faerun," "Tiamat"). Standard models will phonetically map these to common English words (e.g., "Tea a mat").

**Initial Prompt:** Whisper supports an initial\_prompt parameter. This is not an instruction (like in ChatGPT) but a "prefix" to the audio context. By seeding this prompt with a list of proper nouns likely to appear in the session, the model's probability distribution is biased toward those tokens.21

* **Strategy:** Maintain a separate text file vocabulary.txt for the campaign. Read this file and inject it into the prompt.  
* **Prompt Construction:** "The following is a transcript of a Dungeons & Dragons session containing names like Mordenkainen, Faerun, Tiamat, and Strahd."  
* **Limitations:** The prompt influences the first 30-second window most strongly. However, because WhisperX batches segments, the "context" is reset for each batch. This is a known limitation, though some forks of Whisper allow forcing the prompt for every segment.22

### **5.4 Forced Alignment**

Whisper's timestamps are generated at the *utterance* level and can be imprecise. WhisperX fixes this by loading a language-specific alignment model (based on Wav2Vec2).

Python

\# 2\. Align whisper output  
model\_a, metadata \= whisperx.load\_align\_model(language\_code=result\["language"\], device=device)  
result \= whisperx.align(result\["segments"\], model\_a, metadata, audio, device, return\_char\_alignments=False)

\# Free alignment model resources  
del model\_a  
gc.collect()  
torch.cuda.empty\_cache()

This step is non-negotiable for speaker identification. Without precise word-level timestamps, a short interjection by the Dungeon Master overlapping a Player's speech might be misattributed, confusing the downstream RAG system.20

## ---

**6\. Implementation Phase 2: The Diarization Engine**

With accurate text and timestamps, the next step is to determine *who* spoke. This utilizes **pyannote.audio**.

### **6.1 Pyannote Integration via WhisperX**

WhisperX provides a convenient wrapper around pyannote, but understanding the underlying requirements is essential. You must accept the user agreements for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on Hugging Face and generate an access token.25

Python

\# 3\. Diarize  
diarize\_model \= whisperx.DiarizationPipeline(use\_auth\_token="HF\_TOKEN", device=device)  
diarization\_result \= diarize\_model(audio)

\# 4\. Assign Speakers to Words  
final\_result \= whisperx.assign\_word\_speakers(diarization\_result, result)

### **6.2 The "Local" Label Limitation**

At this stage, the output final\_result contains segments labeled SPEAKER\_00, SPEAKER\_01, etc.

* **Crucial Insight:** These labels are valid *only* for this specific audio file. SPEAKER\_00 in session\_1.wav is not necessarily SPEAKER\_00 in session\_2.wav.  
* **The Cluster Problem:** Pyannote performs clustering (usually Agglomerative Hierarchical Clustering) on the embeddings generated *within the file*. It does not inherently look "outside" the file.  
* **Resolution:** To achieve persistent identity, we must intercept the pipeline here and introduce our own **Identification Layer**. We need to extract the embeddings that Pyannote used (or regenerate them) and compare them against a "Master List".2

## ---

**7\. Implementation Phase 3: Persistent Speaker Recognition (The Core Solution)**

This section details the custom Python logic required to build the "Voice Bank" and perform cross-file matching. This is the "missing link" in standard tutorials.

### **7.1 Building the Voice Bank (Enrollment)**

The system needs a ground truth. This requires a manual "Enrollment" phase.

**Data Curation:**

1. Extract 3-5 clean audio clips (30-60 seconds each) for every known participant (e.g., DM, Player\_A, Player\_B).  
2. Ensure these clips capture the speaker's dynamic range. For a D\&D player, this should include their "normal" voice and their "character" voice if they perform one.  
3. Store these in a structured directory: data/enrollment/{speaker\_name}/{clip\_id}.wav.27

**Enrollment Script Logic:**

Using pyannote.audio.Inference, we extract an embedding for each clip.

* **Model:** pyannote/embedding (typically pre-trained on VoxCeleb).  
* **Vector Averaging:** For each speaker, we can compute a "centroid" (average) vector of their clips. However, a more robust approach (discussed in Section 7.2) involves clustering.

Python

from pyannote.audio import Inference  
inference \= Inference("pyannote/embedding", use\_auth\_token="HF\_TOKEN")

def enroll\_speaker(audio\_path):  
    \# 'window="whole"' extracts one embedding for the entire clip  
    embedding \= inference(audio\_path, window="whole")   
    return embedding

### **7.2 Handling Variance: The "k-Centroids" Approach**

A single average vector often fails to capture a speaker's variability (e.g., whispering vs. shouting). Instead of one vector per speaker, we should store ![][image5] **representative vectors**.

* **Method:** Collect all enrollment embeddings for "Player A." Perform k-means clustering (where ![][image6] or ![][image7]) on these embeddings. Store the centroids of these clusters.  
* **Benefit:** This creates a "multi-modal" definition of the speaker, allowing the system to recognize them whether they are speaking calmly or excitedly.9

### **7.3 The Matching Algorithm**

When processing a new session, we iterate through the diarized segments. For each anonymous speaker (e.g., SPEAKER\_00):

1. **Extract:** Isolate all audio segments attributed to SPEAKER\_00 in the new file.  
2. **Embed:** Compute the average embedding for SPEAKER\_00.  
3. **Compare:** Calculate the **Cosine Distance** between SPEAKER\_00's vector and every vector in the Voice Bank.

**Cosine Similarity Formula Implementation:**

We utilize scipy.spatial.distance.cdist or torch.nn.CosineSimilarity for efficient computation.

Python

from scipy.spatial.distance import cdist

\# unknown\_embedding: (1, 192\)  
\# voice\_bank\_embeddings: (N, 192\) where N is total known speakers

distances \= cdist(unknown\_embedding, voice\_bank\_embeddings, metric="cosine")  
min\_dist\_index \= np.argmin(distances)  
min\_distance \= distances\[0, min\_dist\_index\]

### **7.4 Thresholding Strategy**

The decision boundary is critical.

* **If min\_distance \< Threshold:** Match found. Assign the known name.  
* **If min\_distance \> Threshold:** Speaker is "Unknown."

**Setting the Threshold:** There is no "magic number," but empirical data suggests a cosine distance threshold between **0.3 and 0.6** (or similarity of 0.7-0.4) depending on the model. For pyannote/embedding, a distance of **0.5** is a reasonable starting point. If the threshold is too strict (e.g., 0.2), true speakers will be labeled "Unknown." If too loose (e.g., 0.8), distinct speakers may be merged.8

* **Adaptive Thresholding:** Advanced implementations can calibrate the threshold per speaker during enrollment, calculating the intra-speaker variance (how much Player A differs from themselves) and setting the threshold just outside that range.

## ---

**8\. Data Engineering & RAG Integration**

The ultimate goal is not just a transcript, but a searchable database. This requires transforming the linear transcript into a structured knowledge base.

### **8.1 Vector Database Schema Design**

For a RAG system (like ChromaDB) to effectively answer questions like *"What loot did we get in the dragon's lair?"*, the data must be chunked and indexed with rich metadata.

**Why Standard Chunking Fails:**

Standard RAG chunking (e.g., "split every 500 tokens") destroys conversational context.

* *Transcript:*  
  * **DM:** "You see a chest."  
  * **Player:** "I open it."  
  * **DM:** "It mimics you."  
* *Bad Chunk 1:* "You see a chest. I open it."  
* *Bad Chunk 2:* "It mimics you."  
  Chunk 2 has lost the antecedent (the chest). The LLM won't know *what* mimics the player.

**Dialogue-Aware Chunking Schema:** We must use a **Semantic/Dialogue-Aware** strategy.28

1. **Grouping:** Group consecutive turns into logical blocks (e.g., based on topic shifts or time gaps).  
2. **Context Injection:** Prepend the Speaker Name to the text *before* embedding.  
   * *Raw:* "I cast Fireball."  
   * *Embedded Text:* "Garrick the Wizard says: I cast Fireball." This ensures the vector representation encodes the *agent* of the action.30

**ChromaDB Metadata Fields:**

The following schema allows for powerful filtering:

| Field Name | Data Type | Purpose |
| :---- | :---- | :---- |
| chunk\_id | String | Unique UUID. |
| session\_id | String | E.g., "Campaign\_2\_Session\_04". |
| speaker\_id | String | The identified name (e.g., "DM", "Garrick"). |
| timestamp\_start | Float | Seconds from start of file. |
| text\_content | String | The spoken text. |
| embedding | Vector | 1536-dim (OpenAI) or 768-dim (HuggingFace). |
| entity\_tags | List | Extracted via LLM (e.g.,). |

### **8.2 RAG Retrieval Logic**

With this schema, we can perform **Hybrid Search**:

* *Query:* "What did the DM say about the villain's weakness?"  
* *Filter:* where={"speaker\_id": "DM"}  
* *Vector Search:* Semantic match for "villain weakness."

This dramatically reduces "noise" (e.g., players speculating incorrectly about the weakness) and retrieves only the authoritative statements from the Dungeon Master.31

## ---

**9\. Performance Optimization and Constraints**

Running this pipeline on a single RTX 3090 requires rigorous resource management.

### **9.1 VRAM Partitioning**

The 24GB VRAM is ample but finite.

* **Sequential Loading:** Do not keep the Whisper model in memory while running Pyannote. The Python script should strictly enforce garbage collection.  
  Python  
  del whisper\_model  
  gc.collect()  
  torch.cuda.empty\_cache()  
  \# Now load Pyannote

* **Quantization:** Use int8 for WhisperX if you plan to run a local LLM (like Llama-3-8B) for summarization in the same workflow. This reduces Whisper's footprint from \~10GB to \~4GB.16

### **9.2 Batch Size Tuning**

For the RTX 3090:

* **WhisperX:** Batch size 16 is safe. Batch size 32 is often possible with int8 and provides a 10-15% speedup.  
* **Pyannote:** Processing is less memory-intensive but more CPU-bound during the clustering phase. Ensure the system has high single-core CPU performance for the clustering math.33

## ---

**10\. Case Study: The "Dungeons & Dragons" Specifics**

The D\&D use case provides a perfect stress test for this system due to its chaotic acoustic nature.

### **10.1 The "Overlapping Speech" Problem**

D\&D sessions are full of laughter, interruptions, and shouting.

* **WhisperX Limitation:** Whisper is decent at transcribing the *dominant* speaker but often drops the secondary speaker in overlap scenarios.  
* **Pyannote Capability:** Pyannote has an overlapped-speech-detection model.  
* **Strategy:** While we cannot easily transcribe two people talking at once with a single microphone, we can use the overlap detection to flag these segments in the transcript as \`\`. This alerts the reader (or the RAG system) that information might be missing or unreliable in that timestamp.25

### **10.2 Character vs. Player Identity**

Players often speak in character.

* **Acoustic Variation:** A player using a "Gravelly Dwarf Voice" produces a significantly different embedding than their normal voice.  
* **Enrollment Solution:** Enroll *both* voices.  
  * Sample 1: Player A (Normal).  
  * Sample 2: Player A (Dwarf Voice).  
  * Both samples map to the ID Player\_A. This allows the system to recognize the player regardless of the "mode" they are in, unifying the transcript under a single identity.30

## ---

**11\. Conclusion**

The implementation of a persistent speaker recognition system using **WhisperX** and **pyannote.audio** on an **RTX 3090** represents a significant leap forward for local, privacy-focused audio archiving. By moving beyond the ephemeral labeling of standard diarization and building a robust **Voice Bank** based on **ECAPA-TDNN embeddings**, users can unlock the semantic depth of their long-form audio.

This architecture solves the "who said what" problem not through a single magic model, but through a carefully orchestrated pipeline of specialized tools: VAD for segmentation, Whisper for transcription, Wav2Vec2 for alignment, and Pyannote for embedding. When coupled with a **Vector Database** and **Dialogue-Aware Chunking**, this system transforms a chaotic folder of MP3s into a structured, queryable history—a true digital memory for the complex narratives of business, law, and gaming alike.

The future of this technology lies in tighter integration with Large Language Models, where the LLM not only summarizes the text but actively assists in the speaker identification process by analyzing the linguistic patterns (e.g., recognizing that only the DM says "Roll for initiative"), creating a multimodal feedback loop that approaches human-level comprehension of the auditory scene.

### ---

**References & Implementation Sources**

* **WhisperX & Alignment:** 19  
* **Pyannote & Embeddings:** 2  
* **Speaker Recognition Logic:** 7  
* **Vector Database & RAG:** 31  
* **Hardware & VRAM:** 13  
* **D\&D Specifics:** 36

#### **Works cited**

1. This self-hosted tool turns audio into podcast-style Obsidian notes \- XDA Developers, accessed February 7, 2026, [https://www.xda-developers.com/this-self-hosted-tool-turns-audio-into-podcast-style-obsidian-notes/](https://www.xda-developers.com/this-self-hosted-tool-turns-audio-into-podcast-style-obsidian-notes/)  
2. What is Speaker Diarization? \- pyannoteAI, accessed February 7, 2026, [https://www.pyannote.ai/blog/what-is-speaker-diarization](https://www.pyannote.ai/blog/what-is-speaker-diarization)  
3. Feature request: Reuse speaker embeddings across multiple audio ..., accessed February 7, 2026, [https://github.com/m-bain/whisperX/issues/1156](https://github.com/m-bain/whisperX/issues/1156)  
4. Whisper and Pyannote: The Ultimate Solution for Speech Transcription, accessed February 7, 2026, [https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/](https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/)  
5. lablab-ai/Whisper-transcription\_and\_diarization-speaker-identification-: How to use OpenAIs Whisper to transcribe and diarize audio files \- GitHub, accessed February 7, 2026, [https://github.com/lablab-ai/Whisper-transcription\_and\_diarization-speaker-identification-](https://github.com/lablab-ai/Whisper-transcription_and_diarization-speaker-identification-)  
6. Fine-Tune Whisper For Multilingual ASR with Transformers \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/blog/fine-tune-whisper](https://huggingface.co/blog/fine-tune-whisper)  
7. How to Build a Speaker Identification System for Recorded Online Meetings \- Gladia, accessed February 7, 2026, [https://www.gladia.io/blog/build-a-speaker-identification-system-for-online-meetings](https://www.gladia.io/blog/build-a-speaker-identification-system-for-online-meetings)  
8. Speaker identification with the diarization \#1589 \- GitHub, accessed February 7, 2026, [https://github.com/pyannote/pyannote-audio/discussions/1589](https://github.com/pyannote/pyannote-audio/discussions/1589)  
9. Speaker Diarization of Known Speakers · pyannote pyannote-audio · Discussion \#1667, accessed February 7, 2026, [https://github.com/pyannote/pyannote-audio/discussions/1667](https://github.com/pyannote/pyannote-audio/discussions/1667)  
10. pyannote/embedding \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding)  
11. Speaker Verification: All Speakers Getting Perfect 1.000 Similarity Scores \#1839 \- GitHub, accessed February 7, 2026, [https://github.com/pyannote/pyannote-audio/discussions/1839](https://github.com/pyannote/pyannote-audio/discussions/1839)  
12. AI Agent Hardware Requirements: Your Complete Voice Technology Compatibility Guide, accessed February 7, 2026, [https://dialzara.com/blog/ai-voice-hardware-requirements-compatibility-guide](https://dialzara.com/blog/ai-voice-hardware-requirements-compatibility-guide)  
13. Could a Single 3090 handle transcription (whisperx), note/transcription summarization and RAG all near realtime? : r/LocalLLaMA \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1b1z33k/could\_a\_single\_3090\_handle\_transcription\_whisperx/](https://www.reddit.com/r/LocalLLaMA/comments/1b1z33k/could_a_single_3090_handle_transcription_whisperx/)  
14. Local LLM Deployment on 24GB GPUs: Models & Optimizations | IntuitionLabs, accessed February 7, 2026, [https://intuitionlabs.ai/articles/local-llm-deployment-24gb-gpu-optimization](https://intuitionlabs.ai/articles/local-llm-deployment-24gb-gpu-optimization)  
15. Whisper Speaker Diarization: Python Tutorial 2026 \- BrassTranscripts, accessed February 7, 2026, [https://brasstranscripts.com/blog/whisper-speaker-diarization-guide](https://brasstranscripts.com/blog/whisper-speaker-diarization-guide)  
16. LM Studio VRAM Requirements for Local LLMs | LocalLLM.in, accessed February 7, 2026, [https://localllm.in/blog/lm-studio-vram-requirements-for-local-llms](https://localllm.in/blog/lm-studio-vram-requirements-for-local-llms)  
17. \[Advice\] RTX 3090 \+ 64GB RAM for local LLM \+ general use : r/LocalLLaMA \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1q5y8qd/advice\_rtx\_3090\_64gb\_ram\_for\_local\_llm\_general\_use/](https://www.reddit.com/r/LocalLLaMA/comments/1q5y8qd/advice_rtx_3090_64gb_ram_for_local_llm_general_use/)  
18. ChromaDB: Semantic Search with Metadata Filters Using Python | by Sachin Sangal, accessed February 7, 2026, [https://medium.com/@sangal.sachin/chromadb-semantic-search-with-metadata-filters-using-python-456887e5e0cd](https://medium.com/@sangal.sachin/chromadb-semantic-search-with-metadata-filters-using-python-456887e5e0cd)  
19. WhisperX: Automatic Speech Recognition with Word-level Timestamps (& Diarization) \- GitHub, accessed February 7, 2026, [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)  
20. WhisperX · Models \- Dataloop, accessed February 7, 2026, [https://dataloop.ai/library/model/centralogic\_whisperx/](https://dataloop.ai/library/model/centralogic_whisperx/)  
21. \[D\] Custom vocabulary for Whisper? : r/MachineLearning \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/MachineLearning/comments/xnzjpu/d\_custom\_vocabulary\_for\_whisper/](https://www.reddit.com/r/MachineLearning/comments/xnzjpu/d_custom_vocabulary_for_whisper/)  
22. Insanely Fast Whisper | Hacker News, accessed February 7, 2026, [https://news.ycombinator.com/item?id=38266833](https://news.ycombinator.com/item?id=38266833)  
23. How can I teach whisper new words? \#963 \- GitHub, accessed February 7, 2026, [https://github.com/openai/whisper/discussions/963](https://github.com/openai/whisper/discussions/963)  
24. Interview transcription using WhisperX model, Part 1\. \- Valor Software, accessed February 7, 2026, [https://valor-software.com/articles/interview-transcription-using-whisperx-model-part-1](https://valor-software.com/articles/interview-transcription-using-whisperx-model-part-1)  
25. pyannote/speaker-diarization \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)  
26. pyannote/pyannote-audio: Neural building blocks for speaker diarization: speech activity detection, speaker change detection, overlapped speech detection, speaker embedding \- GitHub, accessed February 7, 2026, [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)  
27. Speaker identification on audio files using the pyannote/embedding model. \- GitHub, accessed February 7, 2026, [https://github.com/z3lx/speaker-identification](https://github.com/z3lx/speaker-identification)  
28. Chunk Twice, Retrieve Once: RAG Chunking Strategies Optimized for Different Content Types | Dell Technologies Info Hub, accessed February 7, 2026, [https://infohub.delltechnologies.com/en-uk/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/](https://infohub.delltechnologies.com/en-uk/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/)  
29. Chunking Strategies for LLM Applications \- Pinecone, accessed February 7, 2026, [https://www.pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/)  
30. First Steps Towards Overhearing LLM Agents: A Case Study With Dungeons & Dragons Gameplay \- arXiv, accessed February 7, 2026, [https://arxiv.org/html/2505.22809v1](https://arxiv.org/html/2505.22809v1)  
31. How to Store and Query Transcription with Speaker Labels in a Vector Database for Q\&A?, accessed February 7, 2026, [https://www.reddit.com/r/vectordatabase/comments/1h2jcla/how\_to\_store\_and\_query\_transcription\_with\_speaker/](https://www.reddit.com/r/vectordatabase/comments/1h2jcla/how_to_store_and_query_transcription_with_speaker/)  
32. Metadata-Based Filtering in RAG Systems | CodeSignal Learn, accessed February 7, 2026, [https://codesignal.com/learn/courses/scaling-up-rag-with-vector-databases/lessons/metadata-based-filtering-in-rag-systems](https://codesignal.com/learn/courses/scaling-up-rag-with-vector-databases/lessons/metadata-based-filtering-in-rag-systems)  
33. Diarization too slow · Issue \#274 · m-bain/whisperX \- GitHub, accessed February 7, 2026, [https://github.com/m-bain/whisperX/issues/274](https://github.com/m-bain/whisperX/issues/274)  
34. Speaker Diarization with Pyannote on VAST, accessed February 7, 2026, [https://vast.ai/article/speaker-diarization-with-pyannote-on-vast](https://vast.ai/article/speaker-diarization-with-pyannote-on-vast)  
35. Level up Your RAG Application with Speaker Diarization | Haystack, accessed February 7, 2026, [https://haystack.deepset.ai/blog/level-up-rag-with-speaker-diarization](https://haystack.deepset.ai/blog/level-up-rag-with-speaker-diarization)  
36. Level Up Your D\&D Session Notes: Useful AI Prompts & Cleanup Scripts \- Medium, accessed February 7, 2026, [https://medium.com/@brandonharris\_12357/level-up-your-d-d-session-notes-useful-ai-prompts-cleanup-scripts-ca959de9a541](https://medium.com/@brandonharris_12357/level-up-your-d-d-session-notes-useful-ai-prompts-cleanup-scripts-ca959de9a541)  
37. D\&D Game Master Agent with RAG \- Tasking AI, accessed February 7, 2026, [https://www.tasking.ai/examples/dnd-game-master-agent-with-rag](https://www.tasking.ai/examples/dnd-game-master-agent-with-rag)  
38. D\&D General \- A.I. D\&D Session Transcripts and Summaries Example \- EN World, accessed February 7, 2026, [https://www.enworld.org/threads/a-i-d-d-session-transcripts-and-summaries-example.707199/](https://www.enworld.org/threads/a-i-d-d-session-transcripts-and-summaries-example.707199/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAACdElEQVR4Xu2WW4hNURjH/5IQISWlZEi8uKV4QJlccx08iSa3vAhPcisPitweJJcHb5LCi5RLkuZIkpBLyAPlDSmvkjT+//m+Za/Zp7Nnnz0m47R/9Wuv/a010/7W5VsHKPmvGURf0PZ0RyOygH5PBxuRw7SSDvYWptEH9Ad9Se/Btt9r+tnb1+ia8AcpJtMz9Cxs6x7r3N37aKX76Cx6xWN9YBOxirbRW3Sg94nxsMkQ82CT0pJ09xwj6FV6N/J8pxG1WQnbevPpkVTfJX+OpB+9reKjxLb5+wn62Ns9Rl96kH6hu+gmuoI206nJsEyWIUlUzxhNWOChP6fAEp3t749g23asv3cwne6ALfdaup0u9L71dKO/L/FYFgNg20pnRO2iLEbtRC/7cxS94e3hsHM9ic6kP2Fb/ID3d7AZNhuxx70vFAN5zmNZnET1VitCSFRXRDrRPbSJ3qFzo7h2z3VYchV6gR6K+v/wC5aQZma0x5Z7bGkY1AVb04GCKFFtf63oe1dbVknom4YmQ+vnIpLVO+oxregbWMXLg3ZHc4Yq/3kIiYYVnUg/0GF0dzSuECrlIdFvSA64zm9eTsFmvZY7k6GZqB7EiYq9sDvyNmzSukUFSbJa4U90cDygC1anAwVR1VWiKoIhUVXzJ3QMfYX6vquKcNGGs6pZrAcVg7+B7tFwRuNipOupH12H/HdyTbbAEj2d7sjBONhZ0rXUHVph1XUOqqv9c9j9qOsr3KOF6E/f0Qnpjpw00fv0Jp0BS35IPCAD3YHxb139H036M7rBxyyCJasq/JTu9/g/QedJ57WNvqVfYR+sXywlJSUlJSWNwm+KoYldK8UsdQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAA20lEQVR4XmNgGAWjYAiCdCD+j4YToXJ/0cSdoeI4gRQQOzAgNEwAYgmonAcQ/wLieVA1glBxggBm2BkksQwGiOtUkMRggA2IvdAFYWAfA8JAJ6jYVSCeCVcBAVFAXA/E14B4M5ocHAQwIAzbBBUDsbXgKiBAF4hNgHgFAx7DmIH4HgPCQA0g3o2iAhWADNuCLogMhID4KwPCQBZUaRSwnAGPy2AAFEYww/ABkMu2oguiA0Ugfg/Et9El0ABRhhELQIZtQxckF4AM244uSCooZ4DE8jsg/glljyQAAAw3Ntf6XOuOAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAA7klEQVR4XmNgGAWjYAiCdCD+j4YToXJ/0cSdoeI4gRQQOzAgNEwAYgmonAcQ/wLieVA1glBxggBm2BkksQwGiOtUkMRAwACIC4A4EIg50eTAYB8DwkAnqNhVIJ4JVwEB1UDcCcRBQLwHiB+hSkNAAAPCsE1QMRBbC64CAm4CsTqUzQ7EDxBSCMAMxPcYEAZqAPFuFBUQAJLrQuKvRGKjACEg/sqAMJAFVRoDWDFA1OEEoDCCGYYP8ADxZSCuQ5dABopA/B6Ib6NLIAGQi9cCcRK6BKmAkQGS7iLQJcgB0xkgSaIWCZMNYOGJjEcSAACzrDrUFr2omgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAA30lEQVR4XmNgoBdgBGJFIJZGl0AHWkD8HYj/A3ERmhwG4APifQwQxaZocljBBSD+DMQs6BLYAMhUkOlEAZDiADQxnLaAFAtB2SFAfBSITwMxJ1wFFIgC8XkoOwyITwGxLAPEAH2YIhgAWT8NiF2A+CAQ8wOxNhA/BmIeJHVgAFIAMuUOENcxYLEaBjiA+CcQSwBxFwNE024UFUjAmgGiAAY2Q/mSQBwMxFxIcgwlUEkYADnpNxCzA/E5BkiagQOQO0GCMAAKDZDmBUDMjCQOBspALIImBgoJHTSxUTDYAADoXSUfOaNZmAAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAaCAYAAADIUm6MAAABsklEQVR4Xu2WSygFYRTH/x555FHKIynvJGVha0NKSimlbCwUG2UhK3mFbFixs1AoS5TCwoqSokRWkoWNUpRHsiHx/+bMuN98c2/du5m5i/nVr9s55/vqzMw3Zy4QEjwltMFMJjMn9I3+0iOjFhSZdIL20Uaa4i4LrXQF0vicuxQINfSGLtFN+k3PXSs01iCNt5uFADile1o8BektVcv98wC5sjyz4DMFkCaVDm12rH5dDNuFZS2XSyu12E8yaIUWzyLGHd+yCz12vEt3II9rxFkUEGrS3dNPs6B4hjReCLnLvbSLPtL9yDIPOXSSTidglbUzPuYhR/iANhk1C9X0JS2jZ3bu0M6PO4uikI3EG6+1dsZHJ52hL5Dh4cF5GV7pulFLBrIgg0M9AReq6Wo6QJ9ovbvsK+rD0w/vkVJz/VpP1EHmpsMQZG4qiiFnPRZF9AuRJxaPHdbO2KxC1n0Y+Vt6pyfUXV7Q4jFI8wp1AYNazQ+2IY3/aLl0+k43tBwuaL4Wl9JjyH+WZi3vJ92QEbgI+eSri7hyrSDlZoK0QF6IIEmjo5ATEXUUhoSEhISE+Mof+WhlMFrMRAIAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAr0lEQVR4XmNgoCUwQRcAAmV0ARC4B8QbgLgDiOcA8Wkg/oqiAgoeA/F/JPwUiI1RVEABSCHIxH4gTgJiMVRpBHiALoALgNxYCMTrgLgTiKVQpRHgLhAvBuIwIG4D4o9AnICsAAZOofFnMEA8RRAEMRCp0J4BopAdWfAMVBAZOEHFmJAFlwPxYWQBIChgwNQMtgZkKgxoAPELBojPMUATEC9jgAQRKI7vMOCIwlGAAQCXsiVC1E3t+QAAAABJRU5ErkJggg==>