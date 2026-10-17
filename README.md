SDR IQ capture → feature extraction → LangChain → LangGraph → telecom-radio questions

AI LLM interpret raw IQ samples and useful measurements such as frequency, bandwidth, power, SNR, modulation estimate, etc., and LangGraph orchestrates analysis and querying.

             SDR
              │
              ▼
       IQ / RF Capture
              │
              ▼
      DSP / Feature Extraction
              │
              ▼
      SDR Feature Dataset
              │
              ▼
       LangChain Tool
              │
              ▼
        ┌─────────────┐
        │  LangGraph  │
        │             │
        │ Analyze     │
        │     ↓       │
        │ Generate    │
        └──────┬──────┘
               │
               ▼
          Groq LLM
               │
               ▼
       Natural-language
        SDR analysis
		
- SDR/DSP → measures the signal.
- LangChain → connects the LLM to the SDR-analysis function/tool.
- LangGraph → controls the workflow/state.
- Groq LLM → explains the measurements and answers questions.

CSV can be replaced with a real RTL-SDR .iq/.bin capture, where Python calculates FFT, occupied bandwidth, peak frequency, RSSI/power and SNR, and then sends those extracted features into the LangGraph pipeline. That would connect your earlier RF/SDR expertise directly with your current LangChain/LangGraph work.


Sample Prompts:

Which frequencies were observed?

What modulation types were detected?

What is the average SNR?

Which technology has the strongest signal?

Which observations have poor SNR?

What can you infer about the radio environment?

OUTPUT

>python AI_Agent_SDR.py

Enter Groq API key:
  timestamp  frequency_mhz  bandwidth_khz  power_dbm  snr_db modulation technology
0  10:01:01          915.2            125        -48      22        FSK       LoRa
1  10:01:05          915.2            125        -51      19        FSK       LoRa
2  10:02:10          868.3            250        -62      12       GFSK        ISM
3  10:03:20         2402.0           1000        -45      28       QPSK       WiFi
4  10:04:15         2402.0           1000        -49      25       QPSK       WiFi

Ask about SDR data (type exit to stop): Which frequencies were observed?

AI: Observed frequencies:

- 915.2 MHz
- 2402.0 MHz
- 868.3 MHz

Ask about SDR data (type exit to stop): What modulation types were detected?

AI: The observation identified three modulation families:

- **FSK** (Frequency‑Shift Keying) – detected in 2 instances
- **QPSK** (Quadrature Phase‑Shift Keying) – detected in 2 instances
- **GFSK** (Gaussian‑shaped FSK) – detected in 1 instance

These are the only modulation types present in the provided SDR data.
