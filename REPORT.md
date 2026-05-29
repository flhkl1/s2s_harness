# S2S Comparison Harness — Design Report

## 1. Purpose & Hypothesis

This harness enables a controlled A/B comparison of two real-time speech-to-speech (S2S) architectures as applied to a structured clinical interview task (cognitive assessment across five domains: memory, orientation, judgment, home/hobbies, language).

**Hypothesis**: MoshiRAG and KAME achieve meaningfully different latency and knowledge-fidelity profiles due to their distinct architectural mechanisms for incorporating LLM knowledge into the speech stream, even when all exogenous variables are held constant.

---

## 2. Models Under Comparison

| | **KAME** | **MoshiRAG** |
|---|---|---|
| **Paper** | arXiv:2510.02327 (SakanaAI) | arXiv:2604.12928 (Kyutai Labs) |
| **Base model** | Moshi 8B (Kyutai) | Moshi 8B / Moshika variant |
| **HuggingFace** | `SakanaAI/kame` | `kyutai/moshika-rag-pytorch-bf16` |
| **LLM coupling** | Parallel async oracle injection | Trigger-token retrieval + reference encoder |
| **LLM position** | Concurrent with speech, token-by-token | Asynchronous, fires on retrieval trigger |
| **ASR** | Google Cloud Speech-to-Text | Gradium ASR |
| **Services needed** | 1 | 3 |

---

## 3. Controlled Variables

These are held **identical** across both arms to isolate architecture as the sole independent variable.

| Variable | Value | Rationale |
|---|---|---|
| **Hardware** | Railway A100 40 GB for all GPU services | Eliminates compute-speed confound |
| **Model precision** | BF16 for both | KAME ships F32 (~32 GB); cast to BF16 at load to match MoshiRAG's native BF16 (~16 GB). 4-bit quantization was rejected because it degrades audio quality and would bias results against KAME |
| **External LLM** | OpenAI GPT-4o-mini (same API key, same model name) | KAME uses it as the oracle stream; MoshiRAG uses it as the retrieval backend. Using the same model isolates the architectural difference in *how* each system queries and uses the LLM, not *which* LLM |
| **Session structure** | `clinical_interview.yaml` | Same persona, domains, probe depth, opening line |
| **VAD** | webrtcvad, aggressiveness=2, silence_ms=500 | Same utterance segmentation guarantees both models receive identical audio boundaries |
| **Audio pipeline** | 16 kHz PCM on harness boundary; Opus at 24 kHz to model servers | Both models use the same Moshi WebSocket protocol (Opus frames, 1920-sample frame size, 80 ms) |
| **Frame size** | 1920 samples @ 24 kHz (80 ms) | Moshi's native frame size; deviating would corrupt both models equally |

---

## 4. Independent Variable

**Model architecture** — specifically, the mechanism by which each system integrates external LLM knowledge into the real-time speech generation loop:

- **KAME**: The oracle LLM (GPT-4.1 / here GPT-4o-mini) runs in a parallel async stream alongside speech generation. As the user speaks, the ASR transcription is routed to the LLM; the LLM's token stream is injected token-by-token into Moshi's inner monologue channel, guiding generation continuously.

- **MoshiRAG**: The model predicts a special retrieval trigger token when it determines external information is needed. On trigger, the conversation context is sent to the retrieval backend (here GPT-4o-mini via OpenAI API); the retrieved text is encoded by a separate reference text encoder and injected into the model's generation stream. Retrieval is event-driven, not continuous.

---

## 5. Dependent Variables / Metrics

Logged in `logs/{run_id}.jsonl` (one JSONL event per line):

| Metric | Event | Description |
|---|---|---|
| **TTFA** | `first_audio.ttfa_ms` | Milliseconds from `utterance_sent` to first audio byte received from model |
| **Total response time** | `response_done.total_ms` | Milliseconds from `utterance_sent` to last audio byte (response complete) |
| **Utterance size** | `utterance_sent.bytes` | PCM byte count of each user turn |
| **Run metadata** | `run_start` | adapter, run_id, vad_aggressiveness, vad_silence_ms |

TTFA is the primary latency metric because it determines perceived responsiveness — users notice the gap between speaking and hearing the first word of a reply more than total response duration.

---

## 6. Audio Transport Architecture

```
Client (browser / test script)
  │  16 kHz PCM int16 — raw binary WebSocket frames
  ▼
harness-api (server.py)
  │  VAD segments frames into utterances
  │  Resample 16 kHz → 24 kHz
  │  Opus-encode (1920-sample frames, APPLICATION_VOIP)
  │  Send raw Opus binary frames over WebSocket
  ▼
kame-server / moshirag-main
  │  (internal: Opus decode → Mimi encode → model input)
  │  
  │  Response: binary WebSocket messages with 1-byte prefix
  │    0x00 = handshake
  │    0x01 = Opus audio output
  │    0x02 = text token (LLM oracle prediction)
  ▼
harness-api (adapter._receive_loop)
  │  Parse prefix, decode 0x01 Opus → PCM
  │  Resample 24 kHz → 16 kHz
  │  Send raw 16 kHz PCM to client
```

**Why Opus?** Moshi's WebSocket protocol uses Opus for network transport (compressed), then decodes to PCM internally before the Mimi neural codec stage. This is the native protocol for both KAME and MoshiRAG servers. Using raw PCM over WebSocket (as the original adapter stubs did) would not be understood by either server.

**Response-end detection**: Neither KAME nor MoshiRAG sends an explicit "done" signal over WebSocket. We declare the response complete after **2.0 seconds of output audio silence** — i.e., no new 0x01 frames received for 2 seconds. This threshold exceeds typical inter-sentence pauses in clinical interview speech (~0.5–1.0 s) while remaining sensitive enough to not miss turn-taking cues.

---

## 7. What Is NOT Compared

| System | Reason excluded |
|---|---|
| **Qwen3-Omni** | API-routed (DashScope); network latency is a confound — not isolatable from model quality |
| **OpenAI Realtime API** | Same issue — cloud API latency not under our control |
| **TML Interaction Model** | Closed source, research preview not yet available |

---

## 8. Known Limitations

1. **Single concurrent session per model server**: Both KAME and MoshiRAG reject concurrent WebSocket connections with HTTP 503. The harness is designed for single-session use; parallel A/B runs require two separate deployments.

2. **Infrastructure asymmetry**: MoshiRAG requires 3 Railway services (main + conditioner + retrieval API); KAME requires 1. This means MoshiRAG has more inter-service network hops. This is an architectural property of MoshiRAG, not a harness artifact, and is faithfully reproduced here — but it is a potential confound for tail-latency comparisons.

3. **ASR asymmetry**: KAME uses Google Cloud STT; MoshiRAG uses Gradium ASR. Both are external streaming ASR services but different implementations. This affects the quality of text routed to the LLM backend, not audio generation latency directly.

4. **English only**: Both models are English-only (`en-US`). The clinical interview YAML is also English.

5. **BF16 cast for KAME**: SakanaAI has not published BF16 benchmark numbers for KAME. The BF16 cast (`--dtype bfloat16` flag) is our addition for hardware parity. Any quality delta introduced by this cast is a confound in favour of MoshiRAG (which natively ships as BF16).

6. **Retrieval latency budget**: MoshiRAG's retrieval pipeline degrades if the retrieval LLM takes > 3 seconds. GPT-4o-mini p50 latency is typically 0.8–1.5 seconds for short prompts, which is within budget. Tail latencies (p95 > 3 s) may cause degraded responses in MoshiRAG but not KAME.

---

## 9. Future Work

- Add Whisper-based transcript logging to enable qualitative scoring of clinical interview domain coverage
- Add domain advancement signal from a background scoring LLM (described in NORTH_STAR.md)
- Extend to parallel simultaneous comparison (requires multi-session support patches upstream)
- Add MetaCog-Bench evaluation (streaming metacognition, described in NORTH_STAR.md)
