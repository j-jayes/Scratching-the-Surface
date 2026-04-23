# Skill: Mermaid.js Diagrams in Quarto

## Core Rule
When generating architecture or flow diagrams for this project, always use **Mermaid.js** syntax
inside Quarto `.qmd` files. Never use static images for diagrams that can be expressed in Mermaid.

---

## Quarto Mermaid Block Syntax

````markdown
```{mermaid}
%%| label: fig-cascade-architecture
%%| fig-cap: "Cascade Defect Detection Pipeline"
flowchart LR
    ...
```
````

---

## Preferred Diagram Styles

### 1. System Architecture — Left-to-Right Flowchart

```mermaid
flowchart LR
    INPUT([🖼️ Raw Frame])

    subgraph L1["Layer 1 — Gatekeeper (Autoencoder)"]
        AE[Conv Autoencoder\nMSE Reconstruction]
        T1{MSE >\nThreshold?}
    end

    subgraph L2["Layer 2 — Specialist (YOLOv8)"]
        YOLO[YOLOv8n\nInference]
        T2{Confidence\n> 85%?}
    end

    subgraph L3["Layer 3 — Oracle (GPT-4o)"]
        GPT[Azure OpenAI\ngpt-4o]
        JSON[Pydantic\nJSON Parse]
    end

    DISCARD([✅ No Defect\nDiscard])
    LOG([📋 Log Defect\n+ Retrain Queue])

    INPUT --> AE
    AE --> T1
    T1 -- No --> DISCARD
    T1 -- Yes --> YOLO
    YOLO --> T2
    T2 -- Yes --> LOG
    T2 -- No --> GPT
    GPT --> JSON --> LOG

    style L1 fill:#dbeafe,stroke:#3b82f6
    style L2 fill:#dcfce7,stroke:#22c55e
    style L3 fill:#fef9c3,stroke:#eab308
```

### 2. Data Split Strategy — Pie or Flowchart

```mermaid
flowchart TD
    RAW["NEU Dataset\n1,800 images\n6 classes"]
    SEED["🌱 Few-Shot Seed\n18 images (1%)\nLabels kept"]
    UNLABELLED["🔓 Unlabelled Pool\n~1,420 images (79%)\nLabels stripped"]
    TEST["🧪 Golden Test Set\n360 images (20%)\nGround-truth labels"]

    RAW --> SEED
    RAW --> UNLABELLED
    RAW --> TEST

    SEED --> GPT4O["GPT-4o\nFew-Shot Prompt"]
    UNLABELLED --> GPT4O
    GPT4O --> PSEUDO["Pseudo-Labels\nfor YOLOv8 Training"]

    style SEED fill:#dcfce7
    style TEST fill:#fee2e2
    style UNLABELLED fill:#fef9c3
```

### 3. Azure Infrastructure — Left-to-Right

```mermaid
flowchart LR
    subgraph INGEST["Ingestion"]
        CAM[Factory Camera\nRTSP Stream]
    end

    subgraph STORAGE["Azure Data Lake Gen2"]
        RAW_CONT[raw/]
        PROCESSED_CONT[processed/]
        LOGS_CONT[logs/anomalies/]
    end

    subgraph QUEUE["Azure Service Bus"]
        SBQ[defect-queue]
    end

    subgraph ACA["Azure Container Apps — West Europe (T4 GPU)"]
        L1[layer1-autoencoder\nscale-to-zero]
        L2[layer2-yolo\nscale-to-zero]
    end

    subgraph ORACLE["Azure OpenAI"]
        GPT[gpt-4o\nSwedenCentral]
    end

    CAM --> RAW_CONT
    RAW_CONT --> L1
    L1 -- "MSE > τ" --> SBQ
    SBQ -- KEDA trigger --> L2
    L2 -- "conf < 0.85" --> GPT
    GPT --> LOGS_CONT
    L2 -- "conf ≥ 0.85" --> LOGS_CONT

    style ACA fill:#dbeafe,stroke:#3b82f6
    style QUEUE fill:#fef9c3,stroke:#eab308
    style ORACLE fill:#f3e8ff,stroke:#a855f7
```

### 4. Sequence Diagram — Request Lifecycle

```mermaid
sequenceDiagram
    participant CAM as 🎥 Camera
    participant L1 as Layer 1<br/>(Autoencoder)
    participant SB as Service Bus
    participant L2 as Layer 2<br/>(YOLOv8)
    participant GPT as Layer 3<br/>(GPT-4o)
    participant DB as ADLS Logs

    CAM->>L1: Send frame
    L1->>L1: Compute MSE
    alt MSE < threshold
        L1-->>CAM: ✅ No defect (discard)
    else MSE ≥ threshold
        L1->>SB: Enqueue(image_uri)
        SB->>L2: KEDA trigger (scale up)
        L2->>L2: YOLO inference
        alt confidence ≥ 85%
            L2->>DB: Log defect + bbox
        else confidence < 85%
            L2->>GPT: Few-shot classification request
            GPT-->>L2: JSON {class, confidence, reasoning}
            L2->>DB: Log defect + GPT annotation
        end
    end
```

---

## Styling Guidelines

- Use `fill` and `stroke` to colour subgraph backgrounds consistently:
  - Layer 1 (Autoencoder): `fill:#dbeafe,stroke:#3b82f6` (blue)
  - Layer 2 (YOLOv8):      `fill:#dcfce7,stroke:#22c55e` (green)
  - Layer 3 (GPT-4o):      `fill:#fef9c3,stroke:#eab308` (yellow)
  - Azure infra:            `fill:#f1f5f9,stroke:#64748b` (slate)
- Always add `%%| fig-cap:` labels for Quarto cross-referencing
- Keep node labels short; use `\n` for line breaks inside nodes
- Prefer `flowchart LR` for pipeline diagrams, `flowchart TD` for hierarchical data splits
