# Agentic Network Slice Negotiation Framework with Anchoring Bias Mitigation

## Overview

This project implements an autonomous resource negotiation system focused on mitigating **Anchoring Bias** in network resource allocation. Two AI agents representing **eMBB (Enhanced Mobile Broadband)** and **URLLC (Ultra-Reliable Low-Latency Communication)** network slices negotiate for shared **Radio Access Network (RAN)** bandwidth using specialized randomized protocols.

The agents use **Digital Twins** to predict the impact of proposals on latency and energy consumption, optimizing a custom **Utility Function** that prioritizes SLA compliance while avoiding biased decision-making.

---

## 🚀 Key Features

- **LLM-Powered Reasoning**  
  Uses Google Gemini 2.5 Flash for iterative, turn-based negotiation.

- **High-Fidelity Simulation**  
  Fluid-flow queuing models simulate real-time network behavior.

- **Predictive Digital Twins**  
  Agents evaluate “what-if” scenarios before making decisions.

- **Anchoring Bias Mitigation**  
  Randomized anchoring protocols reduce unfair influence of initial offers.

- **Robust API Integration**  
  Exponential backoff ensures stable LLM API usage.

---

## 🛠️ System Architecture

### Components

- **Network Simulator**  
  Maintains ground-truth network state (queues, traffic, energy).

- **Slice Digital Twin**  
  Mirrored simulator using M/M/1 steady-state approximations.

- **Negotiation Agent**  
  Evaluates proposals using a utility function:
  - SLA Penalty: High cost for exceeding latency targets  
    (50ms for eMBB, 10ms for URLLC)
  - Energy Cost: Secondary objective

- **Agentic Negotiator**  
  Manages trials, warm-up phases, and the propose-evaluate-counter protocol.

---

## ⚓ Anchoring Bias & Mitigation Strategy

Anchoring bias causes agents to rely too heavily on initial offers, often skewing outcomes in favor of the first mover.

### Mitigation Approaches

- **Fixed Anchoring (Baseline)**  
  Deterministic proposals based on Digital Twin requirements.

- **Randomized Anchoring (Mitigation)**  
  Initial anchors are decoupled from immediate needs, encouraging flexible and objective negotiation.

- **Quantification**  
  Anchor Distance CDFs are compared to measure bias “stickiness” and mitigation effectiveness.

---

## 📋 Prerequisites

- Python 3.8+
- Google AI API Key (for Gemini LLM)
- Dependencies:
  - numpy
  - matplotlib
  - httpx

---

## ⚙️ Setup & Installation

### Clone the Repository

```bash
git clone <repository-url>
cd agentic-negotiation
