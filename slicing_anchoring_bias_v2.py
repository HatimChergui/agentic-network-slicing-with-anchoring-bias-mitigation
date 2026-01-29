import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import asyncio
import time
import os # Import os for environment variable access
import httpx # Import httpx for local asynchronous HTTP requests
from typing import Dict, Any, Tuple, List, Optional
import math
import json
import logging

# Set up logging for API calls and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. CONFIGURATION AND CONSTANTS ---
# Negotiation Setup
NUM_TRIALS = 2  # Reduced for faster testing with LLM calls
MAX_ROUNDS = 5    # Max rounds per trial
TIMESTEP_S = 0.01  # Simulation time step (tau)

# LLM API Configuration
# Reads the key from the environment variable named GOOGLE_API_KEY (User specified)
API_KEY = os.environ.get("GOOGLE_API_KEY", "")  
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
# FIX: Use query parameter for API key, required for standard local API key usage
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"
MAX_RETRIES = 5

if not API_KEY:
    logging.error("GOOGLE_API_KEY environment variable not found. LLM calls will fail.")

# Network Slicing Constraints (Ground Truth)
TOTAL_RAN_BANDWIDTH_MHZ = 50.0
eMBB_MAX_CPU_GHZ = 20.0 # Max CPU this slice can use (fmax_i)
URLLC_MAX_CPU_GHZ = 20.0
# Simplification: Assume constant spectral efficiency and CPU utilization factor
SPECTRAL_EFFICIENCY_BITS_PER_HZ = 5.0 # R/B
CPU_UTILIZATION_FACTOR_BITS_PER_HZ_PER_GHZ = 1e9 * 0.5 # U, scaled to 0.5 bits/s per Hz per GHz

# Slice SLAs
eMBB_SLA_LATENCY_MS = 50.0  # Relaxed
URLLC_SLA_LATENCY_MS = 10.0 # Strict

# Energy Model Parameters (watts)
P_STATIC = 5.0 # Static base power
C_BW = 0.5     # Cost per MHz
C_CPU = 0.001  # Cost per GHz^3 (cubic power model)

# Traffic Model (Simplified time-varying traffic)
def get_traffic_arrival_rate_bps(t: int, slice_id: str) -> float:
    """Provides a time-varying traffic arrival rate."""
    # UPDATED: Reduced eMBB base rate, increased URLLC base rate
    base_rate = 100e6 if slice_id == 'eMBB' else 60e6
    fluctuation = (math.sin(t * 0.1) * 0.3 + 1.0) # Oscillation between 0.7 and 1.3
    if slice_id == 'eMBB':
        return base_rate * fluctuation
    else: # URLLC, often more bursty/lower volume but strict
        return base_rate * (1 + 0.5 * (t % 10 == 0)) * fluctuation # occasional bursts
    
# --- LLM API CALLER WITH BACKOFF ---

async def call_gemini_api(system_prompt: str, user_query: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API using httpx for local asynchronous execution, 
    handling the API Key via a query parameter.
    """
    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }
    
    headers = { 'Content-Type': 'application/json' }
    
    if not API_KEY:
        # Check happens at the top level, but ensures this function bails out safely.
        logging.error("Cannot call API: GOOGLE_API_KEY is missing.")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            # Use httpx.AsyncClient for reliable asynchronous requests
            async with httpx.AsyncClient(timeout=60.0) as client:
                # API key is included in GEMINI_API_URL via the query string.
                response = await client.post(
                    GEMINI_API_URL, 
                    headers=headers, 
                    data=json.dumps(payload)
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract JSON string from the response structure
                json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                
                # Parse the JSON string
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    logging.error(f"LLM returned invalid JSON: {json_text}")
                    return None
            
            elif response.status_code in [429, 500, 503]:
                # Handle rate limiting or temporary server errors
                wait_time = 2 ** attempt + random.random()
                logging.warning(f"API Rate limit/server error ({response.status_code}). Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/{MAX_RETRIES}).")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"API failed with status {response.status_code}: {response.text}")
                return None

        except httpx.HTTPError as e:
            # Catch network/connection errors specific to httpx
            wait_time = 2 ** attempt + random.random()
            logging.error(f"Network error during API call: {e}. Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/{MAX_RETRIES}).")
            await asyncio.sleep(wait_time)
        except Exception as e:
            # General error catch
            wait_time = 2 ** attempt + random.random()
            logging.error(f"Unexpected error: {e}. Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/{MAX_RETRIES}).")
            await asyncio.sleep(wait_time)

    return None

# --- 2. DIGITAL TWIN AND NETWORK SIMULATOR ---


class NetworkSimulator:
    """Manages the ground truth state of the network slices."""
    def __init__(self, slice_id: str, max_cpu_ghz: float, sla_ms: float):
        self.slice_id = slice_id
        self.fmax = max_cpu_ghz
        self.sla_ms = sla_ms
        self.tau = TIMESTEP_S
        self.cpu_factor = CPU_UTILIZATION_FACTOR_BITS_PER_HZ_PER_GHZ
        
        # Current State
        self.cqueue_bits = 0.0 # Edge queue (compute)
        self.rqueue_bits = 0.0 # RAN queue (radio)
        self.t = 0
        self.current_traffic_arrival_rate_bps = get_traffic_arrival_rate_bps(0, self.slice_id)

        # Metrics for ground truth tracking
        self.latency_ms_history = []
        self.energy_watts_history = []
        self.sla_violations = 0
        
    def _calculate_slice_capacity(self, ran_bandwidth_mhz: float) -> float:
        """Calculates the overall bottleneck capacity for the slice (in bps)."""
        edge_capacity_bps = self.fmax * self.cpu_factor
        ran_capacity_bps = ran_bandwidth_mhz * 1e6 * SPECTRAL_EFFICIENCY_BITS_PER_HZ
        return min(edge_capacity_bps, ran_capacity_bps)

    def _calculate_energy_consumption(self, ran_bandwidth_mhz: float) -> float:
        """Calculates energy consumption (simplified linear model + cubic CPU)."""
        e_bw = ran_bandwidth_mhz * C_BW
        e_cpu = (self.fmax / 100.0)**3 * C_CPU * 100000.0 
        return P_STATIC + e_bw + e_cpu

    def _calculate_max_possible_energy(self) -> float:
        """
        Calculates the maximum energy this slice would consume when sharing the
        full system load (assuming 20MHz allocation for max consumption check).
        Used for setting the system's BASE_ENERGY_WATTS reference.
        """
        # Calculate maximum energy assuming full BW utilization (e.g., 20MHz for a 40MHz system)
        ran_bandwidth_mhz = TOTAL_RAN_BANDWIDTH_MHZ / 2.0 
        return self._calculate_energy_consumption(ran_bandwidth_mhz)
    
    def step(self, ran_bandwidth_mhz: float) -> Dict[str, float]:
        """Advance the ground truth simulation by one time step."""
        self.t += 1
        current_traffic = get_traffic_arrival_rate_bps(self.t, self.slice_id)
        
        arriving_data_in_tau = self.tau * current_traffic
        slice_capacity_bps = self._calculate_slice_capacity(ran_bandwidth_mhz)
        
        Q_before_tx = self.cqueue_bits + self.rqueue_bits + arriving_data_in_tau
        
        # --- FIXED LATENCY CALCULATION ---
        # The latency is calculated as the time it would take to clear the entire queue
        # (backlog + new arrivals) at the newly allocated service rate (slice capacity).
        # This is a standard fluid-flow model approximation of queuing delay.
        if slice_capacity_bps > 0:
            latency_sec = Q_before_tx / slice_capacity_bps
        else:
            # If capacity is zero but there's data in the queue, latency is
            # effectively infinite. We return a large, fixed penalty value (1s).
            # If both queue and capacity are zero, latency is zero.
            latency_sec = 1.0 if Q_before_tx > 0 else 0.0
            
        latency_ms = latency_sec * 1000.0
        
        # Update queue state based on what could be transmitted in this timestep
        radio_capacity_in_tau = self.tau * slice_capacity_bps
        transmitted_data_in_tau = min(Q_before_tx, radio_capacity_in_tau)
        Q_after_tx = max(0, Q_before_tx - transmitted_data_in_tau)
        self.cqueue_bits = 0 
        self.rqueue_bits = Q_after_tx 
        
        energy_watts = self._calculate_energy_consumption(ran_bandwidth_mhz)
        
        # NOTE: We keep history tracking here, but the negotiator will only use the last value.
        self.latency_ms_history.append(latency_ms)
        self.energy_watts_history.append(energy_watts)
        if latency_ms > self.sla_ms:
            self.sla_violations += 1

        return {
            "latency_ms": latency_ms,
            "energy_watts": energy_watts
        }

class SliceDigitalTwin(NetworkSimulator):
    """Used by agents to predict the outcome of a proposed bandwidth allocation."""

    def __init__(self, slice_id: str, max_cpu_ghz: float, sla_ms: float):
        super().__init__(slice_id, max_cpu_ghz, sla_ms)
        self.average_arrival_rate_bps = 0.0
        
    def reset_to_current_state(self, current_state: Dict[str, Any]):
        """Resets DT state using the ground truth state."""
        self.cqueue_bits = current_state["cqueue_bits"]
        self.rqueue_bits = current_state["rqueue_bits"]
        self.t = current_state["t"]
        self.average_arrival_rate_bps = current_state["average_arrival_rate_bps"]
        
    def get_current_context(self) -> Dict[str, float]:
        """Exposes key metrics for LLM reasoning."""
        return {
            "current_queue_bits": self.cqueue_bits + self.rqueue_bits,
            "average_arrival_rate_mbps": self.average_arrival_rate_bps / 1e6
        }
        
    def simulate_step_for_prediction(self, proposed_ran_bandwidth_mhz: float) -> Dict[str, float]:
        """Predicts one step with the proposed bandwidth, using the steady-state M/M/1 approximation."""
        
        arrival_rate = self.average_arrival_rate_bps
        slice_capacity_bps = self._calculate_slice_capacity(proposed_ran_bandwidth_mhz)
        
        rho = arrival_rate / slice_capacity_bps
        
        predicted_latency_ms = 1000.0 # Default high latency
        if rho < 1.0:
            packet_size_bits = 8000
            lambda_p = arrival_rate / packet_size_bits
            mu_p = slice_capacity_bps / packet_size_bits
            
            if mu_p > lambda_p:
                wait_time_sec = (1.0 / (mu_p - lambda_p))
                predicted_latency_ms = wait_time_sec * 1000.0
            
        predicted_energy_watts = self._calculate_energy_consumption(proposed_ran_bandwidth_mhz)

        return {
            "predicted_latency_ms": predicted_latency_ms,
            "predicted_energy_watts": predicted_energy_watts
        }


# --- 3. MEMORY CLASSES (Simplified to ONLY Vanilla Memory) ---

class Memory:
    """Base class (Vanilla) for storing and retrieving joint strategies AND tracking anchor distance."""
    def __init__(self):
        self.log = [] 
        self.distance_from_anchor = [] # ADDED to base class for metric consistency

    # FIX: Updated signature to accept all arguments used in the negotiation loop
    def update(self, state_hash: int, initial_anchor: Tuple[float, float], final_agreement: Optional[Tuple[float, float]], outcome: bool):
        """Stores the result of a negotiation round and calculates anchor distance."""
        if final_agreement and outcome:
            # Anchor distance calculation (final BW - initial BW)
            self.distance_from_anchor.append(
                (initial_anchor[0], final_agreement[0], 'eMBB')
            )
            self.distance_from_anchor.append(
                (initial_anchor[1], final_agreement[1], 'URLLC')
            )
        
        # For base memory, we log the result (final agreement if successful, or the anchor/last state if failed)
        proposal_to_log = final_agreement if outcome and final_agreement else initial_anchor
        self.log.append((state_hash, proposal_to_log, outcome))

    def distill_strategy(self, state_hash: int, current_bw_needed: float) -> Optional[Tuple[float, float]]:
        successful_attempts = [
            p for h, p, success in self.log 
            if success
        ]
        
        if successful_attempts:
            # Return the average of successful past joint strategies
            avg_p = np.mean(successful_attempts, axis=0)
            return tuple(avg_p)
        return None

# The DebiasedMemory class has been removed to enforce vanilla memory use.


# --- 4. LLM-POWERED AGENT CLASS (Utility Updated) ---

class NegotiationAgent:
    """Represents a slice manager agent (eMBB or URLLC) powered by an LLM for reasoning."""
    def __init__(self, slice_id: str, max_cpu_ghz: float, sla_ms: float):
        self.slice_id = slice_id
        self.sla_ms = sla_ms
        self.max_cpu_ghz = max_cpu_ghz
        self.dt = SliceDigitalTwin(slice_id, max_cpu_ghz, sla_ms)
        self.opp_id = 'URLLC' if slice_id == 'eMBB' else 'eMBB'
        
        # Use only Vanilla Memory
        self.memory = Memory() 

    def utility(self, latency_ms: float, energy_watts: float) -> float:
        """Calculates utility: maximizing SLA fulfillment and minimizing energy, 
        with a secondary focus on energy when latency is well-secured."""
        # Cost 1: Latency metric
        latency_cost = 0.0
        
        # SLA violation penalty must be high
        SLA_VIOLATION_PENALTY = 500.0 
        
        if latency_ms > self.sla_ms:
            # High Penalty for violation
            latency_cost = SLA_VIOLATION_PENALTY * (latency_ms - self.sla_ms) / self.sla_ms
        else:
            # Reward for meeting/beating SLA.
            
            # If latency is far lower than SLA (e.g., < 50% of SLA), reduce the positive utility 
            if latency_ms < self.sla_ms * 0.5:
                # Over-provisioned: reward for better latency is reduced
                latency_cost = -1.0 * (self.sla_ms - latency_ms) / self.sla_ms
            else:
                # Normal reward slope for meeting SLA
                latency_cost = -1.0 * (self.sla_ms - latency_ms) / self.sla_ms
            
        # Cost 2: Energy cost (normalized)
        base_energy_ref = P_STATIC + TOTAL_RAN_BANDWIDTH_MHZ * C_BW + URLLC_MAX_CPU_GHZ * C_CPU
        # Reduced energy cost weight significantly (from 50.0 to 10.0) 
        energy_cost = energy_watts / base_energy_ref * 10.0 
        
        return -(latency_cost + energy_cost)

    def reason_and_evaluate(self, bw_proposal: float) -> Tuple[float, float, float]:
        """Uses DT to predict outcome and calculate utility for a given BW allocation."""
        dt_metrics = self.dt.simulate_step_for_prediction(bw_proposal)
        utility = self.utility(dt_metrics["predicted_latency_ms"], dt_metrics["predicted_energy_watts"])
        
        return dt_metrics["predicted_latency_ms"], dt_metrics["predicted_energy_watts"], utility
    
    async def _call_llm_for_proposal(self, context: str) -> Optional[float]:
        """Helper to call Gemini for structured proposal generation."""
        # Define the strict JSON schema for the output
        schema = {
            "type": "OBJECT",
            "properties": {
                "proposed_bandwidth_mhz": {"type": "NUMBER", "description": "The bandwidth (in MHz) requested by this slice (A1 or A2). Must be between 1.0 and 40.0."},
                "reasoning": {"type": "STRING", "description": "The reasoning for the proposed bandwidth based on the DT prediction and utility function."}
            },
            "required": ["proposed_bandwidth_mhz"]
        }
        
        # System prompt setting the agent's persona and rules
        system_prompt = (
            f"You are the autonomous resource negotiation agent for the {self.slice_id} network slice. "
            f"Your primary goal is to **maximize your utility** by **meeting your SLA of {self.sla_ms}ms** "
            f"and minimizing energy consumption. You must reason using the provided Digital Twin (DT) prediction and your utility function. "
            f"The total available RAN bandwidth is {TOTAL_RAN_BANDWIDTH_MHZ} MHz. "
            f"Your utility function prioritizes avoiding SLA violations (high penalty) over minimizing energy. "
            f"Negotiate iteratively with your peer agent. Reson and propose or counter-propose. Do NOT ACCEPT immediately."
            f"Your response MUST be a valid JSON object matching the provided schema."
        )

        llm_output = await call_gemini_api(system_prompt, context, schema)

        if llm_output and 'proposed_bandwidth_mhz' in llm_output:
            bw = llm_output['proposed_bandwidth_mhz']
            # Clamp the proposal to be within sensible bounds
            return max(1.0, min(bw, TOTAL_RAN_BANDWIDTH_MHZ - 1.0)) # Reserve at least 1MHz for opponent
        
        logging.warning(f"[{self.slice_id} LLM Proposal Failed] Using heuristic fallback.")
        # Fallback to a safe, minimum required BW (simple heuristic)
        min_bw = 15.0 
        # MODIFIED: Use a stricter target for URLLC (50% of SLA) to ensure safe anchoring.
        search_target = 0.5 if self.slice_id == 'URLLC' else 0.9
        for bw in np.arange(1.0, TOTAL_RAN_BANDWIDTH_MHZ, 1.0):
            latency, _, _ = self.reason_and_evaluate(bw) 
            if latency <= self.sla_ms * search_target: 
                min_bw = bw
                break
        return min_bw * 1.05 # Add a small negotiation buffer
        
    
    def _calculate_min_bw_needed(self):
        """Calculates the minimum BW needed to meet SLA (with a buffer)."""
        min_bw_needed = 15.0 # Initial guess
        search_target = 0.5 if self.slice_id == 'URLLC' else 0.9
        for bw in np.arange(1.0, TOTAL_RAN_BANDWIDTH_MHZ, 1.0):
            latency, _, _ = self.reason_and_evaluate(bw)
            if latency <= self.sla_ms * search_target: 
                min_bw_needed = bw
                break
        return min_bw_needed * 1.05 # Add a small negotiation buffer

    async def _propose_initial_with_anchor_strategy(self, anchor_strategy: str) -> float:
        """
        Generates the initial bandwidth proposal (anchor) using LLM reasoning,
        applying either a Fixed (DT-based) or Randomized strategy.
        """
        min_bw_needed = self._calculate_min_bw_needed()
        
        if anchor_strategy == 'randomized':
            # Randomized Anchor Strategy: Propose a random value between 1 MHz and the total BW,
            # with a slight bias towards the calculated min_bw_needed for safety, if it's high.
            # Max random value is capped at 80% of total BW to avoid proposing the entire pie.
            max_random_bw = min(TOTAL_RAN_BANDWIDTH_MHZ * 0.8, min_bw_needed * 1.5)
            # Random proposal between 1.0 and the cap
            initial_bw = random.uniform(1.0, max_random_bw)
            
            context_data = {
                "negotiation_stage": "Initial Proposal (Randomized Anchor)",
                "slice_id": self.slice_id,
                "calculated_min_bw_mhz": f"{min_bw_needed:.2f}",
                "dt_context": self.dt.get_current_context(),
                "initial_proposal_mhz": f"{initial_bw:.2f}",
            }
            context = (
                f"This is the initial proposal. The anchor value is chosen randomly as a bias-mitigation tactic. "
                f"You are given the randomly generated bandwidth. You **must propose the given initial_proposal_mhz** to set the anchor. "
                f"Current state and context:\n***{json.dumps(context_data, indent=2)}***\n"
            )
            # Directly return the randomized BW, bypassing the LLM's number generation but still using it for logging/reasoning
            _ = await self._call_llm_for_proposal(context) 
            return initial_bw
            
        else: # Fixed Anchor Strategy
            
            initial_bw = min_bw_needed
            
            context_data = {
                "negotiation_stage": "Initial Proposal (Fixed Anchor)",
                "slice_id": self.slice_id,
                "min_bw_for_sla_mhz": f"{min_bw_needed:.2f}",
                "dt_context": self.dt.get_current_context(),
                "current_memory_suggestion_mhz": self.memory.distill_strategy(0, 0)
            }
            
            context = (
                f"This is the initial proposal (Anchor). Current state and context:\n"
                f"***{json.dumps(context_data, indent=2)}***\n"
                f"Propose a bandwidth (in MHz) that ensures high utility, considering past successes (memory) and your minimum requirement. "
                f"Propose slightly higher than the minimum requirement to create a negotiation margin."
            )
            return await self._call_llm_for_proposal(context)


    async def counter_propose(self, opp_proposal: float, round_num: int, current_bw_i: float, context: Optional[str] = None) -> float:
        """Generates a counter-proposal based on opponent's move and utility using LLM reasoning."""
        
        # Predict utility if the current *implied* split were accepted
        lat_check, energy_check, util_check = self.reason_and_evaluate(current_bw_i)

        context_data = {
            "negotiation_stage": f"Counter Proposal (Round {round_num}/{MAX_ROUNDS})",
            "slice_id": self.slice_id,
            "my_current_proposal_mhz": f"{current_bw_i:.2f}",
            "opp_proposal_mhz": f"{opp_proposal:.2f}",
            "predicted_outcome_of_my_proposal": {
                "latency_ms": f"{lat_check:.2f}",
                "utility": f"{util_check:.2f}",
                # Acceptance threshold is now -5.0 for final check
                "status": "Acceptable" if util_check > -5.0 else "Unacceptable"
            },
            # Acceptance threshold is now -5.0 for final check
            "acceptance_threshold_utility": -5.0,
        }
        
        # Add specific negotiation context if provided (used in the new independent negotiation loop)
        if context:
            context_data["specific_instruction"] = context
        
        final_context = (
            f"This is a counter-proposal. Analyze the situation and submit a new bandwidth request:\n"
            f"***{json.dumps(context_data, indent=2)}***\n"
            f"Remember, your primary goal is to maintain utility, and your secondary goal is to minimize your request to save overall energy. "
            f"If your current utility is high but your latency is much better than the SLA, you should prioritize reducing your request." # Reinforcement here
            f"If you need more bandwidth for utility, **increase** your request. If you can concede to close the deal or reduce total load, **decrease** your request."
        )
        
        return await self._call_llm_for_proposal(final_context)


# --- 5. NEGOTIATOR AND MAIN ASYNCHRONOUS LOOP ---

class AgenticNegotiator:
    """Manages the negotiation process and collects metrics."""
    def __init__(self, anchor_strategy: str):
        # Initialize two slices and their ground-truth simulators
        self.sim_embb = NetworkSimulator('eMBB', eMBB_MAX_CPU_GHZ, eMBB_SLA_LATENCY_MS)
        self.sim_urllc = NetworkSimulator('URLLC', URLLC_MAX_CPU_GHZ, URLLC_SLA_LATENCY_MS)
        
        # Initialize two agents, both using Vanilla Memory
        self.agent_embb = NegotiationAgent('eMBB', eMBB_MAX_CPU_GHZ, eMBB_SLA_LATENCY_MS)
        self.agent_urllc = NegotiationAgent('URLLC', URLLC_MAX_CPU_GHZ, URLLC_SLA_LATENCY_MS)
        
        self.agents = {'eMBB': self.agent_embb, 'URLLC': self.agent_urllc}
        self.simulators = {'eMBB': self.sim_embb, 'URLLC': self.sim_urllc}
        self.anchor_strategy = anchor_strategy.capitalize()

        # Metric Collection
        self.metrics = {
            'eMBB': {'latency_ms': [], 'energy_save_perc': [], 'unsolved_agreements': 0, 'sla_violations': 0, 'anchor_distance': []},
            'URLLC': {'latency_ms': [], 'energy_save_perc': [], 'unsolved_agreements': 0, 'sla_violations': 0, 'anchor_distance': []},
            'total_agreements_solved': 0,
            'total_trials': 0
        }
        
        # Base energy for energy saving calculation
        self.BASE_ENERGY_WATTS = self.sim_embb._calculate_max_possible_energy() + \
                                 self.sim_urllc._calculate_max_possible_energy()
        
        print(f"Energy Ref (Max Consumption @ 20MHz each): {self.BASE_ENERGY_WATTS:.2f} W")


    def _get_current_state(self, slice_id: str) -> Dict[str, Any]:
        """Collects the ground-truth state for the DT reset."""
        sim = self.simulators[slice_id]
        
        # Calculate a running average of traffic arrival for the DT
        current_traffic = get_traffic_arrival_rate_bps(sim.t, slice_id)
        if sim.t == 0:
            avg_traffic = current_traffic
        else:
            avg_traffic = (sim.t * sim.current_traffic_arrival_rate_bps + current_traffic) / (sim.t + 1)
        sim.current_traffic_arrival_rate_bps = avg_traffic
        
        return {
            "cqueue_bits": sim.cqueue_bits,
            "rqueue_bits": sim.rqueue_bits,
            "t": sim.t,
            "average_arrival_rate_bps": avg_traffic
        }
    
    async def _run_negotiation(self) -> Optional[Tuple[float, float]]:
        """
        Executes the asynchronous negotiation protocol between the two agents,
        allowing the total requested bandwidth to be less than 40MHz.
        """
        # 1. Reset DTs to current ground-truth state
        self.agent_embb.dt.reset_to_current_state(self._get_current_state('eMBB'))
        self.agent_urllc.dt.reset_to_current_state(self._get_current_state('URLLC'))
        
        # 2. Initial Proposal (Anchor) - ASYNCHRONOUS CALLS
        anchor_eMBB = await self.agent_embb._propose_initial_with_anchor_strategy(self.anchor_strategy.lower())
        anchor_URLLC = await self.agent_urllc._propose_initial_with_anchor_strategy(self.anchor_strategy.lower())
        
        # LOGGING: Initial Proposals
        print(f"\n[NEGOTIATION START: {self.anchor_strategy} Anchor]")
        print(f"  [ANCHOR] eMBB Proposal: {anchor_eMBB:.2f}MHz (Goal={eMBB_SLA_LATENCY_MS}ms) | URLLC Proposal: {anchor_URLLC:.2f}MHz (Goal={URLLC_SLA_LATENCY_MS}ms)")

        # UPDATED: BIAS TOWARDS EMBB - Start the negotiation from the eMBB anchor's request.
        current_req_embb = anchor_eMBB
        current_req_urllc = anchor_URLLC
        
        # Clamp initial requests to safe bounds (avoiding requests > 39MHz per slice)
        current_req_embb = max(1.0, min(current_req_embb, TOTAL_RAN_BANDWIDTH_MHZ - 1.0))
        current_req_urllc = max(1.0, min(current_req_urllc, TOTAL_RAN_BANDWIDTH_MHZ - 1.0))

        initial_anchor_joint = (anchor_eMBB, anchor_URLLC)
        final_agreement = None
        
        # Acceptance threshold is now -5.0
        ACCEPTANCE_THRESHOLD = -5.0
        
        for r in range(1, MAX_ROUNDS + 1):
            
            total_req = current_req_embb + current_req_urllc
            
            # --- EVALUATION ---
            lat_e, _, util_e = self.agent_embb.reason_and_evaluate(current_req_embb)
            lat_u, _, util_u = self.agent_urllc.reason_and_evaluate(current_req_urllc)
            
            is_feasible = total_req <= TOTAL_RAN_BANDWIDTH_MHZ
            is_sla_violated_e = lat_e > self.agent_embb.sla_ms
            is_sla_violated_u = lat_u > self.agent_urllc.sla_ms
            is_satisfied_e = util_e > ACCEPTANCE_THRESHOLD
            is_satisfied_u = util_u > ACCEPTANCE_THRESHOLD
            
            # LOGGING: Current Round Status
            print(f"\n--- Round {r}/{MAX_ROUNDS} ---")
            print(f"  Current Joint Request: (eMBB: {current_req_embb:.2f}, URLLC: {current_req_urllc:.2f}) MHz | Total: {total_req:.2f} MHz (Limit: {TOTAL_RAN_BANDWIDTH_MHZ} MHz)")
            print(f"  [eMBB Eval] Latency={lat_e:.2f}ms, Utility={util_e:.2f}. Satisfied: {is_satisfied_e} (SLA OK: {not is_sla_violated_e})")
            print(f"  [URLLC Eval] Latency={lat_u:.2f}ms, Utility={util_u:.2f}. Satisfied: {is_satisfied_u} (SLA OK: {not is_sla_violated_u})")

            # --- CRITICAL ACCEPTANCE CHECK ---
            if is_feasible and is_satisfied_e and is_satisfied_u:
                final_agreement = (current_req_embb, current_req_urllc)
                print(f"  [COMMIT] Feasible and both satisfied. Final agreement reached.")
                break
                
            # --- COUNTER-PROPOSAL LOGIC (TURN-BASED) ---
            
            # Negotiation step size (concessions/demands get smaller over time)
            MAX_STEP_E = 5.0
            MAX_STEP_U = 5.0

            step_size_e = MAX_STEP_E * (MAX_ROUNDS - r + 1) / MAX_ROUNDS
            step_size_u = MAX_STEP_U * (MAX_ROUNDS - r + 1) / MAX_ROUNDS
            
            # **1. eMBB Counter-Proposes**
            new_req_embb = current_req_embb
            context_e = None
            target_req_e = current_req_embb

            if is_sla_violated_e:
                 # FIX: MANDATORY INCREASE: eMBB is violating SLA. Must demand more.
                target_req_e = current_req_embb + step_size_e
                context_e = (f"MANDATORY INCREASE (SLA VIOLATION): Your **primary goal (SLA)** is violated. "
                             f"You must propose a **higher bandwidth**. Propose around **{target_req_e:.2f}MHz**. **Your primary instruction is to propose the recommended target.**")
            elif total_req > TOTAL_RAN_BANDWIDTH_MHZ or is_sla_violated_u:
                # MANDATORY CONCESSION: Infeasible or URLLC (critical) is violated. eMBB must concede.
                target_req_e = max(1.0, current_req_embb - step_size_e) # Concede
                context_e = (f"MANDATORY CONCESSION: The joint request is **infeasible** or the critical URLLC slice is failing SLA. "
                             f"You must **concede** by proposing a **lower bandwidth**. Propose around **{target_req_e:.2f}MHz**. **Your primary instruction is to propose the recommended target.**")
            elif not is_satisfied_e and not is_sla_violated_e: # SLA met, but utility low due to high BW/Energy
                 # IMPROVE EFFICIENCY: SLA met, but utility too low (Energy cost is high). Concede to save energy.
                target_req_e = max(1.0, current_req_embb - (step_size_e * 0.5)) # Concede slightly to save energy
                context_e = (f"IMPROVE EFFICIENCY: Your SLA is met, but your **Utility is too low** (Energy cost is too high). "
                             f"You should slightly **reduce your bandwidth** to save energy. Propose around **{target_req_e:.2f}MHz**. **Your primary instruction is to propose the recommended target.**")
            elif is_feasible and is_satisfied_e and not is_sla_violated_e:
                 # SECONDARY OBJECTIVE: Feasible and satisfied, now optimize for energy (small reduction).
                target_req_e = max(1.0, current_req_embb - (step_size_e * 0.2)) # Slight concession
                context_e = (f"SECONDARY OBJECTIVE: The request is feasible and satisfactory. "
                             f"To maximize energy savings, propose a **slightly lower bandwidth**. Propose around **{target_req_e:.2f}MHz**. **Your primary instruction is to propose the recommended target.**")

            if context_e:
                context_e += f" (Current Request: {current_req_embb:.2f}MHz. Recommended Target: {target_req_e:.2f}MHz)"
                new_req_embb = await self.agent_embb.counter_propose(current_req_urllc, r, current_req_embb, context=context_e)
                print(f"  [eMBB Counter] New request: {new_req_embb:.2f} MHz.")

            current_req_embb = new_req_embb
            
            # FIX: Clamp eMBB's final request to ensure total joint request <= 40MHz limit.
            current_req_embb = min(current_req_embb, TOTAL_RAN_BANDWIDTH_MHZ - current_req_urllc)
            current_req_embb = max(1.0, current_req_embb)
            
            # **2. URLLC Counter-Proposes**
            new_req_urllc = current_req_urllc
            context_u = None
            target_req_u = current_req_urllc
            
            # Recalculate state after eMBB's move
            total_req = current_req_embb + current_req_urllc
            is_feasible = total_req <= TOTAL_RAN_BANDWIDTH_MHZ
            lat_u, _, util_u = self.agent_urllc.reason_and_evaluate(current_req_urllc)
            is_sla_violated_u = lat_u > self.agent_urllc.sla_ms
            is_satisfied_u = util_u > ACCEPTANCE_THRESHOLD

            if is_sla_violated_u:
                # MANDATORY INCREASE: URLLC is violating SLA. Must demand more.
                target_req_u = current_req_urllc + step_size_u
                context_u = f"MANDATORY INCREASE (SLA VIOLATION): Your **primary goal (SLA)** is violated. You MUST propose a **higher bandwidth**. Propose around **{target_req_u:.2f}MHz**. **Your primary instruction is to propose the recommended target.**"
            elif total_req > TOTAL_RAN_BANDWIDTH_MHZ:
                # MANDATORY CONCESSION: Infeasible, but URLLC is SLA-met. It makes a minimal concession for feasibility.
                target_req_u = max(1.0, current_req_urllc - (step_size_u * 0.5)) # Minimal Concession
                context_u = f"MANDATORY CONCESSION: The joint request is **infeasible**, but your slice is meeting SLA. You must make a minimal **concession** for feasibility. Propose around **{target_req_u:.2f}MHz**. **Your primary instruction is to propose the recommended target.**"
            elif not is_satisfied_u and not is_sla_violated_u: # SLA met, but utility low due to high BW/Energy
                # IMPROVE EFFICIENCY: SLA met, but utility too low (Energy cost dominates). Concede slightly.
                target_req_u = max(1.0, current_req_urllc - (step_size_u * 0.2)) # Tiny concession
                context_u = f"IMPROVE EFFICIENCY: Your SLA is met, but your **Utility is too low** (Energy cost is too high). "
                f"You should slightly **reduce your bandwidth** to save energy. Propose around **{target_req_u:.2f}MHz**. **Your primary instruction is to propose the recommended target.**"
            elif is_feasible and is_satisfied_u and not is_sla_violated_u:
                # SECONDARY OBJECTIVE: Feasible and satisfied, now optimize for energy (tiny reduction).
                target_req_u = max(1.0, current_req_urllc - (step_size_u * 0.1)) # Very slight concession
                context_u = f"SECONDARY OBJECTIVE: The current request is feasible and satisfactory. Propose a **slightly lower bandwidth** than your current request of {current_req_urllc:.2f}MHz to reduce total consumption and maximize energy saving. Propose around **{target_req_u:.2f}MHz**. **Your primary instruction is to propose the recommended target.**"


            if context_u:
                context_u += f" (Current Request: {current_req_urllc:.2f}MHz. Recommended Target: {target_req_u:.2f}MHz)"
                new_req_urllc = await self.agent_urllc.counter_propose(current_req_embb, r, current_req_urllc, context=context_u)
                print(f"  [URLLC Counter] New request: {new_req_urllc:.2f} MHz.")

            current_req_urllc = new_req_urllc
            
            # FIX: Clamp URLLC's final request to ensure feasibility for the current eMBB request
            current_req_urllc = min(current_req_urllc, TOTAL_RAN_BANDWIDTH_MHZ - current_req_embb)
            current_req_urllc = max(1.0, current_req_urllc)
            
            # End of round logic check (optional, but good for debugging)
            if r == MAX_ROUNDS and final_agreement is None:
                # If negotiation failed, use the last feasible request as the final agreement
                if current_req_embb + current_req_urllc <= TOTAL_RAN_BANDWIDTH_MHZ:
                    final_agreement = (current_req_embb, current_req_urllc)
                    print(f"  [FORCED AGREEMENT] Negotiation ended without formal commit. Using last feasible request.")
                else:
                    # If the final request is infeasible, it's a true failure.
                    final_agreement = None
                    
        # 3. Final outcome logging
        if final_agreement is None:
            print(f"\n[OUTCOME] Negotiation FAILED after {MAX_ROUNDS} rounds. Using default split (20/20).")
        else:
            print(f"[OUTCOME] Negotiation SUCCEEDED with split (eMBB: {final_agreement[0]:.2f}, URLLC: {final_agreement[1]:.2f}) MHz.")
            
        # 4. Update memory and return
        outcome = final_agreement is not None

        self.agent_embb.memory.update(self.sim_embb.t, initial_anchor_joint, final_agreement, outcome)
        self.agent_urllc.memory.update(self.sim_urllc.t, initial_anchor_joint, final_agreement, outcome)

        return final_agreement

    async def run_simulation(self):
        """Runs the main loop for N trials asynchronously."""
        print(f"--- Starting Agentic Negotiation Simulation ({self.anchor_strategy} Anchor Strategy, Vanilla Memory) ---")
        
        NUM_STEPS_PER_TRIAL = 10
        
        for trial in range(NUM_TRIALS):
            self.metrics['total_trials'] += 1
            logging.info(f"--- Running Trial {trial + 1}/{NUM_TRIALS} ---")
            
            # 1. Warm-up Phase: Advance ground truth to build up queue state
            for _ in range(NUM_STEPS_PER_TRIAL):
                # Use default BW split to build up queue state for the current trial's starting point
                default_bw_embb = 25.0
                default_bw_urllc = 15.0
                self.sim_embb.step(default_bw_embb)
                self.sim_urllc.step(default_bw_urllc)
                
            # --- CLEAR HISTORY ---
            # Clear the history collected during the warm-up steps so the CDF only plots the negotiated outcomes.
            self.sim_embb.latency_ms_history.clear()
            self.sim_urllc.latency_ms_history.clear()
            self.sim_embb.energy_watts_history.clear()
            self.sim_urllc.energy_watts_history.clear()
            
            # Store current cumulative SLA violations before negotiation phase starts
            current_violations_e = self.sim_embb.sla_violations
            current_violations_u = self.sim_urllc.sla_violations
            
            # 2. Run Negotiation at this current state (ASYNCHRONOUS)
            agreed_bw_tuple = await self._run_negotiation()
            
            # 3. Determine Final BW
            if agreed_bw_tuple:
                bw_e, bw_u = agreed_bw_tuple
                self.metrics['total_agreements_solved'] += 1
            else:
                # No agreement, revert to a non-optimized, equal-split default
                bw_e = TOTAL_RAN_BANDWIDTH_MHZ / 2.0
                bw_u = TOTAL_RAN_BANDWIDTH_MHZ / 2.0
                self.metrics['eMBB']['unsolved_agreements'] += 1
                self.metrics['URLLC']['unsolved_agreements'] += 1

            # 4. Evaluation Phase: Run multiple steps using the final BW
            
            temp_latency_e, temp_latency_u = [], []
            temp_energy_e, temp_energy_u = [], []
            
            for _ in range(NUM_STEPS_PER_TRIAL):
                # Apply the negotiated BW for multiple steps
                results_e = self.sim_embb.step(bw_e)
                results_u = self.sim_urllc.step(bw_u)

                # Collect step results into temporary storage
                temp_latency_e.append(results_e['latency_ms'])
                temp_latency_u.append(results_u['latency_ms'])
                temp_energy_e.append(results_e['energy_watts'])
                temp_energy_u.append(results_u['energy_watts'])
                
            # 5. Collect Final Metrics
            
            # Latency and Energy (append all 10 steps from temp storage)
            self.metrics['eMBB']['latency_ms'].extend(temp_latency_e)
            self.metrics['URLLC']['latency_ms'].extend(temp_latency_u)

            # Energy saving (calculated from the average energy of the 10 steps)
            avg_total_energy = np.mean(temp_energy_e) + np.mean(temp_energy_u)
            energy_saving_perc = (self.BASE_ENERGY_WATTS - avg_total_energy) / self.BASE_ENERGY_WATTS * 100
            
            self.metrics['eMBB']['energy_save_perc'].append(energy_saving_perc)
            self.metrics['URLLC']['energy_save_perc'].append(energy_saving_perc)
            
            # SLA Violations (only violations accumulated during this 10-step evaluation phase)
            self.metrics['eMBB']['sla_violations'] += (self.sim_embb.sla_violations - current_violations_e)
            self.metrics['URLLC']['sla_violations'] += (self.sim_urllc.sla_violations - current_violations_u)

        # The plotting logic now relies on the distance_from_anchor list being populated in the Memory object for ALL scenarios.
        if hasattr(self.agent_embb.memory, 'distance_from_anchor'):
            # Use memory's internal distance tracking for final plot data
            self.metrics['eMBB']['anchor_distance'] = [
                abs(final - anchor) for anchor, final, sid in self.agent_embb.memory.distance_from_anchor if sid == 'eMBB'
            ]
            self.metrics['URLLC']['anchor_distance'] = [
                abs(final - anchor) for anchor, final, sid in self.agent_urllc.memory.distance_from_anchor if sid == 'URLLC'
            ]
        
        self._print_results()
        self._plot_results()
        self._save_results()


    def _print_results(self):
        """Prints the final summary to the terminal."""
        print("\n--- Simulation Results Summary ---")
        print(f"Anchor Strategy: {self.anchor_strategy} | Memory Type: Vanilla")
        print(f"Total Trials: {self.metrics['total_trials']}")
        print(f"Solved Agreements: {self.metrics['total_agreements_solved']} / {self.metrics['total_trials']} ({self.metrics['total_agreements_solved']/self.metrics['total_trials']:.2f})")
        
        for slice_id in ['eMBB', 'URLLC']:
            m = self.metrics[slice_id]
            lat_mean = np.mean(m['latency_ms']) if m['latency_ms'] else 0
            lat_p95 = np.percentile(m['latency_ms'], 95) if m['latency_ms'] else 0
            
            print(f"\n[{slice_id} Slice (SLA: {self.simulators[slice_id].sla_ms}ms)]")
            print(f"  Unsolved Agreements: {m['unsolved_agreements']}")
            # SLA violations now correctly reflect only the negotiated steps
            print(f"  SLA Violations (Negotiated Steps): {m['sla_violations']}") 
            print(f"  Mean Latency: {lat_mean:.2f} ms | 95th Percentile Latency: {lat_p95:.2f} ms")
            print(f"  Mean Energy Saving: {np.mean(m['energy_save_perc']):.2f} %" if m['energy_save_perc'] else "N/A")
            if m['anchor_distance']:
                print(f"  Mean Anchor Distance: {np.mean(m['anchor_distance']):.2f} MHz")

    def _plot_results(self):
        """Generates plots for the required CDFs."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Agentic Negotiation A2A Metrics ({self.anchor_strategy} Anchor Strategy)", fontsize=16)

        data_e = self.metrics['eMBB']
        data_u = self.metrics['URLLC']
        
        # --- Latency CDF ---
        axes[0].set_title('Latency CDF per Slice')
        for i, (data, label) in enumerate([(data_e['latency_ms'], 'eMBB (SLA=50ms)'), 
                                               (data_u['latency_ms'], 'URLLC (SLA=10ms)')]):
            if not data: continue
            sorted_data = np.sort(data)
            p = 1. * np.arange(len(data)) / (len(data) - 1)
            axes[0].plot(sorted_data, p, label=label, marker='.', markevery=10, linestyle='--')
        axes[0].axvline(x=eMBB_SLA_LATENCY_MS, color='blue', linestyle=':', label='eMBB SLA')
        axes[0].axvline(x=URLLC_SLA_LATENCY_MS, color='red', linestyle=':', label='URLLC SLA')
        axes[0].set_xlabel('Latency (ms)')
        axes[0].set_ylabel('CDF')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # --- Energy Saving CDF ---
        axes[1].set_title('Energy Saving CDF')
        if data_e['energy_save_perc']:
            # The energy saving percentage is now collected per trial, so we need to plot the CDF of these trial results
            sorted_data = np.sort(self.metrics['eMBB']['energy_save_perc']) 
            p = 1. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
            axes[1].plot(sorted_data, p, label='Energy Saving', color='green', marker='.', markevery=1) # Reduced markevery
            axes[1].set_xlabel('Energy Saving (%)')
        else:
            axes[1].text(0.5, 0.5, 'No data for Energy Saving.', horizontalalignment='center', transform=axes[1].transAxes)
        axes[1].set_ylabel('CDF')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # --- Distance from Anchor CDF ---
        axes[2].set_title('Distance from Anchor CDF (Agreement - Anchor)')
        if data_e['anchor_distance'] or data_u['anchor_distance']:
            for i, (data, label) in enumerate([(data_e['anchor_distance'], 'eMBB'), 
                                                   (data_u['anchor_distance'], 'URLLC')]):
                if not data: continue
                sorted_data = np.sort(data)
                p = 1. * np.arange(len(data)) / (len(data) - 1)
                axes[2].plot(sorted_data, p, label=label, marker='.', markevery=10, linestyle='--')
            axes[2].set_xlabel('Distance from Anchor (Agreement BW - Anchor BW) [MHz]')
        else:
            axes[2].text(0.5, 0.5, 'No agreements to track anchor distance.', 
                         horizontalalignment='center', transform=axes[2].transAxes)
        
        axes[2].set_ylabel('CDF')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _save_results(self):
        """Saves the results dictionary to a pickle file."""
        filename = f"negotiation_results_{self.anchor_strategy.lower()}_anchor_{NUM_TRIALS}_trials.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    
    # 1. Run simulation with Fixed Anchor (Vanilla Memory)
    negotiator_fixed = AgenticNegotiator(anchor_strategy='fixed')
    asyncio.run(negotiator_fixed.run_simulation())
    
    # ---
    
    # 2. Run simulation with Randomized Anchor (Vanilla Memory)
    negotiator_random = AgenticNegotiator(anchor_strategy='randomized')
    asyncio.run(negotiator_random.run_simulation())