# RL Dynamic Pricing Environment Blueprint (Marketplace Delivery/Rides)

This document is a production-oriented blueprint for building a reinforcement-learning (RL) dynamic pricing system in a live marketplace (delivery/ride-hailing style).

---

## 1) State Space (What the agent sees)

Use **zone-time aggregated states** every decision interval (e.g., 5 minutes per zone). Keep state interpretable and stable before adding deep features.

### A. Demand indicators
- `orders_last_5m`, `orders_last_30m` (arrival intensity)
- `search_to_order_rate` (conversion proxy)
- `unserved_requests` (backlog)
- `tod_sin, tod_cos`, `dow_onehot`, `is_holiday`

**Why it matters**
- Pricing should react to short-term demand spikes but also daily/weekly seasonality.
- Backlog is an immediate scarcity signal and often the strongest short-horizon predictor of needed surge.

### B. Supply indicators
- `active_drivers`, `idle_drivers`, `available_capacity`
- `median_pickup_eta`, `p90_pickup_eta`
- `acceptance_rate_last_15m`
- `driver_utilization_last_30m`

**Why it matters**
- Supply friction (high ETA, low idle count) directly drives rider wait time and cancellations.
- Utilization indicates whether pricing needs to attract more drivers or prevent oversupply.

### C. Price and policy state
- `base_price`, `current_surge`
- `effective_fare_index` (fare relative to rolling baseline)
- `price_change_velocity` (how fast multiplier changed recently)
- `compensation_floor_binding` flag

**Why it matters**
- Demand response depends on absolute and relative price levels.
- Fast policy oscillations degrade trust and can trigger gaming behavior.

### D. User behavior / elasticity proxies
- `cancel_prob_model_score`
- `elasticity_bucket` (estimated from historical cohorts)
- `repeat_user_share`, `promo_dependency_rate`
- `churn_risk_proxy_7d`

**Why it matters**
- Same surge produces different outcomes across neighborhoods and user cohorts.
- Captures long-term customer health, not just immediate revenue.

### E. Context and exogenous signals
- `rain_intensity`, `temperature_bin`, `severe_weather_flag`
- `event_flag` (concert, game, transit outage)
- `traffic_index`
- `zone_cluster_id` (urban core/suburban/airport)

**Why it matters**
- These shocks move both demand and supply simultaneously.
- If omitted, agent over-attributes shocks to price and overfits.

### Practical state design notes
- Start with ~40–120 engineered features per zone-step.
- Avoid fully raw event streams in v1; aggregate to robust statistics.
- Version every feature and keep data lineage so you can replay training exactly.

---

## 2) Action Space (What the agent controls)

### Recommended production action
Use a **discrete multiplier ladder** with monotonic constraints, e.g.:
- `surge ∈ {1.00, 1.05, 1.10, ..., 2.50}`
- Optional "hold" action to minimize unnecessary changes.

### Why discrete first (vs continuous)
- Easier policy interpretation and auditability.
- Safer rollout with bounded outcomes.
- Simpler counterfactual evaluation from logged data.

### Constraints
- Hard bounds: `1.0 <= surge <= 2.5`
- Step-change limit: `|Δ surge| <= 0.15 per interval`
- Cooldown: no more than 1 change every N intervals unless guardrail breach.
- Regulatory/product constraints: special cap for vulnerable zones.

### Action frequency
- Typical: every 5 minutes per zone.
- Faster (1 min): reacts quickly but can induce volatility/noise chasing.
- Slower (15 min): stable but may miss shocks.

**Rule of thumb**: if ETA is your main pain point, shorter intervals help; if trust/price stability is your main pain point, lengthen interval and penalize volatility.

---

## 3) Reward Function (What "good" means)

Use a weighted multi-objective reward with explicit penalties.

\[
R_t = w_1 \cdot \text{net_revenue}_t
    + w_2 \cdot \text{fulfilled_orders}_t
    + w_3 \cdot \text{supply_health}_t
    - w_4 \cdot \text{cancellations}_t
    - w_5 \cdot \text{price_extremity}_t
    - w_6 \cdot \text{price_volatility}_t
\]

Where:
- `net_revenue = completed_orders * fare - incentives - support/refund_cost`.
- `supply_health` can be a bounded utility of utilization and ETA (not raw utilization alone).
- `price_extremity = max(0, surge - comfort_cap)^2`.
- `price_volatility = (surge_t - surge_{t-1})^2`.

### Reward shaping choices
- Normalize each component (z-score or robust percentile scaling) before weighting.
- Keep signs intuitive: positive means business value, negative means harm.
- Add delayed terms (e.g., churn proxy at t+1 day) via eligibility traces or delayed reward joins.

### Avoid reward hacking
- If agent can inflate revenue by overpricing and shrinking demand, completion/retention penalties must be strong.
- Add hard constraints outside reward (kill-switches and caps), not only soft penalties.
- Evaluate by decomposed metrics, not just scalar reward.

### Short vs long term incentives
- Short term: margin, fulfillment, ETAs.
- Long term: repeat rate, 7/28-day churn, fairness complaints.
- Use blended objective and monitoring dashboards for both horizons.

---

## 4) Environment Dynamics (Simplified but realistic simulation)

At each step for zone `z`:

1. **Demand arrivals**
\[
\lambda_{t,z} = \lambda^0_{t,z}(context) \cdot \exp(-\beta_z (surge_{t,z} - 1)) \cdot \epsilon^{(d)}_{t,z}
\]
- `λ^0` from historical baseline model.
- `β_z` elasticity coefficient (segment-specific).
- `ε(d)` lognormal or Gamma noise.

2. **Driver supply response (with lag)**
\[
S_{t+1,z} = S_{t,z} + \alpha_z (surge_{t,z} - 1) - \delta_z congestion_{t,z} + \epsilon^{(s)}_{t,z}
\]
- Supply response is delayed and bounded by nearby-zone rebalancing.

3. **Matching and fulfillment**
\[
fulfilled_{t,z} = \min(arrivals_{t,z} + backlog_{t,z},\ capacity(S_{t,z}, ETA_{t,z}))
\]

4. **Cancellation probability**
\[
p_{cancel} = \sigma(c_0 + c_1 \cdot ETA + c_2 \cdot surge + c_3 \cdot user\_elasticity)
\]

5. **State transition**
- Backlog updates with unfulfilled demand.
- ETAs worsen with utilization/congestion.
- Observations may be delayed/noisy to mimic real telemetry delays.

### Non-stationarity
- Seasonal multipliers by month/week.
- Trend term for platform growth.
- Shock process for weather/events.
- Periodic model refresh (weekly) to avoid stale dynamics.

---

## 5) Algorithm Choice (Opinionated recommendation)

### Start simple: contextual bandit / conservative policy improvement
For many pricing teams, this beats full RL initially because:
- Easier offline validation.
- Lower variance and safer rollout.
- Action impact is mostly immediate at short intervals.

### When to use full RL
Use RL (PPO/SAC or constrained variants) if:
- Delayed effects are material (driver repositioning, churn feedback).
- You can simulate/estimate transitions reasonably.
- You have strong safety infrastructure.

### Practical picks
- **Bandit baseline**: Thompson sampling / LinUCB with action constraints.
- **Discrete RL**: distributional DQN + conservative regularization for logged data.
- **Continuous RL**: SAC only if continuous pricing is truly needed.
- **Offline RL**: CQL/IQL-style conservative approaches when online exploration is costly.

### Why simple often wins in pricing
- Better interpretability for product/legal/ops.
- Lower data requirements.
- Fewer catastrophic failure modes from distribution shift.

---

## 6) Training Setup

### Offline training from logs
- Build logged dataset: `(state_t, action_t, reward_t, state_{t+1}, done, propensities)`.
- Include policy metadata: which rule/model set the historical price.
- De-bias where possible (IPS/DR style methods).

### Counterfactual evaluation
- Off-policy evaluation with:
  - IPS/SNIPS (high variance but useful diagnostics)
  - Doubly robust estimators
  - Fitted Q evaluation (model-based check)
- Sanity check on slices: city, zone type, weather, peak/off-peak.

### Exploration strategy (online)
- Start with **safe exploration envelope**: small traffic %, capped deltas.
- Epsilon or Thompson exploration within allowed action subset.
- Never explore into legally/product disallowed prices.

### Safety rollout
- Shadow mode -> 1% traffic -> staged ramp.
- Real-time guardrails trigger fallback policy.
- Human-on-call escalation for anomaly spikes.

---

## 7) Evaluation Metrics in Production

### Primary KPIs
- Net revenue per available hour.
- Fulfillment/completion rate.
- Median and p90 pickup ETA.

### Guardrails
- Cancellation rate.
- Repeat-user 7-day retention.
- Complaint/refund rate.
- Price volatility index.
- Fairness gap across protected/geographic segments.

### Long vs short horizon
- Daily: fulfillment, ETA, cancellation, net rev.
- Weekly/monthly: retention, cohort LTV proxy, supply stability.

### A/B testing strategy
- Cluster-randomized by zone-time blocks to reduce interference.
- Pre-register metrics and stopping rules.
- Track spillovers between nearby zones.

---

## 8) Weekly Workflow (What your real job feels like)

### Daily checks
- Data freshness, delayed joins, missing feature rates.
- Drift dashboard: demand/supply/elasticity shifts.
- Guardrail breaches by city and hour.

### Common debugging issues
- Feature leakage (future ETAs accidentally used).
- Logging gaps in actions or propensities.
- Latency causing stale state at decision time.
- Reward attribution delays (refund/cancellation events late).

### Failure modes to expect
- Overreaction to transient spikes.
- Zone-specific collapse due to sparse data.
- Policy oscillation when nearby zones interact.
- Revenue up but retention down (classic misaligned objective).

### Cross-functional realities
- Product sets customer trust constraints (caps, communication).
- Ops constraints supply incentives and region exceptions.
- Engineering constraints latency, feature availability, rollout tooling.
- Legal/comms constraints around surge fairness and transparency.

---

## 9) Python-style pseudocode (environment + reward + training loop)

```python
import numpy as np

class MarketplacePricingEnv:
    def __init__(self, config, demand_model, supply_model, cancel_model):
        self.cfg = config
        self.demand_model = demand_model
        self.supply_model = supply_model
        self.cancel_model = cancel_model
        self.reset()

    def reset(self):
        self.t = 0
        self.state = self._init_state()
        return self._observe(self.state)

    def step(self, action_idx):
        surge = self._action_to_surge(action_idx)
        surge = self._apply_constraints(surge, self.state)

        s = self.state

        # 1) demand response
        lam = self.demand_model.base_rate(s) * np.exp(-s["elasticity"] * (surge - 1.0))
        arrivals = np.random.poisson(max(lam, 1e-6))

        # 2) supply response (lagged)
        next_supply = self.supply_model.transition(current_supply=s["active_drivers"],
                                                   surge=surge,
                                                   context=s)

        # 3) matching / fulfillment
        capacity = self._capacity(next_supply, s["traffic_index"], s["eta"])
        requested = arrivals + s["backlog"]
        fulfilled = min(requested, capacity)
        unfulfilled = max(0, requested - fulfilled)

        # 4) cancellations
        cancel_p = self.cancel_model.predict_proba(
            eta=s["eta"], surge=surge, elasticity=s["elasticity"]
        )
        cancels = np.random.binomial(fulfilled, np.clip(cancel_p, 0, 1))
        completed = fulfilled - cancels

        # 5) economics
        gross_fare = completed * s["base_price"] * surge
        incentives = self._driver_incentive_cost(next_supply, surge)
        refunds = self._refund_cost(cancels)
        net_revenue = gross_fare - incentives - refunds

        # 6) reward
        reward = self._reward(
            net_revenue=net_revenue,
            completed=completed,
            cancels=cancels,
            surge=surge,
            prev_surge=s["current_surge"],
            eta=s["eta"],
            utilization=self._utilization(next_supply, fulfilled)
        )

        # 7) transition
        next_state = dict(s)
        next_state["current_surge"] = surge
        next_state["active_drivers"] = next_supply
        next_state["backlog"] = unfulfilled
        next_state["eta"] = self._next_eta(next_supply, requested)
        next_state = self._apply_exogenous_shocks(next_state, self.t)

        self.state = next_state
        self.t += 1
        done = self.t >= self.cfg.max_steps

        info = {
            "net_revenue": net_revenue,
            "completed": completed,
            "cancels": cancels,
            "surge": surge,
        }
        return self._observe(next_state), reward, done, info

    def _reward(self, net_revenue, completed, cancels, surge, prev_surge, eta, utilization):
        w = self.cfg.reward_weights
        price_extremity = max(0.0, surge - self.cfg.comfort_cap) ** 2
        volatility = (surge - prev_surge) ** 2
        supply_health = np.clip(utilization, 0, 1) - 0.1 * max(0, eta - self.cfg.eta_target)

        return (
            w["rev"] * self._scale("rev", net_revenue)
            + w["fulfill"] * self._scale("fulfill", completed)
            + w["supply"] * self._scale("supply", supply_health)
            - w["cancel"] * self._scale("cancel", cancels)
            - w["extreme"] * price_extremity
            - w["vol"] * volatility
        )
```

```python
# Sample conservative training loop (offline -> online gated rollout)

def train_policy_offline(dataset, policy, ope_estimators, safety_rules):
    # 1) fit on logged transitions
    for epoch in range(NUM_EPOCHS):
        batch = dataset.sample_batch(BATCH_SIZE)
        loss = policy.update(batch)

    # 2) off-policy eval
    eval_report = {}
    for est in ope_estimators:
        eval_report[est.name] = est.estimate(policy, dataset)

    # 3) safety checks before shipping
    if not safety_rules.pass_all(eval_report):
        raise RuntimeError("Policy failed offline safety gate")

    return policy, eval_report


def deploy_with_ramp(policy, router, monitor):
    # shadow mode first
    router.enable_shadow(policy)

    # small traffic ramp with guardrails
    for pct in [1, 5, 10, 25, 50]:
        router.set_live_traffic_percent(pct)
        metrics = monitor.wait_and_collect(hours=6)
        if monitor.guardrail_breach(metrics):
            router.rollback_to_baseline()
            return "rolled back"

    return "fully ramped"
```

---

## 10) Production pitfalls and non-academic compromises

1. **You will optimize what you can measure, not necessarily what matters.**
   Invest early in retention/churn proxies and delayed reward joins.

2. **Causal bias from logged policies is unavoidable.**
   Treat OPE as directional, not truth.

3. **Interference across zones breaks clean assumptions.**
   Use geographically aware experimentation.

4. **Latency and data quality beat model complexity.**
   A simpler model with reliable features usually wins.

5. **Interpretability is a feature.**
   You need to explain bad outcomes to ops/product in minutes, not days.

6. **Hard safety constraints are mandatory.**
   Reward penalties alone are insufficient in production.

---

## Strong default recommendation for your first 90 days

- Start with **constrained contextual bandit** on discrete multipliers.
- Build robust feature pipelines + guardrail monitoring.
- Add delayed retention proxy into evaluation dashboards.
- Move to full RL only after proving stable causal lift and safe operations.

If you can consistently run reliable experiments and avoid trust-damaging pricing spikes, you're doing the right thing.
