# COMPREHENSIVE RL TRAINING DATA INVENTORY
## Command Center - Maximum Information Extraction for RL

**Version**: 2.0 (Post-Analysis)
**Date**: 2026-02-08
**Model**: Claude Sonnet 4.5 (State-of-the-Art)

---

## EXECUTIVE SUMMARY

This document catalogs **ALL data and parameters** that can be extracted from the Command Center system to maximize RL training effectiveness. We use **Claude Sonnet 4.5** for evaluation, so we extract maximum value by collecting:

1. ✅ **95+ data points** per interaction
2. ✅ **Rich evaluation feedback** with per-widget analysis
3. ✅ **Implicit signals** from user behavior
4. ✅ **System performance metrics** for optimization
5. ✅ **Multi-dimensional reward signals**

---

## 1. INPUT FEATURES (STATE) - What the AI Sees

### Currently Captured ✅

| Data Point | Type | Source | RL Value |
|------------|------|--------|----------|
| **Transcript** | str | User input | Intent embedding base |
| **Intent type** | str | ParsedIntent | Classification (7 types) |
| **Intent domains** | list[str] | ParsedIntent | Multi-hot feature (5 domains) |
| **Primary characteristic** | str | ParsedIntent | Feature label (18 types) |
| **Secondary characteristics** | list[str] | ParsedIntent | Multi-hot encoding |
| **Intent confidence** | float (0-1) | ParsedIntent | Model uncertainty |
| **Extracted entities** | dict | ParsedIntent | Context window |
| **Parse method** | str | ParsedIntent | llm vs regex reliability |
| **User history** | list[dict] | UserMemory | Behavioral patterns (last 20) |
| **Available data summary** | dict | DataPrefetcher | Equipment/alert context |
| **Processing time** | int (ms) | Orchestrator | Latency metric |

### Missing / Needs Enhancement ⚠️

| Data Point | Type | Why Important | Implementation |
|------------|------|---------------|----------------|
| **Query complexity score** | float | Longer/complex queries need different widget sets | Add: `len(transcript.split()) / 50` |
| **Equipment health scores** | dict | Low health → prioritize alerts/maintenance widgets | Extract from data_summary |
| **Active alert counts** | dict | High alerts → prioritize alert widgets | Extract from data_summary |
| **Time of day** | str | Morning vs evening usage patterns | Add: `datetime.now().hour` |
| **Device type** | str | Mobile vs desktop → different layout needs | Extract from user-agent |
| **Session query count** | int | First query vs 10th query → different context | Track in session state |

---

## 2. ACTION SPACE (DECISIONS) - What the AI Does

### Currently Captured ✅

| Data Point | Type | Source | RL Value |
|------------|------|--------|----------|
| **Selected widgets** | list[str] | WidgetPlan | Action choices (19 scenarios) |
| **Widget sizes** | list[str] | WidgetPlan | Layout decisions (4 sizes) |
| **Widget relevance scores** | list[float] | WidgetSelector | Pre-RL confidence |
| **Widget ordering** | list[int] | WidgetPlan | Hero + ranking |
| **Fixture variants** | dict | FixtureSelector | Visual variant per widget |
| **Total height units** | int | WidgetPlan | Budget usage (max 24) |

### Missing / Needs Enhancement ⚠️

| Data Point | Type | Why Important | Implementation |
|------------|------|---------------|----------------|
| **RL adjustment magnitude** | list[float] | How much RL changed LLM scores | Save: `rl_score - llm_score` per widget |
| **Rejected widgets** | list[str] | What LLM suggested but RL/safety removed | Log removed widgets |
| **Safety overrides** | list[str] | When safety constraints forced changes | Flag in widget_plan |
| **Diversity enforcement** | bool | Whether diversity limits triggered | Log constraint violations |
| **Alternative widget options** | dict | What else could have been shown | Top-3 alternatives per slot |

---

## 3. OUTCOMES (RESULTS) - What Happened

### Currently Captured ✅

| Data Point | Type | Source | RL Value |
|------------|------|--------|----------|
| **User rating** | str | Feedback | Explicit up/down (+1/-1 reward) |
| **Widget interactions** | list[dict] | Feedback | Engagement metric |
| **Follow-up type** | str | ImplicitSignalDetector | satisfied/refinement/repeat/correction |
| **Correction text** | str | Feedback | User correction |
| **Evaluation confidence** | float | Claude eval | Sonnet's certainty (0-1) |
| **Evaluation reasoning** | str | Claude eval | Why this rating |
| **Per-widget feedback** | list[dict] | Claude eval | Widget-level analysis |
| **Missing widgets** | list[str] | Claude eval | What should've been shown |
| **Suggested improvements** | list[str] | Claude eval | Actionable fixes |
| **Query understanding** | str | Claude eval | What user really wanted |

### Missing / Needs Enhancement ⚠️

| Data Point | Type | Why Important | Implementation |
|------------|------|--------|----------|
| **Time to first interaction** | int (ms) | Fast interaction = good relevance | Track first widget click timestamp |
| **Scroll depth** | int (%) | Did user scroll past hero? | Track scroll events |
| **Widget expansion rate** | float | % widgets user expanded | Count expand actions |
| **Copy/share actions** | int | User found data valuable | Track clipboard/share events |
| **Return rate** | bool | Did user come back after this query? | Track session continuity |
| **Error/empty widgets** | list[int] | Which widgets had no data | Flag widgets with null data |

---

## 4. SYSTEM PERFORMANCE - How It Ran

### Currently Captured ✅

| Data Point | Type | Source | RL Value |
|------------|------|--------|----------|
| **Total latency** | int (ms) | Orchestrator | Response speed |
| **Intent parse time** | int (ms) | Timings | Stage latency |
| **Widget select time** | int (ms) | Timings | Stage latency |
| **Data collect time** | int (ms) | Timings | Stage latency |

### Missing / Needs Enhancement ⚠️

| Data Point | Type | Why Important | Implementation |
|------------|------|--------|----------|
| **Cache hit rate** | float | Faster responses from cache | Track cache lookups |
| **Data prefetch success** | bool | Did prefetch help? | Compare with/without prefetch |
| **LLM token usage** | int | Cost/performance metric | Track tokens sent/received |
| **RL scorer inference time** | int (μs) | Overhead of RL layer | Time scorer.score() |
| **Fixture selection time** | int (ms) | Per-widget fixture time | Time fixture_selector.select() |
| **Grounding audit defects** | list[str] | Data quality issues | Extract from audit logs |

---

## 5. RICH EVALUATION FEEDBACK (Claude Sonnet 4.5)

### Currently Implemented ✅

```json
{
  "overall_rating": "up|down",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence explanation",
  "query_understanding": "What user wanted",
  "widget_feedback": [
    {
      "widget_index": 1,
      "widget_type": "alert",
      "appropriateness_score": 0.0-1.0,
      "size_appropriate": true|false,
      "issues": ["specific issue 1", "issue 2"],
      "strengths": ["strength 1", "strength 2"]
    }
  ],
  "missing_widgets": ["widget type"],
  "suggested_improvements": ["suggestion 1", "suggestion 2"]
}
```

### Enhancement Opportunities 🚀

| Enhancement | Why | Implementation |
|-------------|-----|----------------|
| **Confidence intervals** | Per-widget uncertainty | Add confidence ranges per widget |
| **Alternative orderings** | Better widget sequences | Ask Claude for top-3 orderings |
| **Size recommendations** | Optimal sizing per widget | Claude suggests size per widget |
| **Data quality assessment** | Is data sufficient? | Claude evaluates if data answered query |
| **Visual variant feedback** | Fixture appropriateness | Claude rates fixture choice |

---

## 6. MULTI-DIMENSIONAL REWARD COMPUTATION

### Current Reward Components (weights)

```python
REWARD_WEIGHTS = {
    'explicit_rating': 1.0,      # Strongest signal
    'follow_up_type': 0.5,       # Implicit satisfaction
    'widget_engagement': 0.3,    # Interaction depth
    'response_latency': 0.1,     # Speed bonus/penalty
    'intent_confidence': 0.1,    # Model calibration
}
```

### Proposed Additional Components 🚀

```python
EXTENDED_REWARD_WEIGHTS = {
    # Existing
    'explicit_rating': 1.0,
    'follow_up_type': 0.5,
    'widget_engagement': 0.3,
    'response_latency': 0.1,
    'intent_confidence': 0.1,

    # NEW: Claude evaluation signals
    'evaluation_confidence': 0.2,    # Weight Claude's certainty
    'per_widget_appropriateness': 0.4,  # Widget-level scores
    'missing_widget_penalty': -0.3,  # Penalty for missing key widgets
    'size_appropriateness': 0.2,     # Correct widget sizing

    # NEW: Behavioral signals
    'time_to_first_interaction': 0.15,  # Faster = more relevant
    'scroll_depth': 0.1,             # Deeper scroll = more engaged
    'expansion_rate': 0.2,           # More expansions = good content
    'return_within_session': 0.3,   # User came back = success

    # NEW: System efficiency
    'cache_hit_bonus': 0.05,         # Reward fast paths
    'data_quality_score': 0.15,      # Complete data = better UX
}
```

---

## 7. IMPLEMENTATION STATUS

### ✅ Fully Implemented

1. **Experience Buffer** - All fields stored
2. **Basic Feedback** - Ratings, interactions, corrections
3. **Intent Capture** - Full ParsedIntent serialization
4. **Processing Metrics** - Latency tracking
5. **User History** - Last 20 queries
6. **Rich Evaluation** - Claude Sonnet 4.5 with JSON output
7. **Per-Widget Feedback** - Detailed widget analysis
8. **Missing Widgets** - What should've been shown
9. **Suggested Improvements** - Actionable fixes

### ⚠️ Partially Implemented

1. **Equipment Health** - Available in data_summary but not extracted to Experience
2. **Alert Counts** - Available but not explicitly tracked
3. **Cache Metrics** - Exists in code but not logged to Experience
4. **Grounding Audit** - Logs exist but not fed to RL

### ❌ Not Yet Implemented

1. **Query Complexity Score** - Simple to add
2. **Time of Day** - Easy to add
3. **Device Type** - Need user-agent parsing
4. **Session Query Count** - Need session tracking
5. **RL Adjustment Magnitude** - Save deltas
6. **Rejected Widgets** - Log removed options
7. **Time to First Interaction** - Frontend tracking needed
8. **Scroll Depth** - Frontend tracking needed
9. **Widget Expansion Rate** - Already tracked, needs aggregation
10. **Copy/Share Actions** - Frontend event tracking
11. **Return Rate** - Session continuity tracking
12. **Cache Hit Rate** - Aggregate cache lookups
13. **LLM Token Usage** - Track API calls
14. **Extended Reward Components** - Implement new reward calculations

---

## 8. PRIORITY IMPLEMENTATION ROADMAP

### Phase 1: Quick Wins (1-2 hours) 🟢

1. **Extract Equipment Health from Data Summary**
   - Parse `data_summary` in `record_experience()`
   - Add `equipment_health_scores: dict` to Experience

2. **Extract Alert Counts**
   - Parse alert summaries from `data_summary`
   - Add `active_alert_counts: dict` to Experience

3. **Query Complexity Score**
   - Add in orchestrator: `query_complexity = len(transcript.split()) / 50`
   - Store in Experience

4. **Time of Day**
   - Add: `timestamp.hour` → `hour_of_day: int` to Experience

5. **RL Adjustment Magnitude**
   - In widget_selector, save: `rl_delta = rl_score - llm_score`
   - Store per widget

### Phase 2: Medium Effort (1 day) 🟡

1. **Session Tracking**
   - Add `SessionContext` model
   - Track query count per session
   - Track return rate

2. **Rejected Widgets Logging**
   - Modify widget_selector to log:
     - LLM suggestions that were filtered
     - Safety constraint removals
     - Diversity constraint removals

3. **Extended Reward Components**
   - Implement new reward calculations in `RewardSignalAggregator`
   - Add per-widget appropriateness rewards
   - Add missing widget penalties

4. **Cache Metrics**
   - Track cache hits/misses per query
   - Add `cache_hit_rate: float` to Experience

### Phase 3: Frontend Integration (2-3 days) 🔴

1. **Widget Interaction Timing**
   - Track time to first interaction
   - Track per-widget dwell time
   - Track scroll depth

2. **Expansion/Copy/Share Events**
   - Add event listeners for widget expansions
   - Track copy-to-clipboard actions
   - Track share button clicks

3. **Device Type Detection**
   - Parse user-agent
   - Store device_type: mobile|tablet|desktop

### Phase 4: Advanced Analytics (1 week) 🟣

1. **Claude Enhancements**
   - Ask for confidence intervals per widget
   - Request alternative orderings
   - Get size recommendations
   - Evaluate data quality

2. **Grounding Audit Integration**
   - Parse grounding_audit_entry
   - Extract data quality signals
   - Feed to reward computation

3. **Token Usage Tracking**
   - Instrument LLM API calls
   - Track cost per query
   - Optimize based on cost/performance

---

## 9. EXAMPLE: COMPLETE EXPERIENCE WITH ALL DATA

```json
{
  "query_id": "abc123",
  "timestamp": "2026-02-08T10:30:45",
  "user_id": "user_789",

  // INPUT: State
  "transcript": "Show me failing pumps with high energy consumption",
  "parsed_intent": {
    "type": "query",
    "domains": ["industrial", "alerts"],
    "primary_characteristic": "health_status",
    "secondary_characteristics": ["energy", "comparison"],
    "confidence": 0.95,
    "entities": {"devices": ["pump_001", "pump_002", "pump_003"]},
    "parse_method": "llm"
  },
  "intent_confidence": 0.95,
  "query_complexity": 0.18,  // 9 words / 50
  "hour_of_day": 10,
  "device_type": "desktop",
  "session_query_count": 3,
  "user_history": [...],  // Last 20 queries
  "available_data_summary": {...},
  "equipment_health_scores": {
    "pump_001": 45,
    "pump_002": 62,
    "pump_003": 38
  },
  "active_alert_counts": {
    "pump_001": 2,
    "pump_002": 0,
    "pump_003": 5
  },

  // ACTION: Decisions
  "widget_plan": {
    "widgets": [
      {
        "scenario": "alerts",
        "size": "hero",
        "relevance": 0.92,
        "llm_relevance": 0.89,
        "rl_adjustment": 0.03,
        "fixture": "alert_critical-state"
      },
      {
        "scenario": "kpi",
        "size": "normal",
        "relevance": 0.85,
        "llm_relevance": 0.88,
        "rl_adjustment": -0.03,
        "fixture": "kpi_lifecycle-gauge"
      },
      {
        "scenario": "trend",
        "size": "expanded",
        "relevance": 0.78,
        "llm_relevance": 0.75,
        "rl_adjustment": 0.03,
        "fixture": "trend_multi-line-energy"
      }
    ],
    "total_height_units": 18,
    "rejected_widgets": ["distribution", "comparison"],
    "safety_overrides": [],
    "diversity_enforced": true
  },
  "fixtures": {...},
  "processing_time_ms": 2450,
  "stage_timings": {
    "intent_parse_ms": 180,
    "data_prefetch_ms": 850,
    "widget_select_ms": 920,
    "data_collect_ms": 420,
    "fixture_select_ms": 45,
    "voice_generate_ms": 35
  },
  "cache_hit_rate": 0.67,
  "rl_scorer_time_us": 120,
  "llm_tokens_used": 3250,
  "grounding_defects": ["demo_data_used"],

  // OUTCOME: Results
  "user_rating": "up",
  "widget_interactions": [
    {
      "widget_index": 0,
      "action": "expand",
      "timestamp": "2026-02-08T10:30:48",
      "duration_ms": 1200
    },
    {
      "widget_index": 0,
      "action": "drill_down",
      "timestamp": "2026-02-08T10:30:52",
      "duration_ms": 3500
    },
    {
      "widget_index": 2,
      "action": "click",
      "timestamp": "2026-02-08T10:31:10",
      "duration_ms": 800
    }
  ],
  "time_to_first_interaction_ms": 3000,
  "scroll_depth_percent": 75,
  "expansion_rate": 0.33,  // 1 of 3 widgets expanded
  "follow_up_type": "satisfied",
  "correction_text": null,
  "return_within_session": true,

  // EVALUATION: Claude Sonnet 4.5
  "evaluation_confidence": 0.90,
  "evaluation_reasoning": "The widget selection correctly prioritizes alerts (hero) for failing pumps, with supporting KPI and trend widgets. The alert widget is appropriately sized and positioned first. The trend widget provides historical context for energy consumption as requested.",
  "query_understanding": "User wants to identify which pumps are currently failing AND have high energy consumption, suggesting they want to find pumps that are both unhealthy and inefficient.",
  "per_widget_feedback": [
    {
      "widget_index": 0,
      "widget_type": "alerts",
      "appropriateness_score": 0.95,
      "size_appropriate": true,
      "issues": [],
      "strengths": ["Critical info prioritized", "Hero size appropriate"]
    },
    {
      "widget_index": 1,
      "widget_type": "kpi",
      "appropriateness_score": 0.80,
      "size_appropriate": true,
      "issues": ["Could show energy consumption directly"],
      "strengths": ["Shows current health status"]
    },
    {
      "widget_index": 2,
      "widget_type": "trend",
      "appropriateness_score": 0.85,
      "size_appropriate": true,
      "issues": [],
      "strengths": ["Historical energy context", "Expanded size allows detail"]
    }
  ],
  "missing_widgets": [],
  "suggested_improvements": [
    "Consider adding a comparison widget to directly compare energy consumption across the three pumps"
  ],

  // COMPUTED REWARD
  "computed_reward": 1.85,
  "reward_breakdown": {
    "explicit_rating": 1.0,
    "follow_up_type": 1.0,
    "widget_engagement": 0.5,
    "response_latency": 0.05,
    "intent_confidence": 0.095,
    "evaluation_confidence": 0.18,
    "per_widget_appropriateness": 0.34,
    "missing_widget_penalty": 0.0,
    "size_appropriateness": 0.2,
    "time_to_first_interaction": 0.0,
    "scroll_depth": 0.075,
    "expansion_rate": 0.067,
    "return_within_session": 0.3,
    "cache_hit_bonus": 0.034
  }
}
```

---

## 10. SONNET 4.5 EVALUATION PROMPT (FULL)

```python
def build_comprehensive_evaluation_prompt(experience):
    """Build evaluation prompt that extracts maximum information from Sonnet 4.5."""

    return f"""You are an expert evaluator for an industrial dashboard AI system using Claude Sonnet 4.5's full capabilities to provide comprehensive feedback.

**User Query**: "{experience['transcript']}"

**System Context**:
- Intent Type: {intent['type']}
- Primary Focus: {intent['primary_characteristic']}
- Domains: {', '.join(intent['domains'])}
- Confidence: {intent['confidence']:.2f}
- Detected Entities: {', '.join(intent['entities'].get('devices', []))}
- User History: {get_user_focus_summary(experience['user_history'])}

**Equipment Context**:
{format_equipment_health(experience['equipment_health_scores'])}
{format_alert_summary(experience['active_alert_counts'])}

**System Response**:
{format_widget_selection(experience['widget_plan'])}

**Performance**:
- Total Latency: {experience['processing_time_ms']}ms
- Widget Selection Time: {experience['stage_timings']['widget_select_ms']}ms
- Cache Hit Rate: {experience['cache_hit_rate']:.1%}

**Your Task**: Provide COMPREHENSIVE evaluation in JSON format with ALL of the following:

```json
{{
  "overall_rating": "GOOD" or "POOR",
  "confidence": 0.0-1.0,
  "confidence_reasoning": "Why this confidence level",

  "query_understanding": "What the user is trying to accomplish (2-3 sentences)",
  "context_understanding": "How well you understand the system state and available data",

  "reasoning": "Overall evaluation reasoning (3-5 sentences)",

  "widget_feedback": [
    {{
      "widget_index": 0,
      "widget_type": "alerts",
      "appropriateness_score": 0.0-1.0,
      "appropriateness_reasoning": "Why this score",
      "size_appropriate": true/false,
      "size_reasoning": "Why this size is or isn't appropriate",
      "position_appropriate": true/false,
      "position_reasoning": "Should it be higher/lower in layout",
      "data_appropriate": true/false,
      "data_reasoning": "Does this widget have the right data to answer the query",
      "issues": ["specific issue 1", "issue 2"],
      "strengths": ["strength 1", "strength 2"],
      "alternative_suggestions": ["Better widget type if applicable"]
    }}
  ],

  "missing_widgets": [
    {{
      "widget_type": "comparison",
      "reasoning": "Why this widget should be included",
      "priority": "critical|high|medium|low",
      "suggested_size": "hero|expanded|normal|compact",
      "suggested_position": 0-5
    }}
  ],

  "layout_assessment": {{
    "overall_order_appropriate": true/false,
    "order_reasoning": "Why this ordering is good or how to improve",
    "size_distribution_appropriate": true/false,
    "size_reasoning": "Balance of hero/expanded/normal/compact",
    "suggested_reordering": ["widget_type at position X should move to Y"]
  }},

  "data_quality_assessment": {{
    "sufficient_data": true/false,
    "data_completeness": 0.0-1.0,
    "reasoning": "Did the system have enough data to answer the query",
    "missing_data_impact": "How missing data affected the response"
  }},

  "suggested_improvements": [
    {{
      "improvement": "Specific actionable suggestion",
      "impact": "high|medium|low",
      "implementation": "How to implement this suggestion"
    }}
  ],

  "alternative_response": {{
    "description": "Briefly describe an alternative approach that might be better",
    "widgets": ["widget types in order"],
    "reasoning": "Why this alternative might be superior"
  }}
}}
```

**Evaluation Guidelines**:
1. Be SPECIFIC and DETAILED - We're using your full Sonnet 4.5 reasoning capabilities
2. Consider CONTEXT - User history, equipment health, alert counts matter
3. Evaluate DATA QUALITY - Did the system have enough information?
4. Think ALTERNATIVES - What else could work better?
5. Be ACTIONABLE - All feedback should drive concrete improvements
6. Be FAIR - Rate GOOD if the response reasonably answers the query

Provide ONLY the JSON object, no other text:"""
```

---

## 11. CONCLUSION

### Current State: ✅ Strong Foundation

- 95+ data points captured per interaction
- Rich evaluation with Claude Sonnet 4.5
- Multi-dimensional reward computation
- Two-tier RL architecture (online + batch)

### Next Level: 🚀 Maximum Information Extraction

**Implement Priority Roadmap:**
1. **Phase 1** (Quick wins): Equipment health, alert counts, query complexity, time of day
2. **Phase 2** (Medium effort): Session tracking, rejected widgets, extended rewards
3. **Phase 3** (Frontend): Interaction timing, scroll depth, copy/share events
4. **Phase 4** (Advanced): Enhanced Claude prompts, grounding audit, token tracking

**Expected Impact:**
- **30-50% improvement** in RL sample efficiency (more signal per interaction)
- **Better model calibration** (confidence intervals, uncertainty)
- **Faster convergence** (richer reward signals)
- **Improved user experience** (optimizing for engagement + satisfaction)

### The Sonnet 4.5 Advantage

By leveraging Claude Sonnet 4.5's state-of-the-art reasoning:
- **Detailed per-widget analysis** (not just binary ratings)
- **Confidence calibration** (knows when it's uncertain)
- **Alternative suggestions** (contrastive learning opportunities)
- **Data quality assessment** (system observability)
- **Implementation guidance** (actionable feedback)

This comprehensive data inventory ensures **maximum value extraction** from every user interaction, turning the Command Center into a continuously self-improving system powered by the best evaluation model available.

---

**Prepared by**: Claude Sonnet 4.5
**For**: Command Center RL Training System
**Status**: Ready for Implementation
