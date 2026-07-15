# 🏔️ Technical Reference: Alpin Architect

This document contains the complete **mathematical foundation** and **algorithm methodology** for ski resort planning. It serves as a technical backup documenting how paths and lifts are generated.

For user workflow, see [DETAILS_UI.md](DETAILS_UI.md).

---

## Table of Contents

1. [Core Helper Functions](#1-core-helper-functions)
2. [Terrain Gradient Calculation](#2-terrain-gradient-calculation)
3. [Traverse Physics](#3-traverse-physics-the-core-relationship)
4. [Civil Engineering: Earthwork](#4-civil-engineering-earthwork--excavation)
5. [Path Generation Algorithm](#5-path-generation-algorithm)
6. [Difficulty Classification](#6-difficulty-classification)
7. [Custom Direction / Connect Paths](#7-custom-direction--connect-paths)
8. [Lift Pylon Placement](#8-lift-pylon-placement)
9. [OpenStreetMap Import](#9-openstreetmap-import)

---

## 1. Core Helper Functions

These functions read from the DEM (Digital Elevation Model) and are used throughout the path planning algorithm.

### 1.1 `get_elevation(lon, lat)` → float

Returns the elevation in meters at the given coordinates, or `None` if:
- Position is outside DEM bounds
- Position falls in a no-data area

**Implementation:** Direct single-cell lookup from the DEM array — no interpolation or multi-point sampling. The coordinate is transformed to the DEM's native CRS, converted to array indices, and the value at that cell is returned.

### 1.2 `get_terrain_gradient(lon, lat)` → $(S_{\text{terrain}}, \theta_{\text{fall}})$

Returns:
- $S_{\text{terrain}}$: Terrain steepness as percentage
- $\theta_{\text{fall}}$: Fall line bearing (direction of steepest descent, 0°=North)

**Implementation:** Uses **weighted multi-point sampling** (16 elevation lookups around the center point) to reduce DEM noise. See Section 2 below.

---

## 2. Terrain Gradient Calculation

### 2.1 The Problem: DEM Grid Noise

A 60m Digital Elevation Model (DEM) has inherent noise. A single elevation difference can give misleading slope values. We use **weighted multi-point sampling** to get robust terrain measurements.

> **Note:** This approach is inspired by the ArcGIS Surface Parameters tool, which recommends larger neighborhoods over the traditional 3×3 Horn algorithm for noisy terrain.

### 2.2 Weighted Gradient Calculation ("Magic 8")

We sample elevations $z_i$ at 8 compass bearings on two concentric rings:
- **Inner Ring ($r_1 = 0.5 \times$ `STEP_SIZE_M` $\approx 15\text{ m}$):** Weight $w_{\text{inner}} = 2$
- **Outer Ring ($r_2 = 1.0 \times$ `STEP_SIZE_M` $\approx 30\text{ m}$):** Weight $w_{\text{outer}} = 1$

For each sample point $i$ at bearing $\phi_i$, calculate the slope ratio (rise/run) from center:
$$slope_i = \frac{z_{\text{center}} - z_i}{d_i}$$

This is a dimensionless ratio (positive = downhill from center, negative = uphill).

Decompose into East-West and North-South gradient components:
$$\frac{\partial z}{\partial x} \approx \frac{1}{\sum w} \sum_{i=1}^{n} (slope_i \cdot \sin(\phi_i) \cdot w_i)$$
$$\frac{\partial z}{\partial y} \approx \frac{1}{\sum w} \sum_{i=1}^{n} (slope_i \cdot \cos(\phi_i) \cdot w_i)$$

### 2.3 Output Values

The **gradient magnitude** gives the slope ratio:
$$r = \sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}$$

**Terrain Steepness** as percentage (consistent with ArcGIS):
$$S_{\text{terrain}} = r \times 100$$

**Fall Line Bearing** (direction of steepest descent, 0°=North):
$$\theta_{\text{fall}} = \text{atan2}\left(\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}\right)$$

> **Source:** Based on principles from **Zevenbergen & Thorne (1987)** and **Horn (1981)**. See [ArcGIS: How Slope Works](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-slope-works.htm)

---

## 3. Traverse Physics: The Core Relationship

### 3.1 The Three Slopes

When a skier traverses across a slope, three distinct gradients are involved:

| Slope | Symbol | Description |
|-------|--------|-------------|
| **Terrain Slope** | $S_{\text{terrain}}$ | Natural mountain steepness (measured from DEM) |
| **Effective Slope** | $S_{\text{eff}}$ | What the skier actually experiences (design target) |
| **Side Slope** | $S_{\text{side}}$ | Cross-slope perpendicular to ski direction |

### 3.2 The Fundamental Equations

Given a **traverse angle** $\theta$ (offset from fall line):

**Effective Slope** (component in ski direction):
$$S_{\text{eff}} = S_{\text{terrain}} \cdot \cos(\theta)$$

**Side Slope** (component perpendicular to ski direction):
$$S_{\text{side}} = S_{\text{terrain}} \cdot \sin(\theta)$$

These satisfy the Pythagorean identity:
$$S_{\text{eff}}^2 + S_{\text{side}}^2 = S_{\text{terrain}}^2$$

**Intuition:**
- $\theta = 0°$: Skiing straight down the fall line → $S_{\text{eff}} = S_{\text{terrain}}$, $S_{\text{side}} = 0$
- $\theta = 90°$: Skiing perpendicular (contouring) → $S_{\text{eff}} = 0$, $S_{\text{side}} = S_{\text{terrain}}$
- $\theta > 90°$: Tilting back **against** the fall line → $S_{\text{eff}} < 0$ (a climb)

### 3.3 Calculating Traverse Angle from Target

The designer sets a **signed target effective slope** ($S_{\text{target}}$) for each path. The algorithm calculates the required traverse angle from the **signed** ratio, so one formula spans descend/contour/climb:

$$\theta = \arccos\left(\frac{S_{\text{target}}}{S_{\text{terrain}}}\right)$$

Because $\arccos$ ranges over $[0°, 180°]$: a positive target gives $\theta < 90°$ (tilt toward the fall line, descend), zero gives $\theta = 90°$ (contour), and a negative target gives $\theta > 90°$ (tilt against the fall line, climb). The reference bearing is **always** the fall line — direction is carried entirely by $\theta$.

### 3.4 Difficulty Thresholds

| Difficulty | Effective Slope Range |
|------------|----------------------|
| 🟢 Green | 0% – 15% |
| 🔵 Blue | 15% – 25% |
| 🔴 Red | 25% – 40% |
| ⚫ Black | 40%+ |

### 3.5 Example: Blue Run on Black Terrain

- Terrain: $S_{\text{terrain}} = 50\%$ (Black-rated natural slope)
- Target: $S_{\text{target}} = 22\%$ (Blue difficulty)

Calculate traverse angle:
$$\theta = \arccos\left(\frac{22}{50}\right) = \arccos(0.44) \approx 64°$$

The resulting side slope:
$$S_{\text{side}} = 50 \cdot \sin(64°) \approx 45\%$$

The skier experiences a comfortable 22% slope, but the cross-slope is 45% — requiring significant earthwork.

---

## 4. Civil Engineering: Earthwork & Excavation

### 4.1 Side Slope Creates a Cross-Section Problem

When traversing, the natural terrain slopes perpendicular to the ski direction. To create a level piste:
- **Excavate** (cut) on the uphill/inner side
- **Fill** on the downhill/outer side

```
                    CROSS-SECTION VIEW

        Original Terrain Surface
              ╲
               ╲    ← Inner edge (excavate below terrain)
                ╲
             I───●───I     ← Planned Centerline (on terrain)
                  ╲
                   ╲  ← Outer edge (fill above terrain)
                    ╲
```

### 4.2 The Belt Model

A ski piste has a physical width called the **Belt**. The path is planned along the **centerline** of this belt.

Belt width is calculated adaptively from side slope to keep excavation within limits:

$$W = \frac{H_{\text{threshold}} \cdot 200}{S_{\text{side}}}$$

Where $H_{\text{threshold}} = 2.5\text{m}$ is the maximum acceptable excavation depth.

For steeper side slopes, the belt narrows to reduce excavation. For gentler side slopes, a wider belt can be used.

### 4.3 Vertical Cut/Fill Depth

**Vertical displacement** at edge (cut or fill depth):
$$H_{\text{edge}} = \frac{S_{\text{side}}}{100} \cdot \frac{W}{2} = \frac{S_{\text{side}} \cdot W}{200}$$

- **Inner edge:** Excavated $H_{\text{edge}}$ meters below original terrain
- **Outer edge:** Filled $H_{\text{edge}}$ meters above original terrain

### 4.4 🚜 Side Cut Warning

A warning is triggered when the side slope exceeds what even the minimum belt width can handle:

$$S_{\text{side}} > \frac{H_{\text{threshold}} \cdot 200}{W_{\text{min}}} = \frac{2.5 \cdot 200}{10} = 50\%$$

When side slope exceeds 50%, the excavation would exceed 2.5m even at minimum belt width.

### 4.5 📐 Too Flat Warning

When terrain is gentler than the minimum skiable slope:

$$S_{\text{avg}} < 5\% \implies \text{Too Flat Warning}$$

### 4.6 Key Insight: Any Terrain Can Be Skied

**Side slope can always be excavated away.** There is no terrain too steep for any difficulty level — the excavator simply does more work. However:
- High side slope = massive cross-slope earthwork
- The warnings alert designers to reconsider the route

---

## 5. Path Generation Algorithm

### 5.1 Core Principle

The tracer holds a **signed target grade** and follows the terrain:
1. **Fixes the target grade** (what the traveller experiences) — its **sign** sets the direction along the path's length: positive **descends** the fall line, negative **climbs** against it, zero **contours** across it.
2. **Dynamically calculates the traverse angle** at each step from the **signed** grade ratio (§3.3), so a single $\arccos$ over $[0°, 180°]$ covers descending, contouring, and climbing off one fixed reference (the fall line).

This allows routes to **naturally curve around terrain features**. Ski **slopes** always descend, so they use positive targets (§5.2). **Roads** may descend, climb, or run flat, so they use the signed green targets (§7.3). A contour (zero target) drives the traverse angle to ~90°, tracing across the slope at near-constant elevation.

### 5.2 Target Effective Slopes

| Difficulty | Gentle Target | Steep Target | Threshold Range |
|------------|---------------|--------------|-----------------|
| 🟢 Green | 7% | 12% | 0–15% |
| 🔵 Blue | 17% | 22% | 15–25% |
| 🔴 Red | 28% | 37% | 25–40% |
| ⚫ Black | 45% | 60% | 40%+ |

> Targets are set 2-3% inside threshold bounds to prevent misclassification.

### 5.3 Path Variants Per Difficulty

| Variant | Side | When Generated |
|---------|------|----------------|
| Left-Gentle/Steep | Left of fall line | When gentle/steep target < terrain slope |
| Right-Gentle/Steep | Right of fall line | When gentle/steep target < terrain slope |
| Center-Gentle/Steep | Straight down | When gentle/steep target ≥ terrain slope |

### 5.4 Center-Stop Rule

Generate paths in order from easiest to hardest target, stopping after **4 center paths** are created:

1. 🟢 Gentle Green (7%)
2. 🟢 Steep Green (12%)
3. 🔵 Gentle Blue (17%)
4. 🔵 Steep Blue (22%)
5. 🔴 Gentle Red (28%)
6. 🔴 Steep Red (37%)
7. ⚫ Gentle Black (45%)
8. ⚫ Steep Black (60%)

**Why?** Center paths all follow the same fall line, so additional ones are redundant.

**Note:** When designing a fan of paths from a node, the generation always loops through all 8 variants from flat to steep, and during each steepness target level it will generate either left and right or only a center path depending on terrain slope.

### 5.5 Cumulative Drop Tracking (Feedback Loop)

**Problem:** DEM grid resolution (60m) causes mismatch between gradient predictions and actual elevation changes.

**Solution:** Track cumulative elevation drop and dynamically adjust each step's target.

**Pre-calculate at initialization:**
$$\text{targetTotalDrop} = \frac{S_{\text{target}}}{100} \times L_{\text{target}}$$

**At each step:**
1. $\text{remainingDrop} = \text{targetTotalDrop} - \text{accumulatedDrop}$ (signed)
2. $\text{remainingDistance} = L_{\text{target}} - d_{\text{total}}$
3. $S_{\text{step}} = \frac{\text{remainingDrop}}{\text{remainingDistance}} \times 100$

$S_{\text{step}}$ carries the **sign** of the target. It is clamped to a band that keeps the step running the target's way — a descent step never climbs, a climb step never descends, a contour stays near level: a descent to $[\,0, S_{\text{target}}\cdot$`CLAMP_FACTOR`$\,]$, a climb to $[\,S_{\text{target}}\cdot$`CLAMP_FACTOR`$, 0\,]$, a contour to $[-$`MIN_SKIABLE`$, +$`MIN_SKIABLE`$]$. The band is floored at **0** (not `MIN_SKIABLE`), so a run that has drifted too steep can ask for a gentle — even flat — step and pull its average back toward the target.

**Why this works:** The path self-corrects toward the target average without retries. Flooring at 0 (rather than `MIN_SKIABLE`) is what lets an over-steep run recover; a `MIN_SKIABLE` floor would trap every step at ≥ 5% and ratchet the average upward.

### 5.6 Step-by-Step Tracing

**Step 1: Sample Local Terrain**

Sample the gradient at the **current point** (the step's start); the cumulative-drop feedback (§5.5) corrects any per-step lag, so no midpoint sampling is needed:
$$S_{\text{terrain}}, \theta_{\text{fall}} = \textrm{getTerrainGradient}(\text{current point})$$

**Step 2: Calculate Traverse Angle** (from the **signed** grade ratio)

$$\theta_{\text{traverse}} = \arccos\left(\frac{S_{\text{step}}}{S_{\text{terrain}}}\right)$$

Cases (with $S_{\text{step}}$ **signed**, §3.3):
- $S_{\text{step}} \geq S_{\text{terrain}}$: $\theta = 0°$ (straight down the fall line — a descent at/above terrain steepness)
- $0 < S_{\text{step}} < S_{\text{terrain}}$: $0° < \theta < 90°$ (descending traverse)
- $S_{\text{step}} = 0$ (contour): $\arccos(0) = 90°$ — a traverse across the slope
- $S_{\text{step}} < 0$ (climb): $\theta > 90°$ (tilts against the fall line)
- Clamp to $[2°, 178°]$ (keeps left/right diverging and off exactly straight up/down); $S_{\text{terrain}} = 0$ (flat DEM cell) → $\theta = 90°$

**Step 3: Calculate Step Bearing**

$$\theta_{\text{step}} = \theta_{\text{fall}} + \text{sign} \cdot \theta_{\text{traverse}} + \epsilon$$

Where:
- $\theta_{\text{fall}}$ is the fall line — **always** the reference; direction (descend/climb) is carried by $\theta_{\text{traverse}}$ crossing 90°, never by flipping the reference
- $\text{sign} = -1$ for Left, $+1$ for Right
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is adaptive Gaussian noise

**Adaptive Noise:** Scale noise inversely with traverse angle to prevent Green paths on steep terrain from drifting to Blue (floored at 0, so a climbing step $\theta > 90°$ gets no noise):

$$\sigma_{\text{adaptive}} = \sigma_{\text{base}} \cdot \max\left(0, \frac{90° - \theta_{\text{traverse}}}{90°}\right)$$

**Step 4: Take Step**

$$(\text{lon}_{\text{new}}, \text{lat}_{\text{new}}) = \textrm{destinationPoint}(\text{lon}, \text{lat}, \theta_{\text{step}}, \Delta d)$$

Where $\Delta d$ is the step size (default 30m).

**Step 5: Update State**

$$\text{accumulatedDrop} += z_{\text{current}} - z_{\text{new}}$$

The step drop is **signed**: positive when the step descended, negative when it climbed.

**Step 6: Loop** until $d_{\text{total}} \geq L_{\text{target}}$

### 5.7 Why Paths Curve Around Terrain

The **fall line direction changes** as you move across the mountain:

1. At Point A: $\theta_{\text{fall}} = 180°$ (south)
2. Move 30m to the left
3. At Point B: $\theta_{\text{fall}} = 195°$ (south-southwest)

This creates **natural curving**:
- On convex hills: paths curve outward
- On concave valleys: paths curve inward

### 5.8 Whole-Path Smoothing on Finish

Each segment is spline-smoothed independently at trace time, so two segments meet at a shared junction node with different tangents — a visible **kink**. When a slope or road is **finished**, the whole path is smoothed in one pass by a single cubic smoothing spline fitted over the full polyline (parametrised by cumulative distance, resampled every `RESAMPLE_STEP_M` ≈ 7 m), then re-sliced back to the original segments so the ribbon is continuous across junctions.

- The fit is a **weighted least-squares spline**: the boundary **nodes** get a moderately higher weight (`NODE_WEIGHT`) than the raw planner **corridor points** (`CORRIDOR_WEIGHT`), with a smoothing budget scaled by the point count. The planner's grid path is a staircase; at a switchback it reverses across sub-metre jitter. A *smoothing* spline averages that jitter into a real turn **radius**. The node weight is deliberately **moderate** (≈10, not huge): an extreme node weight makes the fit near-singular at the pinned point and manufactures a cusp there.
- **Roads and slopes smooth differently.** Roads use `ROAD_SMOOTHING_FACTOR` (≈50) — cars need broad, smooth curves and roads accept the earthwork. Slopes use the lower `SLOPE_SMOOTHING_FACTOR` (≈30) so the ribbon **hugs the terrain** more: skiers are flexible, and a slope should follow the ground rather than build up large cut/fill.
- **Outer endpoints are pinned exactly** (the entity termini, shared with other slopes/lifts/roads). **Internal junctions** are left where the weighted spline places them — about half a metre from the node, shared by value between the two adjacent segments — so the node marker still sits on the ribbon and any node can be a branch point, without snapping a switchback back into a kink.
- Elevation is **smoothed along the spline, not re-sampled from the DEM**. A finished deck may therefore float slightly off the ground between nodes — treat it as a bridge / cut / fill.
- Finish smoothing **never rejects** a path and does **not** re-apply the ±15% road cap (§7.3). Rounding a corner can nudge a road's steepest 300 m section; a finished road is allowed to exceed the build cap (bridge/cut/fill).

---

## 6. Difficulty Classification

### 6.1 Segment Classification

When a path is committed, it becomes a segment classified by the **steepest 300m section** (rolling window):

$$S_{\text{max}} = \max_{\text{window}} \left( \frac{\Delta h_{\text{window}}}{L_{\text{window}}} \right) \times 100\%$$

| Steepest Section | Classification |
|------------------|----------------|
| < 15% | 🟢 Green |
| 15% – 25% | 🔵 Blue |
| 25% – 40% | 🔴 Red |
| ≥ 40% | ⚫ Black |

### 6.2 Slope Classification (Multi-Segment)

The final slope classification is determined by the **steepest section among all segments**. A short steep section will make the entire slope not skiable for beginners, even if the overall average is low.

---

## 7. Custom Direction / Connect Paths

When the automatically generated fan paths don't include the direction you want, just **click the point you want to reach** while building the slope — terrain-adaptive path(s) are routed to that target. Clicking an existing node routes a connector that auto-finishes the slope on commit.

### 7.1 Multi-Grade Path Search

The algorithm tries **8 difficulty-grade targets** (4 difficulties × gentle/steep) to find viable paths:

| Difficulty | Grades | Total |
|------------|--------|-------|
| 🟢 Green | Gentle (7%), Steep (12%) | 2 |
| 🔵 Blue | Gentle (17%), Steep (22%) | 2 |
| 🔴 Red | Gentle (28%), Steep (37%) | 2 |
| ⚫ Black | Gentle (45%), Steep (60%) | 2 |

Unlike the fan tracer (§5.3), the grid-Dijkstra planner has no left/right **side** — it finds the single least-cost route for a target grade, so there are 8 searches, not 16. Similar paths are deduplicated, keeping only the easiest difficulty when paths overlap.

### 7.2 Grid-Based Dijkstra Algorithm

Each path variant uses **Dijkstra's algorithm** (via SciPy's C-optimized implementation) to find terrain-adaptive paths:

**Algorithm Phases:**

1. **Grid Construction:** Create a grid of candidate points (15m spacing) covering the area between start and target with a buffer zone.

2. **Graph Building:** Each grid cell connects to its 8 neighbors. Edge costs are computed based on terrain slope.

3. **Dijkstra Search:** SciPy's `shortest_path()` finds the minimum-cost path through the sparse graph.

4. **Spline Smoothing:** The raw grid path has staircase artifacts (only 8 movement directions). A cubic smoothing spline is fitted through the points and resampled at 7m intervals. Elevations are re-queried from the DEM for accuracy.

**Cost Function:**

$$\text{cost} = d \times \exp\left(\frac{|\text{slope}_{\text{actual}} - \text{slope}_{\text{target}}|}{\sigma}\right) \times P_{\text{uphill}}$$

Where:
- $d$ = horizontal distance between grid nodes (~15m cardinal, ~21m diagonal)
- $\sigma$ = slope sensitivity parameter (default: 8)
- $P_{\text{uphill}}$ = uphill penalty: 1.0 if downhill, $\exp(|\text{slope}|/\sigma)$ if uphill



**Advantages:**
- Fast: SciPy's C implementation provides 10-50x speedup over pure Python
- Terrain-adaptive: naturally creates traverses on steep terrain
- Smooth output: spline interpolation removes grid artifacts
- Robust: soft uphill penalty handles DEM noise without hard cutoffs

### 7.3 Roads (for cars)

A **Road** is a vehicle road built **segment-by-segment**. Like a slope, a road is routed two ways: a **fan** that radiates candidate routes from the current endpoint (§7.3.1), and **custom-connect** to a clicked target via grid-Dijkstra with a direct-line fallback (§7.3.2). The one road-specific rule is the **±15% hard cap** (§7.3.3).

**Target grade.** A road reuses the **green** slope targets (7% gentle, 12% steep, from `SlopeConfig.DIFFICULTY_TARGETS["green"]`). Because a road may climb, descend, or run flat, the targets are **signed**: descend ($+$), climb ($-$), and a flat contour ($0$).

#### 7.3.1 Road fan

Like the slope fan, the road fan is traced by the gradient-agnostic tracer (§5) from the current endpoint. Its target set is the signed green grades plus a contour:

$$g_{\text{target}} \in \{+7, +12, -7, -12, 0\}$$

giving up to five spokes (each a left/right traverse, or a center path where the target magnitude meets the terrain steepness). There is no center-stop rule (only five targets). Every spoke is hard-capped at ±15% (§7.3.3); on steep ground the steep-green spokes are filtered out while the gentle and contour spokes survive, so the fan degrades gracefully rather than emptying.

#### 7.3.2 Custom-connect + straight-line fallback

Clicking a target routes to it by the same grid-Dijkstra algorithm and cost function as §7.2, against the signed green $g_{\text{target}}$. A road picks its `GradientMode` from the endpoints — `DOWNHILL` when it descends, `UPHILL` when it climbs — and the monotonicity penalty keeps the segment one-way (no looping), exactly as for a descent-only slope. On gentle ground both targets collapse to one straight route; on steep ground each serpentines, giving two proposals.

If **no** serpentine fits within ±15%, the caller offers a **direct road** (a straight 2-point line, treated as a bridge/cut) — but **only if that direct line itself is within ±15%**. This is the key slope-vs-road difference: a slope's straight-line fallback is *always* offered (any grade is a valid, if steep, run), whereas a road is genuinely **refused** ("Too steep for a car road") when even the direct line is too steep.

#### 7.3.3 ±15% is a HARD cap at build time

The exponential cost term is only a *soft* preference; every road proposal — fan spoke, serpentine, or direct fallback — is additionally **hard-capped** at $g_{\max} = 15\%$ (`ROAD_MAX_GRADIENT_PCT`) by the caller. Since `max_slope_pct` is a magnitude, the cap catches steep climbs and descents alike. A committed road therefore never exceeds the cap. The green targets and $g_{\max}$ are single-sourced constants — no hardcoded percentages.

---

## 8. Lift Pylon Placement

### 8.1 Cable Sag Model

Using normalized position $t = x/L$ where $t \in [0, 1]$:

$$z_{\text{cable}}(t) = (1-t) \cdot z_0 + t \cdot z_1 - 4 \cdot s \cdot t(1-t)$$

| Variable | Description |
|----------|-------------|
| $z_0$ | Cable elevation at start pylon |
| $z_1$ | Cable elevation at end pylon |
| $s$ | Max sag at midpoint: $s = \textrm{sagFactor} \times L$ |

**Sag Factors:** Typically 5-6% for most lift types, accounting for cable weight and passenger loading.

### 8.2 3-Phase Catenary Algorithm

**Phase 1 — Clearance Violations:**
Find where `cable_elev - terrain_elev < min_clearance` and place a pylon at the worst violation point.

**Phase 2 — Max Spacing Enforcement:**
If any span exceeds `max_spacing_m`, insert a midpoint pylon.

**Phase 3 — Re-check Clearance:**
Spacing pylons may affect adjacent spans. Re-run Phase 1 to fix new violations.

---

## 9. OpenStreetMap Import

An **Import from OpenStreetMap** control (sidebar, idle only) fetches the real lifts & pistes within a square area around the map center and adds them to the graph. **Geometry only** — we take just the lon/lat polylines and lift stations; elevation, difficulty, pylons, and **belt width** are all recomputed by our own pipeline. OSM's own attributes (including `piste:width`) are ignored.

### 9.1 Region + single-query fetch

A **square bounding box**: the current map center + a half-width from a slider (`HALF_WIDTH_MIN/MAX/DEFAULT_KM`). A lift/piste-only query is **light** — even a full-size box returns in a few seconds.


### 9.2 Mapping OSM → graph

- **Pistes** — `piste:type=downhill` only. The polyline is linearly resampled every `RESAMPLE_STEP_M` (~30 m, no cubic spline — OSM pistes are already smooth) with DEM elevation, then `commit_paths` → `finish_slope`. Difficulty comes from the DEM `max_slope_pct`; name from `name → piste:name → piste:ref → ref`.
- **Lifts** — import ONLY `aerialway` values defined by us (drag/t-bar/j-bar/platter → surface_lift, chair_lift → chairlift, gondola/mixed_lift → gondola, cable_car → aerial_tram). Any other value is ignored.

### 9.3 Only full, non-trivial, NAMED entities; one undoable batch

Every element that is not imported is logged with its reason. A way with any vertex outside the box, or over a DEM nodata hole, is skipped entirely (never half-imported). **Unnamed** lifts/pistes are skipped — they are frequently outdated or duplicate, so only named entities import. Lifts under `MIN_LIFT_LENGTH_M` (300 m) and pistes under `MIN_PISTE_LENGTH_M` (200 m) are skipped as trivial. The whole import is one `ImportOSMAction` — a single Undo removes it all.

### 9.4 Idempotent re-import

An incoming run is skipped if the graph already has a slope/lift with the **same two endpoints**.
