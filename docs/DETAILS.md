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

### 3.3 Calculating Traverse Angle from Target

The designer sets a **target effective slope** ($S_{\text{target}}$) for each path. The algorithm calculates the required traverse angle:

$$\theta = \arccos\left(\frac{S_{\text{target}}}{S_{\text{terrain}}}\right)$$

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

The algorithm:
1. **Fixes the target effective slope** (what the skier experiences)
2. **Dynamically calculates the traverse angle** at each step based on local terrain

This allows paths to **naturally curve around terrain features**.

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
1. $\text{remainingDrop} = \text{targetTotalDrop} - \text{accumulatedDrop}$
2. $\text{remainingDistance} = L_{\text{target}} - d_{\text{total}}$
3. $S_{\text{step}} = \frac{\text{remainingDrop}}{\text{remainingDistance}} \times 100$ (clamped to $[0.3, 2.5] \times S_{\text{target}}$)

**Why this works:** The path self-corrects toward the target average without retries.

### 5.6 Step-by-Step Tracing

**Step 1: Sample Local Terrain (Midpoint Sampling)**

Sample at the midpoint of each step to prevent "lag":
$$S_{\text{terrain}}, \theta_{\text{fall}} = \textrm{getTerrainGradient}(\text{midpoint})$$

**Step 2: Calculate Traverse Angle**

$$\theta_{\text{traverse}} = \arccos\left(\frac{S_{\text{step}}}{S_{\text{terrain}}}\right)$$

Cases:
- If $S_{\text{step}} \geq S_{\text{terrain}}$: $\theta = 0°$ (straight down)
- If $S_{\text{step}} < S_{\text{terrain}}$: Calculate traverse angle
- Clamp to $[0°, 89°]$

**Step 3: Calculate Step Bearing**

$$\theta_{\text{step}} = \theta_{\text{fall}} + \text{sign} \cdot \theta_{\text{traverse}} + \epsilon$$

Where:
- $\text{sign} = -1$ for Left, $+1$ for Right
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is adaptive Gaussian noise

**Adaptive Noise:** Scale noise inversely with traverse angle to prevent Green paths on steep terrain from drifting to Blue:

$$\sigma_{\text{adaptive}} = \sigma_{\text{base}} \cdot \frac{90° - \theta_{\text{traverse}}}{90°}$$

**Step 4: Take Step**

$$(\text{lon}_{\text{new}}, \text{lat}_{\text{new}}) = \textrm{destinationPoint}(\text{lon}, \text{lat}, \theta_{\text{step}}, \Delta d)$$

Where $\Delta d$ is the step size (default 30m).

**Step 5: Update State**

$$\text{accumulatedDrop} += z_{\text{current}} - z_{\text{new}}$$

**Step 6: Loop** until $d_{\text{total}} \geq L_{\text{target}}$

### 5.7 Why Paths Curve Around Terrain

The **fall line direction changes** as you move across the mountain:

1. At Point A: $\theta_{\text{fall}} = 180°$ (south)
2. Move 30m to the left
3. At Point B: $\theta_{\text{fall}} = 195°$ (south-southwest)

This creates **natural curving**:
- On convex hills: paths curve outward
- On concave valleys: paths curve inward

---

## 6. Difficulty Classification

### 6.1 Segment Classification

When a path is committed, it becomes a segment classified by the **steepest 300m section** (rolling window):

$$S_{\text{max}} = \max_{\text{window}} \left( \frac{\Delta h_{\text{window}}}{L_{\text{window}}} \times 100\% \right)$$

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

When the automatically generated fan paths don't include the direction you want, the **Custom Direction** feature lets you click anywhere downhill to set a target.

### 7.1 Multi-Grade Path Search

The algorithm tries **16 combinations** (8 difficulty-grade variants × 2 sides) to find viable paths:

| Difficulty | Grades | Sides | Total |
|------------|--------|-------|-------|
| 🟢 Green | Gentle (7%), Steep (12%) | Left, Right | 4 |
| 🔵 Blue | Gentle (17%), Steep (22%) | Left, Right | 4 |
| 🔴 Red | Gentle (28%), Steep (37%) | Left, Right | 4 |
| ⚫ Black | Gentle (45%), Steep (60%) | Left, Right | 4 |

Similar paths are deduplicated, keeping only the easiest difficulty when paths overlap.

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
