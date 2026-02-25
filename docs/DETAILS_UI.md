# 🎮 User Guide: Alpin Architect

A complete guide to designing ski resorts with Alpin Architect.

For technical/mathematical details, see [DETAILS.md](DETAILS.md).

---

## Quick Overview

Alpin Architect lets you design ski slopes on real terrain by clicking on a map:

1. **Click** anywhere on terrain to generate path proposals
2. **Select** a path by clicking on it
3. **Review** the stats (gradient, length, warnings)
4. **Commit** the path with a button or by clicking on the orange endpoint — new proposals appear from the endpoint
5. **Finish** the slope when complete

---

## The Main Interface

### Map (Center)

| Element | Appearance | Meaning |
|---------|------------|---------|
| **Proposed paths** | Dashed colored lines | Uncommitted path options |
| **Selected path** | Bold dashed line | Currently selected |
| **Committed slopes** | Solid colored ribbons | Finalized segments |
| **Nodes** | White circles | Junction points |
| **Active endpoints** | Yellow circles | Continue building here |
| **Lifts** | Purple dashed lines | Ski lifts |
| **Orientation arrows** | Gray arrows | Fall line direction |

### Right Panel

| Section | Purpose |
|---------|---------|
| **Selected Path Stats** | Gradient, drop, length, warnings |
| **Commit Button** | Finalize the selected path |
| **Custom Direction** | Pick a target point not in proposals |
| **Elevation Profile** | Visual profile of descent |

### Sidebar (Left)

| Control | Function |
|---------|----------|
| **Mode selector** | Switch between Slopes / Lifts |
| **Segment Length** | Control path length (100-1000m) |
| **Finish Slope** | Complete current slope |
| **Undo / Clear** | Remove last action |
| **Stats** | Total slopes, drop, length |
| **Export GPX** | Download design |

---

## Designing Slopes

### Step 1: Click to Start

1. Navigate to any mountain (pan/zoom)
2. **Click on terrain** to place a starting point
3. Path proposals appear radiating from the click

### Step 2: Select a Path

1. **Click on any dashed line** to select it
2. The path becomes bold
3. Stats appear in the right panel

### Step 3: Review Stats

| Stat | What to Look For |
|------|------------------|
| **Overall Gradient** | Average over full segment |
| **Steepest Section** | Determines difficulty classification |
| **Drop** | Vertical meters |
| **Warnings** | Construction issues |

### Step 4: Commit

1. Click **"✅ Commit This Path"**
2. Path becomes a solid ribbon
3. New proposals appear from endpoint
4. Repeat to continue building

### Step 5: Finish

1. Click **"🏁 Finish Slope"**
2. Final difficulty = maximum segment difficulty
3. All segments get unified name

---

## Path Proposals

### How Many Paths?

The number varies by terrain steepness:

**On steep terrain:** Up to 16 paths (4 difficulties × 2 variants × 2 sides)

**On flat terrain:** Fewer paths due to "center-stop rule" — when paths go straight down, there's no left/right distinction

### Path Variants

For each difficulty level two steepness variants and if terrain is steep enough, also left and right variants are generated:

| Label | Meaning |
|-------|---------|
| **Left-Gentle** | Traverse left, easier target |
| **Left-Steep** | Traverse left, steeper target |
| **Right-Gentle** | Traverse right, easier target |
| **Right-Steep** | Traverse right, steeper target |
| **Center** | Straight down fall line |

### Path Ordering

Paths are sorted **left-to-right** relative to the fall line. Arrow buttons cycle through in spatial order.

---

## Custom Direction / Connect

When the automatically proposed paths don't include the direction you want, use **🎯 Custom Direction** to manually select a target:

### How It Works

1. Click **"🎯 Custom Direction"** button in the path selection panel
2. All current proposals are hidden
3. **Click anywhere downhill** on the map to set your target
4. A path is generated to your clicked point using terrain-adaptive traversal

### Two Use Cases

| Click Target | What Happens |
|--------------|--------------|
| **On terrain** | Creates a path to that location → new node created → continue building |
| **Directly on node marker** | Creates a connection path → **auto-finishes the slope** |

**Note:** Target must be within the current segment length (slider). Increase the slider to reach distant nodes.

---

## Warnings

### 🚜 Side Cut Warning

**What:** Path needs excavation to create level piste

**Why:** Traversing across steep terrain creates cross-slope

**OK?** Yes — just needs construction work

###  Too Flat Warning

**What:** Average slope < 5% — may need to pole

**Why:** Terrain is very gentle

**OK?** Yes — valid connector segment



---

## Difficulty Classification

After finishing a slope, its difficulty is determined by the **steepest 300m section** within any segment. The classification follows the European standard:
| Color | Steepest Section |
|-------|------------------|
| 🟢 Green | < 15% |
| 🔵 Blue | 15% – 25% |
| 🔴 Red | 25% – 40% |
| ⚫ Black | ≥ 40% |

**Note:** Final classification is based on actual terrain, not target difficulty. A "Black" path on flat terrain becomes Green.

---

## Starting from Nodes

Click on any **white circle** (node) to start new paths from that junction. This enables:
- Branching slopes
- Alternative descents
- Connecting paths

---

## Designing Lifts

### Step 1: Switch Mode

Click **"Design Lifts"** in sidebar

### Step 2: Select Start Point

Click on the map to set the bottom station:
- **On an existing node** — uses that junction
- **On empty terrain** — creates a new node at that location

A purple marker with uphill arrow appears.

### Step 3: Select End Point

Click on the map to set the top station:
- **On an existing node** — uses that junction
- **On empty terrain** — creates a new node at that location

The lift is created once both points are set.

### Lift Types

| Type | Purpose |
|------|---------|
| **Surface Lift** | Short distances |
| **Chairlift** | Standard lift |
| **Gondola** | Longer distances, weather protection |
| **Aerial Tram** | Very long spans |

---

## Elevation Profile

Shows the path descent:
- **X-axis:** Distance (meters)
- **Y-axis:** Elevation (meters)
- **Color:** Matches difficulty

---

## Undo

**↩️ Undo** removes the last action:
- In BUILDING: removes last segment
- After FINISH: restores slope to building mode

---

## Segment Length

The slider (100-1000m) controls path segment length. Paths auto-regenerate when changed.

---

## Tips

### Good Starting Points
✅ Ridges, summits, peaks, saddles

### Avoid
❌ Valley floors, flat plateaus

### Slope Length
✅ 3-6 segments (300-800m vertical)
❌ Single segments or 20+ segments

### Lift Placement
✅ Valley floor to summit
❌ Along the slope direction

---

## Keyboard & Mouse

| Action | How |
|--------|-----|
| Pan | Click + drag |
| Zoom | Mouse wheel |
| Start slope | Click terrain |
| Select path | Click dashed line |
| Start from node | Click white circle |
| Place lift | Click on map (Lift mode) |

---
