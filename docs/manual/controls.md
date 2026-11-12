# Controls

## Camera

- **Right-drag**: orbit (right = scene swings left, like the original)
- **Wheel**: zoom · **W A S D / Q E**: pan · **double-click**: focus
- **F**: frame the whole network

## Selecting

- Click an object to select it; **Shift-click** adds to the selection.
- Drag on empty space draws a box select.
- Esc clears a drag/connect first, then the mode.

## Moving objects

Drag a selected object (navigate or move mode) to slide it on the
work plane — the dragged grid shows the plane through the grabbed
object, with a line from the start to the preview position.

During a move drag:

- **Shift** (hold): vertical plane facing the camera (x–y or y–z,
  chosen by where you look from). Releasing Shift returns to the
  toolbar plane.
- **X / Y / Z**: lock to that axis.
- **Esc**: cancel the move (snap back).
- Switching plane or axis re-anchors under the cursor — the object
  never jumps.

A hint strip at the bottom of the viewport shows the active plane and
the keys while dragging. Neurons and inputs move; readouts stay
auto-placed above their units.

## Context menu

Right-click: neuron → Watch v(t) / Connect from here / Delete;
input → Set value / Connect; readout → Plot weights; empty space →
insert, paste, frame.
