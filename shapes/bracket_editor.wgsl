// Wall bracket set up entirely with the Design step's primitives (bracket_editor.prims.json) instead
// of script fields; compare shapes/bracket_domain.wgsl. Open it in the app and select a primitive:
//   domain    box, add            the design domain (where material may go)
//   plate     box, add, keep      mounting plate, kept solid
//   bolt 1/2  cylinders, subtract, fixed     the bolt holes; their walls are clamped
//   pin       cylinder, subtract, load       100 N down on the pin hole's wall (load case 1)
//   pin side  cylinder, region, load         50 N sideways on the same wall (load case 2)
// The part is optimized for both load cases (their summed compliance).
//! bounds -40 -12 0  40 12 40
//! voxel 1
