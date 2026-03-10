/** Inferno colormap — 256 RGB triplets, values 0-255 */
export const INFERNO_LUT: [number, number, number][] = [];

// Generate inferno colormap using smoothstep approximation
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  const r = Math.round(255 * Math.min(1, Math.max(0,
    -0.0155 + t * (5.3711 + t * (-14.099 + t * (13.457 - t * 4.716))))));
  const g = Math.round(255 * Math.min(1, Math.max(0,
    0.0109 + t * (-0.670 + t * (3.448 + t * (-5.691 + t * 3.889))))));
  const b = Math.round(255 * Math.min(1, Math.max(0,
    0.178 + t * (3.298 + t * (-12.425 + t * (17.326 - t * 8.376))))));
  INFERNO_LUT.push([r, g, b]);
}

/** Map a [0,1] value to inferno RGB */
export function infernoRGB(t: number): [number, number, number] {
  const idx = Math.round(Math.min(1, Math.max(0, t)) * 255);
  return INFERNO_LUT[idx];
}
