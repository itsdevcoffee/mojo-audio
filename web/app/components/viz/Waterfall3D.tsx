import { useRef, useEffect } from "react";
import { infernoRGB } from "./shared/inferno";

interface Waterfall3DProps {
  melData: number[];
  nMels: number;
  nFrames: number;
}

export function Waterfall3D({ melData, nMels, nFrames }: Waterfall3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!containerRef.current || melData.length === 0) return;

    import("three").then((THREE) => {
      const container = containerRef.current;
      if (!container) return;

      if (cleanupRef.current) cleanupRef.current();

      const W = container.clientWidth;
      const H = 280;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x060d0a);
      scene.fog = new THREE.Fog(0x060d0a, 8, 18);

      const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
      camera.position.set(0, 4, 7);
      camera.lookAt(0, 0, 0);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(W, H);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);

      const maxFrames = 300;
      const step = Math.max(1, Math.floor(nFrames / maxFrames));
      const usedFrames = Math.floor(nFrames / step);

      let min = Infinity, max = -Infinity;
      for (const v of melData) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min || 1;

      const geo = new THREE.PlaneGeometry(6, 4, usedFrames - 1, nMels - 1);
      const positions = geo.attributes.position;
      const colors = new Float32Array(positions.count * 3);
      const maxH = 1.5;

      for (let j = 0; j < nMels; j++) {
        for (let fi = 0; fi < usedFrames; fi++) {
          const srcFrame = fi * step;
          const idx = srcFrame * nMels + j;
          const val = (melData[idx] - min) / range;
          const vertIdx = j * usedFrames + fi;
          (positions.array as Float32Array)[vertIdx * 3 + 2] = val * maxH;

          const [r, g, b] = infernoRGB(val);
          colors[vertIdx * 3] = r / 255;
          colors[vertIdx * 3 + 1] = g / 255;
          colors[vertIdx * 3 + 2] = b / 255;
        }
      }

      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.computeVertexNormals();
      geo.rotateX(-Math.PI / 2);

      const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 30,
      });

      scene.add(new THREE.Mesh(geo, mat));
      scene.add(new THREE.AmbientLight(0xffffff, 0.4));
      const dir = new THREE.DirectionalLight(0xffffff, 0.8);
      dir.position.set(2, 5, 3);
      scene.add(dir);

      let angle = 0;
      let animId: number;

      function animate() {
        animId = requestAnimationFrame(animate);
        angle += 0.003;
        camera.position.x = 7 * Math.sin(angle);
        camera.position.z = 7 * Math.cos(angle);
        camera.lookAt(0, 0, 0);
        renderer.render(scene, camera);
      }
      animate();

      cleanupRef.current = () => {
        cancelAnimationFrame(animId);
        renderer.dispose();
        geo.dispose();
        mat.dispose();
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      };
    });

    return () => {
      if (cleanupRef.current) cleanupRef.current();
    };
  }, [melData, nMels, nFrames]);

  return (
    <div className="viz-content">
      <div className="viz-content__header">
        <span className="panel-label">3D Waterfall · {nMels} bands</span>
        <span className="panel-meta">auto-rotate</span>
      </div>
      <div ref={containerRef} style={{ width: "100%", height: 280, borderRadius: 4, overflow: "hidden" }} />
    </div>
  );
}
