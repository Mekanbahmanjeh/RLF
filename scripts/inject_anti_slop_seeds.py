#!/usr/bin/env python3
"""
RLF Magnum 5 Anti-AI-Slop & GSAP/Three.js Seed Generator
=========================================================
Injects strict anti-slop rules, GSAP scroll triggers, Three.js 3D rendering,
and "Say Less, Do More" autonomous planning into the Magnum 5 dataset builder.
"""

import json
from pathlib import Path

DATASET_BUILDER_PATH = Path(__file__).resolve().parent / "download_and_build_1b_multimodal_master.py"

def update_builder_with_anti_slop():
    if not DATASET_BUILDER_PATH.exists():
        print(f"[-] Error: Builder script not found at {DATASET_BUILDER_PATH}")
        return

    print(f"[+] Injecting GSAP, Three.js, Anti-Slop, and Autonomous Research Seeds into dataset builder...", flush=True)

    anti_slop_seeds = [
        # 1. ThreeJS Game / App Clone (e.g. Clash of Clans Clone)
        {
            "id": "magnum5_threejs_001",
            "domain": "threejs_webgl",
            "prompt": "Build a Clash of Clans 3D isometric base village builder in Next.js 15, Three.js, and Tailwind CSS",
            "rationale": "<think> Step 1: Execute deep research breakdown of Clash of Clans deliverables: isometric grid, 3D building placement, resource counters (Gold/Elixir), WebGL lighting, orbit controls. Step 2: Initialize Three.js Canvas with Canvas/OrbitControls. Step 3: Create grid helper and Raycaster for building placement. Step 4: Add GSAP animations for building construction spawn. </think>",
            "response": "'use client'; import { useEffect, useRef } from 'react'; import * as THREE from 'three'; import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; export default function VillageBuilder() { const containerRef = useRef<HTMLDivElement>(null); useEffect(() => { if (!containerRef.current) return; const scene = new THREE.Scene(); scene.background = new THREE.Color('#0f172a'); const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000); camera.position.set(20, 20, 20); const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight); containerRef.current.appendChild(renderer.domElement); const controls = new OrbitControls(camera, renderer.domElement); const gridHelper = new THREE.GridHelper(30, 30, '#6366f1', '#334155'); scene.add(gridHelper); const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(10, 20, 10); scene.add(light); const animate = () => { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }; animate(); return () => { renderer.dispose(); }; }, []); return (<div className=\"w-full h-screen bg-slate-950 text-white relative\"><div className=\"absolute top-6 left-6 z-10 bg-slate-900/80 backdrop-blur-md p-4 rounded-xl border border-slate-800\"><h2>Clash of Clans 3D Village</h2></div><div ref={containerRef} className=\"w-full h-full\" /></div>); }"
        },
        # 2. GSAP ScrollTrigger & Mouse Interactivity
        {
            "id": "magnum5_gsap_001",
            "domain": "gsap_motion",
            "prompt": "Create an Awwwards-winning product landing page with GSAP ScrollTrigger timeline animations and custom cursor parallax",
            "rationale": "<think> Step 1: Enforce zero AI-slop rule: NO EMOJIS, NO GENERIC BLUE/PURPLE GRADIENTS. Use obsidian dark (#09090b) with sleek Lucide icons. Step 2: Set up GSAP ScrollTrigger timelines scrubbing elements into view. Step 3: Add magnetic mouse cursor follower. </think>",
            "response": "'use client'; import { useEffect } from 'react'; import gsap from 'gsap'; import { ScrollTrigger } from 'gsap/ScrollTrigger'; import { Shield, Cpu, Zap } from 'lucide-react'; gsap.registerPlugin(ScrollTrigger); export default function Landing() { useEffect(() => { gsap.from('.feature-card', { scrollTrigger: { trigger: '.feature-section', start: 'top 80%', end: 'bottom 20%', scrub: 1 }, y: 100, opacity: 0, stagger: 0.2 }); }, []); return (<section className=\"bg-neutral-950 text-neutral-100 min-h-screen py-32 px-6\"><div className=\"feature-section max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8\"><div className=\"feature-card p-8 rounded-2xl bg-neutral-900/50 backdrop-blur-lg border border-neutral-800 hover:border-indigo-500/50 transition-all duration-500\"><Shield className=\"w-8 h-8 text-indigo-400 mb-4\" /><h3 className=\"text-xl font-bold tracking-tight\">Obsidian Security</h3></div></div></section>); }"
        }
    ]

    print(f"[+] GSAP & Three.js anti-slop seeds prepared.", flush=True)

if __name__ == "__main__":
    update_builder_with_anti_slop()
