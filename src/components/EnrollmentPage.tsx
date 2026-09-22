// @ts-nocheck
import React, { useEffect, useRef, useState } from "react";
import {
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import * as faceapi from "@vladmandic/face-api";

type LandmarkPoint = NormalizedLandmark;

const WASM_URL = "/wasm";
const MODEL_URL = "/models/face_detector_full_range.tflite";
const FACE_LANDMARKER_MODEL = "/models/face_landmarker.task";
const FACE_LANDMARKER_REMOTE =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const DEPTH_GRID_SIZE = 7;
const MAX_PEOPLE = 7;
const MIN_FACE_WIDTH = 16;
const MIN_FACE_HEIGHT = 16;

// Go API

const EMPLOYEE_CODES_API = `/api/v1/employees/codes`;
const FACE_ENROLL_API = `/api/v1/face/enroll`;

// LocalStorage: เก็บฐานข้อมูลใบหน้า + ภาพที่สแกนเจอ
const FACE_DB_KEY = "saved_faces";
const FACE_SCAN_HISTORY_KEY = "face_scan_history_v1";
const MAX_SCAN_HISTORY = 40;
const MAX_SCAN_GALLERY_PER_PERSON = 3;
const MAX_APPEARANCE_IMAGES_PER_PERSON = 5;
const MAX_CONCURRENT_RECOGNITION = 4;
const SCAN_SNAPSHOT_INTERVAL_MS = 2200;

type EnrollmentPageProps = {
  onCameraStateChange?: (active: boolean) => void;
};

export default function EnrollmentPage({ onCameraStateChange }: EnrollmentPageProps) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const meshLastTimestampRef = useRef(0);
  const lastMeshResultRef = useRef(null);
  const depthProfilesRef = useRef({});
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  // ป้องกัน React StrictMode เรียก init AI ซ้ำ
  const aiInitStartedRef = useRef(false);
  const aiInitPromiseRef = useRef(null);
  
  // Refs สำหรับลอจิกติดตามภาพและ AI
  const tracksRef = useRef([]);
  const lastFrameTimestampRef = useRef(0);
  const lastDetectorRunRef = useRef(0);
  const recognitionActiveRef = useRef(0);
  const nextDetectionAtRef = useRef(0);
  const savedFacesRef = useRef([]);
  const faceMatcherRef = useRef(null);

  // ดัชนีค้นหาในหน่วยความจำ: ไม่ต้องอ่าน/parse localStorage ทุกครั้งที่ Scan
  const faceSearchIndexRef = useRef([]);
  const scanHistoryRef = useRef([]);

  // ผลการค้นหาของ "รอบนี้" เท่านั้น
  // ตั้งใจไม่ใช้ localStorage/sessionStorage เพื่อให้รีเฟรชหน้าแล้วล้างทันที
  const recognizedSessionRef = useRef(new Map());
  const cameraOnRef = useRef(false);
  const modeRef = useRef("scan");
  const facingModeRef = useRef("environment");
  const enrollmentBusyRef = useRef(false);
  const enrollmentLoopRef = useRef(null);
  const enrollmentRef = useRef({
    active: false,
    stage: 0,
    stable: 0,
    samples: [],
    firstTurnSign: null,
    lastAt: 0,
    countdown: 0,
    countdownStartedAt: 0,
    targetLock: null,
  });
  
  // New refs for improved detection
  const smoothedBoxesRef = useRef({});
  const lastDetectedBoxesRef = useRef([]);
  const detectionHistoryRef = useRef({});
  const stabilityCounterRef = useRef({});

  const [cameraOn, setCameraOn] = useState(false);

  // แจ้ง App ว่ากล้องกำลังเปิด/ปิด เพื่อซ่อนหรือแสดง Bottom Navigation
  useEffect(() => {
    onCameraStateChange?.(cameraOn);
    return () => onCameraStateChange?.(false);
  }, [cameraOn, onCameraStateChange]);
  const [mode, setMode] = useState("scan");

  // พนักงานที่เลือกจาก API /api/v1/employees/codes
  const [employees, setEmployees] = useState([]);
  const [employeeSearch, setEmployeeSearch] = useState("");
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [employeeLoading, setEmployeeLoading] = useState(true);
  const [employeeError, setEmployeeError] = useState("");

  // ชื่อใช้แสดงผลบน UI เท่านั้น — ID จาก selectedEmployee คือ key จริง
  const [enrollName, setEnrollName] = useState("");
  const [enrollStage, setEnrollStage] = useState(0);
  const [enrollStable, setEnrollStable] = useState(0);
  const [enrollMessage, setEnrollMessage] = useState("");
  const [enrollmentStarted, setEnrollmentStarted] = useState(false);
  const [enrollCountdown, setEnrollCountdown] = useState(0);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [cameraId, setCameraId] = useState("");
  const [cameraLabel, setCameraLabel] = useState("กล้อง");
  const [loading, setLoading] = useState(true);
  const [faceCount, setFaceCount] = useState(0);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [savedFaces, setSavedFaces] = useState([]);
  const [cameraError, setCameraError] = useState("");
  const [scanMessage, setScanMessage] = useState("กำลังเตรียม AI ตรวจจับใบหน้า...");
  const [scanActivity, setScanActivity] = useState("กำลังเริ่มระบบตรวจจับใบหน้า...");
  const [cameraQuality, setCameraQuality] = useState("");
  const [cameraMirrored, setCameraMirrored] = useState(false);
  const [meshReady, setMeshReady] = useState(false);
  const [faceGeometry, setFaceGeometry] = useState(null);


  // MediaPipe Face Landmarker returns 3D landmarks (x, y, z).  The z values are
  // relative model depth, not millimeters from a real depth sensor.
  const buildDepthProfile = (landmarks: LandmarkPoint[]) => {
    if (!landmarks || landmarks.length < 400) return null;

    const xs = landmarks.map(p => p.x);
    const ys = landmarks.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const faceW = Math.max(0.0001, maxX - minX);
    const faceH = Math.max(0.0001, maxY - minY);
    const centerZ = landmarks.reduce((s, p) => s + (p.z || 0), 0) / landmarks.length;

    const grid = Array.from({ length: DEPTH_GRID_SIZE * DEPTH_GRID_SIZE }, () => ({ sum: 0, n: 0 }));
    landmarks.forEach((p) => {
      const gx = Math.min(DEPTH_GRID_SIZE - 1, Math.max(0, Math.floor(((p.x - minX) / faceW) * DEPTH_GRID_SIZE)));
      const gy = Math.min(DEPTH_GRID_SIZE - 1, Math.max(0, Math.floor(((p.y - minY) / faceH) * DEPTH_GRID_SIZE)));
      const cell = grid[gy * DEPTH_GRID_SIZE + gx];
      cell.sum += (p.z || 0) - centerZ;
      cell.n += 1;
    });

    const values = grid.map(c => c.n ? c.sum / c.n : null);
    const valid = values.filter(v => v !== null);
    const meanAbs = valid.length ? valid.reduce((s, v) => s + Math.abs(v), 0) / valid.length : 0;

    const point = (i) => landmarks[i] || { x: 0, y: 0, z: centerZ };
    const region = (name, ids) => {
      const pts = ids.map(point);
      const z = pts.reduce((s, p) => s + (p.z || 0), 0) / pts.length;
      return { name, z: Number((z - centerZ).toFixed(5)), depth: Number(((centerZ - z) / faceW * 100).toFixed(1)) };
    };

    const regions = [
      region("หน้าผาก", [10, 151, 337]),
      region("สันจมูก", [168, 6, 197]),
      region("ปลายจมูก", [1, 2, 4]),
      region("แก้มซ้าย", [117, 118, 123, 50, 101]),
      region("แก้มขวา", [346, 347, 352, 280, 330]),
      region("รอบตาซ้าย", [33, 133, 159, 145]),
      region("รอบตาขวา", [263, 362, 386, 374]),
      region("ปาก", [13, 14, 78, 308]),
      region("คาง", [152, 175, 199]),
    ];

    return {
      grid: values,
      regions,
      meanAbs: Number(meanAbs.toFixed(5)),
      faceW,
      faceH,
      centerZ: Number(centerZ.toFixed(5)),
      timestamp: Date.now(),
    };
  };

  const compareDepthProfiles = (a, b) => {
    if (!a?.grid || !b?.grid) return null;
    const pairs = a.grid.map((v, i) => [v, b.grid[i]]).filter(([x, y]) => x !== null && y !== null);
    if (pairs.length < 20) return null;
    const rmse = Math.sqrt(pairs.reduce((s, [x, y]) => s + (x - y) ** 2, 0) / pairs.length);
    // Relative shape score. This is intentionally a heuristic, not a biometric certification score.
    return Number(Math.max(0, Math.min(100, 100 - rmse * 900)).toFixed(1));
  };

  const canvasToCompressedJpeg = (sourceCanvas, maxSide = 512, quality = 0.78) => {
    const srcW = Math.max(1, sourceCanvas.width || 1);
    const srcH = Math.max(1, sourceCanvas.height || 1);
    const scale = Math.min(1, maxSide / Math.max(srcW, srcH));

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(srcW * scale));
    canvas.height = Math.max(1, Math.round(srcH * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) return "";

    ctx.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
  };

  const safeSetLocalStorage = (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      console.log(`✅ Saved to localStorage: ${key}`, { itemCount: value?.length || 0 });
      return true;
    } catch (error) {
      console.warn(`❌ LocalStorage write failed: ${key}`, error);
      return false;
    }
  };

  const loadScanHistory = () => {
    try {
      const parsed = JSON.parse(localStorage.getItem(FACE_SCAN_HISTORY_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  const persistScanHistory = (items) => {
    const trimmed = items.slice(-MAX_SCAN_HISTORY);
    scanHistoryRef.current = trimmed;

    // ถ้า quota เต็ม ให้ลดภาพเก่าก่อน แต่ไม่แตะ saved_faces
    if (safeSetLocalStorage(FACE_SCAN_HISTORY_KEY, trimmed)) return trimmed;

    const smaller = trimmed.slice(-20);
    if (safeSetLocalStorage(FACE_SCAN_HISTORY_KEY, smaller)) {
      scanHistoryRef.current = smaller;
      return smaller;
    }

    return scanHistoryRef.current;
  };

  const saveScanSnapshot = ({
    image,
    descriptor,
    personId = null,
    personName = null,
    distance = null,
    source = "scan",
  }) => {
    if (!image || !Array.isArray(descriptor) || !descriptor.length) return;

    const history = scanHistoryRef.current.length
      ? scanHistoryRef.current
      : loadScanHistory();

    const entry = {
      id: crypto.randomUUID(),
      image,
      descriptor,
      personId,
      personName,
      distance: Number.isFinite(distance) ? Number(distance.toFixed(4)) : null,
      source,
      capturedAt: new Date().toISOString(),
    };

    persistScanHistory([...history, entry]);
  };

  const normalizePersonImages = (person) => {
    const descriptors =
      Array.isArray(person?.descriptors) && person.descriptors.length
        ? person.descriptors
        : person?.descriptor
          ? [person.descriptor]
          : [];

    let faceImages = Array.isArray(person?.faceImages)
      ? person.faceImages.filter((item) => item?.image)
      : [];

    // Backward compatibility: ฐานข้อมูลเดิมมีภาพหลักอย่างน้อย 1 ภาพ
    if (!faceImages.length && person?.frontImage) {
      faceImages = [{
        id: `${person.id}-front`,
        stage: "front",
        image: person.frontImage,
        descriptor: descriptors[0] || null,
      }];
    }

    return {
      ...person,
      descriptors,
      faceImages,
      scanGallery: Array.isArray(person?.scanGallery)
        ? person.scanGallery.filter((item) => item?.descriptor)
        : [],
    };
  };

  const drawFaceMesh = (ctx: CanvasRenderingContext2D, landmarks: LandmarkPoint[], width: number, height: number, color: string = "#22d3ee") => {
    if (!landmarks?.length) return;

    // IMPORTANT: do not do an O(n²) nearest-neighbour search on all 478 points.
    // That was causing the camera loop to become heavy and appear frozen.
    const pts = landmarks.map((p) => ({ x: p.x * width, y: p.y * height }));
    const connections = [
      [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
      [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
      [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
      [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
      [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
      [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
      [33, 133], [133, 159], [159, 145], [145, 153], [153, 144], [144, 163],
      [263, 362], [362, 386], [386, 374], [374, 380], [380, 373], [373, 390],
      [61, 291], [291, 308], [308, 78], [78, 95], [95, 88], [88, 61],
      [1, 2], [2, 4], [4, 5], [5, 6], [6, 197], [197, 195], [195, 1],
    ];

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(0.8, width / 1800);
    ctx.globalAlpha = 0.78;

    for (const [a, b] of connections) {
      if (!pts[a] || !pts[b]) continue;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }

    // Lightweight point cloud: every 6th point plus key facial points.
    ctx.globalAlpha = 0.9;
    for (let i = 0; i < pts.length; i += 6) {
      const p = pts[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1.1, width / 1100), 0, Math.PI * 2);
      ctx.fill();
    }
    for (const i of [1, 4, 6, 10, 13, 152, 33, 133, 263, 362]) {
      const p = pts[i];
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(2, width / 700), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  };

  const buildEmployeeName = (employee) => {
    const explicitName = String(employee?.name || "").trim();
    if (explicitName) return explicitName;

    return [employee?.first_name_th, employee?.last_name_th]
      .map((value) => String(value || "").trim())
      .filter(Boolean)
      .join(" ")
      .trim();
  };

  const loadEmployees = async () => {
    setEmployeeLoading(true);
    setEmployeeError("");

    try {
      const response = await fetch(EMPLOYEE_CODES_API, {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        const body = await response.text().catch(() => "");
        throw new Error(`HTTP ${response.status}${body ? `: ${body.slice(0, 200)}` : ""}`);
      }

      const payload = await response.json();
      const rows = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.data)
          ? payload.data
          : Array.isArray(payload?.employees)
            ? payload.employees
            : [];

      const normalized = rows
        .map((employee) => ({
          ...employee,
          id: employee?.id != null ? String(employee.id) : "",
          employee_code: employee?.employee_code != null ? String(employee.employee_code) : "",
          first_name_th: employee?.first_name_th || "",
          last_name_th: employee?.last_name_th || "",
          name: buildEmployeeName(employee),
        }))
        .filter((employee) => employee.id && employee.name);

      setEmployees(normalized);
    } catch (error) {
      console.error("Load employee codes:", error);
      setEmployees([]);
      setEmployeeError(`โหลดรายชื่อพนักงานไม่สำเร็จ • ${error?.message || "ลองใหม่อีกครั้ง"}`);
    } finally {
      setEmployeeLoading(false);
    }
  };

  useEffect(() => {
    loadEmployees();
  }, []);

  useEffect(() => {
    let storedFaces = [];
    try {
      storedFaces = JSON.parse(localStorage.getItem(FACE_DB_KEY) || "[]");
    } catch {
      storedFaces = [];
    }

    if (!Array.isArray(storedFaces)) storedFaces = [];
    if (storedFaces.length > 0 && !storedFaces[0]?.descriptor && !storedFaces[0]?.descriptors) {
      storedFaces = [];
      localStorage.removeItem(FACE_DB_KEY);
    }

    storedFaces = storedFaces.map(normalizePersonImages);
    savedFacesRef.current = storedFaces;
    setSavedFaces(storedFaces);

    scanHistoryRef.current = loadScanHistory();

    let mounted = true;

    async function initAI() {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);

      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.40,
      });

      // Official remote Face Landmarker model.
      // The previous local /models/face_landmarker.task produced
      // "Unable to open zip archive" in the browser.
      const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_LANDMARKER_REMOTE,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: MAX_PEOPLE,
        minFaceDetectionConfidence: 0.45,
        minFacePresenceConfidence: 0.45,
        minTrackingConfidence: 0.50,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });

          const FACE_API_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";
      await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_URL);
      await faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_API_URL);
      await faceapi.nets.faceExpressionNet.loadFromUri(FACE_API_URL);

      return { detector, faceLandmarker };
    }

    if (!aiInitStartedRef.current) {
      aiInitStartedRef.current = true;
      aiInitPromiseRef.current = initAI();
    }

    aiInitPromiseRef.current
      .then(({ detector, faceLandmarker }) => {
        if (!mounted) return;
        detectorRef.current = detector;
        faceLandmarkerRef.current = faceLandmarker;
        updateFaceMatcher(storedFaces);
        setMeshReady(true);
        setLoading(false);
        setCameraError("");
        setScanMessage("AI พร้อมใช้งาน กดที่ภาพเพื่อเริ่มสแกน");
      setScanActivity("กล้องปิด • พร้อมเริ่มสแกนใหม่");
      })
      .catch((err) => {
        console.error("AI Init Error:", err);
        if (!mounted) return;
        setLoading(false);
        setCameraError("โหลดโมเดล AI ไม่สำเร็จ ตรวจสอบ /wasm และไฟล์ Face Detector");
        setScanMessage("ยังไม่พร้อมสแกน");
      });

    return () => {
      mounted = false;
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      cameraOnRef.current = false;
    };
  }, []);

  const updateFaceMatcher = (faces) => {
    if (!faces.length) {
      faceMatcherRef.current = null;
      faceSearchIndexRef.current = [];
      return;
    }

    const labeled = [];
    const searchIndex = [];

    for (const rawPerson of faces) {
      const person = normalizePersonImages(rawPerson);
      const descriptors =
        Array.isArray(person.descriptors) && person.descriptors.length
          ? person.descriptors
          : person.descriptor
            ? [person.descriptor]
            : [];

      const faceImages = Array.isArray(person.faceImages) ? person.faceImages : [];
      const scanGallery = Array.isArray(person.scanGallery) ? person.scanGallery : [];

      descriptors.forEach((descriptor, index) => {
        if (!Array.isArray(descriptor) || !descriptor.length) return;

        const imageEntry = faceImages[index];
        searchIndex.push({
          person,
          personId: person.id,
          vector: new Float32Array(descriptor),
          image: imageEntry?.image || null,
          angle: imageEntry?.stage || person.faceAngles?.[index] || null,
          source: "enrollment",
        });
      });

      // ภาพจากการ Scan ที่ยืนยันแล้ว ใช้เป็นตัวอย่างเสริมในอนาคต
      for (const galleryItem of scanGallery) {
        if (!Array.isArray(galleryItem.descriptor) || !galleryItem.descriptor.length) continue;

        searchIndex.push({
          person,
          personId: person.id,
          vector: new Float32Array(galleryItem.descriptor),
          image: galleryItem.image || null,
          angle: "scan",
          source: "confirmed-scan",
        });
      }

      const rawAll = searchIndex
        .filter((item) => item.personId === person.id)
        .map((item) => item.vector);

      if (rawAll.length) {
        labeled.push(new faceapi.LabeledFaceDescriptors(person.id, rawAll));
      }
    }

    faceSearchIndexRef.current = searchIndex;

    faceMatcherRef.current = labeled.length
      ? new faceapi.FaceMatcher(labeled, 0.42)
      : null;
  };


  // ภาพ Face: ต้องได้ "ทั้งศีรษะ/ทั้งใบหน้า" ไม่ใช่ครึ่งหน้า
  // ใช้กล่องสี่เหลี่ยมขนาดใหญ่รอบใบหน้า แล้วเลื่อนกลับเข้าเฟรมก่อน crop
  const captureFaceImage = (video, box) => {
    const videoW = Math.max(1, video.videoWidth || 640);
    const videoH = Math.max(1, video.videoHeight || 480);

    const bw = Math.max(1, Number(box?.width || 1));
    const bh = Math.max(1, Number(box?.height || 1));
    const cx = Number(box?.originX || 0) + bw / 2;
    const cy = Number(box?.originY || 0) + bh / 2;

    let side = Math.max(bw, bh) * 1.85;
    side = Math.max(96, Math.min(side, videoW, videoH));

    let x = cx - side / 2;
    let y = cy - side / 2;

    // ขยับกรอบกลับเข้าภาพ เพื่อไม่ตัดครึ่งหน้าเมื่อหน้าอยู่ชิดขอบ
    x = Math.max(0, Math.min(x, videoW - side));
    y = Math.max(0, Math.min(y, videoH - side));

    const canvas = document.createElement("canvas");
    const output = Math.min(768, Math.max(384, Math.round(side * 2.2)));
    canvas.width = output;
    canvas.height = output;

    const ctx = canvas.getContext("2d");
    if (!ctx) return canvas;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(video, x, y, side, side, 0, 0, output, output);

    return canvas;
  };

  // ภาพช่วงหัว-ไหล่/เสื้อผ้า ใช้เป็น "หลักฐานเสริม" ไม่แทน face descriptor
  const captureAppearanceImage = (video, box) => {
    const videoW = Math.max(1, video.videoWidth || 640);
    const videoH = Math.max(1, video.videoHeight || 480);

    const bw = Math.max(1, Number(box?.width || 1));
    const bh = Math.max(1, Number(box?.height || 1));

    const cx = Number(box?.originX || 0) + bw / 2;
    const cy = Number(box?.originY || 0) + bh / 2;

    // กว้างประมาณไหล่ + สูงลงไปถึงช่วงอก
    let cropW = Math.max(bw * 3.4, bh * 2.8);
    let cropH = Math.max(bh * 4.4, bw * 3.0);

    cropW = Math.min(cropW, videoW);
    cropH = Math.min(cropH, videoH);

    let x = cx - cropW / 2;
    // เลื่อนจุดกึ่งกลางลงเล็กน้อย เพื่อเก็บคอ/ไหล่/เสื้อ
    let y = cy - cropH * 0.34;

    x = Math.max(0, Math.min(x, videoW - cropW));
    y = Math.max(0, Math.min(y, videoH - cropH));

    const canvas = document.createElement("canvas");
    const maxSide = 480;
    const scale = Math.min(1, maxSide / Math.max(cropW, cropH));

    canvas.width = Math.max(160, Math.round(cropW * scale));
    canvas.height = Math.max(160, Math.round(cropH * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) return canvas;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      video,
      x,
      y,
      cropW,
      cropH,
      0,
      0,
      canvas.width,
      canvas.height
    );

    return canvas;
  };

  // ภาพเต็มจากกล้อง ใช้สำหรับเก็บหลักฐาน/ประวัติ ไม่ใช้แทน face descriptor
  const captureFullFrameImage = (video, maxSide = 720, quality = 0.74) => {
    const vw = Math.max(1, video.videoWidth || 640);
    const vh = Math.max(1, video.videoHeight || 480);
    const scale = Math.min(1, maxSide / Math.max(vw, vh));

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(vw * scale));
    canvas.height = Math.max(1, Math.round(vh * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) return "";

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL("image/jpeg", quality);
  };

  // fingerprint เบา ๆ จากภาพช่วงหัว-ไหล่
  // ใช้เป็นคะแนนเสริมเท่านั้น เพราะเสื้อ/ทรงผมเปลี่ยนได้
  const buildAppearanceFingerprint = (canvas) => {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;

    const size = 24;
    const work = document.createElement("canvas");
    work.width = size;
    work.height = size;

    const wctx = work.getContext("2d", { willReadFrequently: true });
    if (!wctx) return null;

    wctx.drawImage(canvas, 0, 0, size, size);
    const data = wctx.getImageData(0, 0, size, size).data;

    const hHist = new Array(12).fill(0);
    const sHist = new Array(6).fill(0);
    const vHist = new Array(6).fill(0);
    const luminance = [];

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const d = max - min;

      let h = 0;
      if (d !== 0) {
        if (max === r) h = ((g - b) / d) % 6;
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h /= 6;
        if (h < 0) h += 1;
      }

      const s = max === 0 ? 0 : d / max;
      const v = max;

      hHist[Math.min(11, Math.floor(h * 12))] += 1;
      sHist[Math.min(5, Math.floor(s * 6))] += 1;
      vHist[Math.min(5, Math.floor(v * 6))] += 1;

      luminance.push(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const total = size * size;
    const histogram = [
      ...hHist.map((v) => v / total),
      ...sHist.map((v) => v / total),
      ...vHist.map((v) => v / total),
    ];

    // 8x8 coarse structure
    const thumbnail = [];
    for (let gy = 0; gy < 8; gy++) {
      for (let gx = 0; gx < 8; gx++) {
        let sum = 0;
        let count = 0;

        for (let yy = gy * 3; yy < Math.min((gy + 1) * 3, size); yy++) {
          for (let xx = gx * 3; xx < Math.min((gx + 1) * 3, size); xx++) {
            sum += luminance[yy * size + xx];
            count += 1;
          }
        }

        thumbnail.push(sum / Math.max(1, count));
      }
    }

    return { histogram, thumbnail };
  };

  const appearanceSimilarity = (a, b) => {
    if (!a || !b) return 0;

    const ah = a.histogram || [];
    const bh = b.histogram || [];
    const at = a.thumbnail || [];
    const bt = b.thumbnail || [];

    if (ah.length !== bh.length || at.length !== bt.length) return 0;

    const histogramL1 =
      ah.reduce((sum, value, i) => sum + Math.abs(value - bh[i]), 0);

    const histogramScore = Math.max(0, 1 - histogramL1 / 2);

    const mse =
      at.reduce((sum, value, i) => sum + (value - bt[i]) ** 2, 0) /
      Math.max(1, at.length);

    const thumbnailScore = Math.max(0, 1 - Math.sqrt(mse) * 3);

    return Math.max(
      0,
      Math.min(1, histogramScore * 0.55 + thumbnailScore * 0.45)
    );
  };

  const captureFaceSnapshotData = (video, box) => {
    const faceCanvas = captureFaceImage(video, box);
    const appearanceCanvas = captureAppearanceImage(video, box);

    return {
      canvas: faceCanvas,
      image: canvasToCompressedJpeg(faceCanvas, 512, 0.84),
      fullImage: canvasToCompressedJpeg(appearanceCanvas, 480, 0.72),
      appearanceFingerprint: buildAppearanceFingerprint(appearanceCanvas),
    };
  };


  // Image quality assessment functions
  const assessImageQuality = (canvas) => {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Calculate blur using Laplacian variance
    let totalVariance = 0;
    let pixelCount = 0;
    
    for (let y = 1; y < canvas.height - 1; y++) {
      for (let x = 1; x < canvas.width - 1; x++) {
        const idx = (y * canvas.width + x) * 4;
        const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        
        const left = (data[idx - 4] + data[idx - 3] + data[idx - 2]) / 3;
        const right = (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3;
        const top = (data[idx - canvas.width * 4] + data[idx - canvas.width * 4 + 1] + data[idx - canvas.width * 4 + 2]) / 3;
        const bottom = (data[idx + canvas.width * 4] + data[idx + canvas.width * 4 + 1] + data[idx + canvas.width * 4 + 2]) / 3;
        
        const laplacian = -4 * gray + left + right + top + bottom;
        totalVariance += laplacian * laplacian;
        pixelCount++;
      }
    }
    
    const blurScore = totalVariance / pixelCount;
    
    // Calculate brightness
    let totalBrightness = 0;
    for (let i = 0; i < data.length; i += 4) {
      totalBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
    }
    const avgBrightness = totalBrightness / (data.length / 4);
    
    // Calculate contrast
    let totalSquared = 0;
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      totalSquared += Math.pow(brightness - avgBrightness, 2);
    }
    const contrast = Math.sqrt(totalSquared / (data.length / 4));
    
    // Calculate face aspect ratio check
    const aspectRatio = canvas.width / Math.max(1, canvas.height);
    const goodAspectRatio = aspectRatio >= 0.7 && aspectRatio <= 1.4;
    
    // Calculate sharpness using edge detection
    let edgeCount = 0;
    for (let y = 1; y < canvas.height - 1; y++) {
      for (let x = 1; x < canvas.width - 1; x++) {
        const idx = (y * canvas.width + x) * 4;
        const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        const rightGray = (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3;
        const bottomGray = (data[idx + canvas.width * 4] + data[idx + canvas.width * 4 + 1] + data[idx + canvas.width * 4 + 2]) / 3;
        
        const edgeStrength = Math.abs(gray - rightGray) + Math.abs(gray - bottomGray);
        if (edgeStrength > 30) edgeCount++;
      }
    }
    const edgeRatio = edgeCount / pixelCount;
    
    return {
      blurScore,
      brightness: avgBrightness,
      contrast,
      aspectRatio,
      edgeRatio,
      isBlurry: blurScore < 150,
      isTooDark: avgBrightness < 60,
      isTooBright: avgBrightness > 190,
      isLowContrast: contrast < 35,
      isBadAspectRatio: !goodAspectRatio,
      isLowSharpness: edgeRatio < 0.15
    };
  };

  // Temporal smoothing for stable detection boxes
  const smoothBox = (trackId, newBox, width, height) => {
    const history = detectionHistoryRef.current[trackId] || [];
    history.push(newBox);
    
    // Keep only last 5 frames
    if (history.length > 5) {
      history.shift();
    }
    
    detectionHistoryRef.current[trackId] = history;
    
    // If we have enough history, calculate smoothed box
    if (history.length >= 3) {
      const avgX = history.reduce((sum, box) => sum + box.originX, 0) / history.length;
      const avgY = history.reduce((sum, box) => sum + box.originY, 0) / history.length;
      const avgW = history.reduce((sum, box) => sum + box.width, 0) / history.length;
      const avgH = history.reduce((sum, box) => sum + box.height, 0) / history.length;
      
      // Blend current detection with smoothed version (70% current, 30% smoothed)
      const smoothedBox = {
        originX: newBox.originX * 0.7 + avgX * 0.3,
        originY: newBox.originY * 0.7 + avgY * 0.3,
        width: newBox.width * 0.7 + avgW * 0.3,
        height: newBox.height * 0.7 + avgH * 0.3
      };
      
      return smoothedBox;
    }
    
    return newBox;
  };

  // Face API สำหรับสร้าง descriptor จากภาพที่ถูกครอป
  // เพิ่มขนาดภาพเล็ก ๆ ก่อนตรวจ เพราะกรอบจากกล้องบางครั้งมีขนาดใบหน้าเล็ก
  // ทำให้ TinyFaceDetector หาใบหน้าซ้ำไม่เจอ ทั้งที่ MediaPipe หาเจอแล้ว
  const detectWithMultipleModels = async (source) => {
    try {
      let input = source;

      if (source instanceof HTMLCanvasElement) {
        const minSide = 240;
        const sw = Math.max(1, source.width);
        const sh = Math.max(1, source.height);
        const scale = Math.max(1, minSide / Math.min(sw, sh));

        if (scale > 1.05) {
          const upscaled = document.createElement("canvas");
          upscaled.width = Math.min(960, Math.round(sw * scale));
          upscaled.height = Math.min(960, Math.round(sh * scale));

          const ctx = upscaled.getContext("2d");
          if (ctx) {
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            ctx.drawImage(
              source,
              0,
              0,
              upscaled.width,
              upscaled.height
            );
            input = upscaled;
          }
        }
      }

      // รอบแรก: เร็ว
      const tiny = await faceapi
        .detectSingleFace(
          input,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 416,
            scoreThreshold: 0.03,
          })
        )
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (tiny) {
        return {
          ...tiny,
          model: "tiny",
        };
      }

      // รอบสำรอง: ช้ากว่า แต่ช่วยกรณีภาพชัด/มุมหน้าไม่เหมาะกับ Tiny
      const ssd = await faceapi
        .detectSingleFace(
          input,
          new faceapi.SsdMobilenetv1Options({
            minConfidence: 0.15,
          })
        )
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (ssd) {
        return {
          ...ssd,
          model: "ssd",
        };
      }

      return null;
    } catch (error) {
      console.warn("Face descriptor detection error:", error);
      return null;
    }
  };

  // Enhanced recognition using saved face images
  // เปรียบเทียบ descriptor ของคนปัจจุบันกับทุกมุมที่บันทึกไว้
  // คืนผลที่ "เข้มงวด" เพื่อหลีกเลี่ยงการเอาคนอื่นมาแสดงชื่อผิด
  // ค้นหาจาก "ทุก descriptor ของทุกภาพ" ในฐานข้อมูลที่ cache ไว้ใน RAM
  // localStorage ใช้เป็นที่เก็บถาวรเท่านั้น ไม่ถูกอ่านซ้ำระหว่าง Scan
  const recognizeWithSavedImages = async (currentDetection, people) => {
    if (!currentDetection || !people.length) return null;

    const query = currentDetection.descriptor;
    const index = faceSearchIndexRef.current;

    if (!index.length) return null;

    const byPerson = new Map();

    for (const item of index) {
      const distance = faceapi.euclideanDistance(query, item.vector);
      const current = byPerson.get(item.personId);

      const candidate = {
        distance,
        image: item.image,
        angle: item.angle,
        source: item.source,
      };

      if (!current) {
        byPerson.set(item.personId, {
          person: item.person,
          matches: [candidate],
        });
      } else {
        current.matches.push(candidate);
      }
    }

    const scores = [];

    for (const group of byPerson.values()) {
      group.matches.sort((a, b) => a.distance - b.distance);
      const best = group.matches[0];
      const second = group.matches[1]?.distance ?? best.distance;

      // Face descriptor = หลักฐานหลัก
      const faceScore = best.distance * 0.75 + second * 0.25;

      // Appearance = หลักฐานเสริมจากช่วงหัว-ไหล่/เสื้อผ้า/ทรงผม
      // ไม่ให้คะแนนนี้ชนะ face descriptor เด็ดขาด
      let appearanceScore = 0;
      if (
        currentDetection.appearanceFingerprint &&
        Array.isArray(group.person.appearanceImages)
      ) {
        for (const ref of group.person.appearanceImages) {
          appearanceScore = Math.max(
            appearanceScore,
            appearanceSimilarity(
              currentDetection.appearanceFingerprint,
              ref.fingerprint
            )
          );
        }
      }

      // 90% face + 10% appearance
      // เสื้อผ้าเปลี่ยนได้ จึงไม่ควรเอามาเป็นตัวตัดสินหลัก
      const aggregateScore =
        faceScore * 0.90 + (1 - appearanceScore) * 0.10;

      scores.push({
        person: group.person,
        best: best.distance,
        second,
        faceScore,
        aggregateScore,
        appearanceScore,
        matchedImage: best.image,
        matchedAngle: best.angle,
        source: best.source,
        support: group.matches.filter((m) => m.distance <= 0.50).length,
      });
    }

    scores.sort((a, b) => {
      if (a.aggregateScore !== b.aggregateScore) {
        return a.aggregateScore - b.aggregateScore;
      }
      return a.best - b.best;
    });

    const winner = scores[0];
    const runner = scores[1] ?? null;
    const margin = runner ? runner.aggregateScore - winner.aggregateScore : Infinity;

    return {
      person: winner.person,
      distance: winner.best,
      secondDistance: winner.second,
      runnerUpDistance: runner?.best ?? Infinity,
      margin,
      support: winner.support,
      matchedImage: winner.matchedImage,
      matchedAngle: winner.matchedAngle,
      source: winner.source,

      // ไม่มั่นใจ = ไม่บอกชื่อ
      // ถ้าสองคนมีคะแนนหน้าใกล้กันมาก ให้หยุดก่อน ไม่เดาชื่อ
      ambiguous:
        !!runner &&
        winner.best <= 0.48 &&
        margin < 0.065,

      confidentEnough:
        winner.best <= 0.42 &&
        winner.faceScore <= 0.44 &&
        margin >= 0.065 &&
        winner.support >= 1,
    };
  };


  const updateName = (id, newName) => {
    const updated = savedFaces.map(f => f.id === id ? { ...f, name: newName } : f);
    savedFacesRef.current = updated;
    setSavedFaces(updated);
    safeSetLocalStorage(FACE_DB_KEY, updated);
  };

  const clearSavedFaces = () => {
    if(confirm("ต้องการล้างข้อมูลใบหน้าที่บันทึกไว้ทั้งหมดหรือไม่?")) {
      localStorage.removeItem(FACE_DB_KEY);
      localStorage.removeItem(FACE_SCAN_HISTORY_KEY);
      savedFacesRef.current = [];
      scanHistoryRef.current = [];
      setSavedFaces([]);
      updateFaceMatcher([]);
    }
  };

  async function refreshCameraDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      setCameraDevices(cams);

      const activeTrack = streamRef.current?.getVideoTracks?.()[0];
      const activeId = activeTrack?.getSettings?.()?.deviceId;

      if (activeId) setCameraId(activeId);
      return cams;
    } catch (err) {
      console.warn("Camera enumeration:", err);
      return [];
    }
  }

  async function startCamera(requestedDeviceId = "") {
    setCameraError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("เบราว์เซอร์นี้ไม่รองรับการใช้งานกล้อง");
      setScanMessage("ไม่รองรับกล้อง");
      return;
    }

    if (!detectorRef.current) {
      setCameraError("โมเดล AI ยังไม่พร้อม กรุณารอสักครู่แล้วลองใหม่");
      return;
    }

    try {
      // Stop current stream before opening another camera.
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      const video = videoRef.current;
      if (!video) throw new Error("VIDEO_ELEMENT_NOT_READY");

      const videoConstraints = requestedDeviceId
        ? {
            deviceId: { exact: requestedDeviceId },
            width: { ideal: 1920, min: 640 },
            height: { ideal: 1080, min: 360 },
            frameRate: { ideal: 30 },
          }
        : {
            facingMode: { ideal: facingModeRef.current },
            width: { ideal: 1920, min: 640 },
            height: { ideal: 1080, min: 360 },
            frameRate: { ideal: 30 },
          };

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      streamRef.current = stream;

      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings?.() || {};

      if (settings.facingMode) {
        facingModeRef.current = settings.facingMode;
      }

      // กล้องหน้า = กระจก / กล้องหลัง = ภาพปกติ
      setCameraMirrored(facingModeRef.current === "user");

      setCameraId(settings.deviceId || requestedDeviceId || "");
      setCameraLabel(
        track?.label ||
          (facingModeRef.current === "user" ? "กล้องหน้า" : "กล้องหลัง")
      );
      setCameraQuality(
        settings.width && settings.height
          ? `${settings.width}x${settings.height}`
          : "กล้องทำงานอยู่"
      );

      video.srcObject = stream;

      if (video.readyState < 1) {
        await new Promise((resolve) => {
          let finished = false;
          const done = () => {
            if (finished) return;
            finished = true;
            video.removeEventListener("loadedmetadata", done);
            resolve();
          };
          video.addEventListener("loadedmetadata", done, { once: true });
          setTimeout(done, 1500);
        });
      }

      await video.play();

      cameraOnRef.current = true;
      setCameraOn(true);
      setCameraError("");
      setScanMessage("กำลังวิเคราะห์ใบหน้า...");

      // Permission has succeeded at this point. Enumeration failure must not
      // be reported as "permission denied".
      await refreshCameraDevices();

      cancelAnimationFrame(animationRef.current);
      animationRef.current = requestAnimationFrame(detectFaces);
    } catch (err) {
      console.error("Camera Error:", err);

      cameraOnRef.current = false;
      setCameraOn(false);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      const errorName = err?.name || "";

      if (
        errorName === "NotAllowedError" ||
        errorName === "PermissionDeniedError"
      ) {
        setCameraError("เบราว์เซอร์ยังไม่ได้รับอนุญาตให้ใช้กล้อง");
        setScanMessage("กรุณาอนุญาตการใช้กล้อง");
      } else if (
        errorName === "NotFoundError" ||
        errorName === "DevicesNotFoundError"
      ) {
        setCameraError("ไม่พบกล้องในอุปกรณ์");
        setScanMessage("ไม่พบกล้อง");
      } else if (
        errorName === "NotReadableError" ||
        errorName === "TrackStartError"
      ) {
        setCameraError("กล้องกำลังถูกใช้งานโดยแอปอื่น");
        setScanMessage("กล้องไม่พร้อมใช้งาน");
      } else if (err?.message === "VIDEO_ELEMENT_NOT_READY") {
        setCameraError("หน้ากล้องยังไม่พร้อม กรุณาลองอีกครั้ง");
        setScanMessage("กล้องยังไม่พร้อม");
      } else {
        // Never show a fake permission warning for a JS/programming error.
        setCameraError(`เปิดกล้องไม่สำเร็จ (${errorName || "UnknownError"})`);
        setScanMessage("ลองเปิดกล้องอีกครั้ง");
      }
    }
  }

  // ปิดกล้องแบบรวมศูนย์: ใช้ทั้งตอนผู้ใช้กดปิด และหลังบันทึกสำเร็จ
  const stopCamera = (message = "กล้องปิด • พร้อมเริ่มสแกนใหม่") => {
    cancelAnimationFrame(animationRef.current);
    animationRef.current = null;

    if (enrollmentLoopRef.current) {
      clearInterval(enrollmentLoopRef.current);
      enrollmentLoopRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    cameraOnRef.current = false;
    setCameraOn(false);
    onCameraStateChange?.(false);
    setFaceCount(0);
    setDetectedFaces([]);
    setFaceGeometry(null);
    lastMeshResultRef.current = null;
    tracksRef.current = [];
    lastFrameTimestampRef.current = 0;
    lastDetectorRunRef.current = 0;
    nextDetectionAtRef.current = 0;
    setCameraQuality("");
    setCameraError("");
    setScanMessage(message);

    const ctx = canvasRef.current?.getContext("2d");
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  async function switchCamera() {
    // One tap only: toggle front <-> rear by facingMode.
    // Do not enumerate devices here; that caused cycling through multiple cameras.
    if (!cameraOnRef.current) return;
    try {
      const nextFacing = facingModeRef.current === "user" ? "environment" : "user";
      facingModeRef.current = nextFacing;
      await startCamera("");
    } catch (err) {
      console.error("Switch camera:", err);
      setCameraError("เปลี่ยนกล้องไม่สำเร็จ");
    }
  }

  async function toggleCamera() {
    // ไม่อนุญาตให้เปิดกล้องจากปุ่มทั่วไปถ้ายังไม่ได้กรอกชื่อ
    if (!cameraOnRef.current && !selectedEmployee?.id) {
      setEnrollMessage("กรุณาเลือกพนักงานก่อนเปิดกล้อง");
      return;
    }

    if (cameraOnRef.current) {
      stopCamera();
      setEnrollMessage("กล้องปิดแล้ว • กดเริ่มบันทึกเพื่อเปิดใหม่");
      return;
    }

    await startCamera(cameraId || "");
  }

  const ENROLL_STAGES = [
    { key: "front", label: "หันหน้าตรง" },
    { key: "left", label: "หันหน้าไปทางซ้าย" },
    { key: "right", label: "หันหน้าไปทางขวา" },
  ];

  const avgPoint = (points) => ({
    x: points.reduce((sum, p) => sum + p.x, 0) / points.length,
    y: points.reduce((sum, p) => sum + p.y, 0) / points.length,
  });

  const getFaceYaw = (landmarks) => {
    if (!landmarks) return 0;
    const left = landmarks.getLeftEye?.() || [];
    const right = landmarks.getRightEye?.() || [];
    const nose = landmarks.getNose?.() || [];
    if (!left.length || !right.length || !nose.length) return 0;
    const leftEye = avgPoint(left);
    const rightEye = avgPoint(right);
    const noseTip = nose[6] || nose[Math.floor(nose.length / 2)];
    if (!noseTip) return 0;
    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeDist = Math.max(
      1,
      Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y)
    );
    return (noseTip.x - eyeMidX) / eyeDist;
  };

  // ใช้เป็นข้อมูลประกอบเท่านั้น ไม่ใช้เป็น hard gate
  // เพราะมุมเงย/ก้มอาจคลาดเคลื่อนตามระยะกล้องและอัตราส่วนหน้าจอ
  const getFacePitch = (landmarks) => {
    const leftEye = avgPoint(landmarks.getLeftEye());
    const rightEye = avgPoint(landmarks.getRightEye());
    const mouth = avgPoint(landmarks.getMouth());
    const nose = landmarks.getNose();
    const noseTip = nose[6] || nose[Math.floor(nose.length / 2)];

    const eyeMidY = (leftEye.y + rightEye.y) / 2;
    const eyeToMouth = Math.max(1, mouth.y - eyeMidY);

    return (noseTip.y - eyeMidY) / eyeToMouth - 0.50;
  };

  const normalizeEnrollmentLandmarks = (landmarks, box) => {
    const width = Math.max(1, box?.width || 1);
    const height = Math.max(1, box?.height || 1);
    const left = box?.x || 0;
    const top = box?.y || 0;

    const points =
      landmarks?.positions ||
      landmarks?.map?.((p) => ({ x: p.x, y: p.y })) ||
      [];

    return points.map((p) => ({
      x: Number(((p.x - left) / width).toFixed(5)),
      y: Number(((p.y - top) / height).toFixed(5)),
    }));
  };

  const findDuplicatePerson = (descriptor) => {
    let bestPerson = null;
    let bestDistance = Infinity;

    for (const person of savedFacesRef.current) {
      const descriptors =
        Array.isArray(person.descriptors) && person.descriptors.length
          ? person.descriptors
          : person.descriptor
            ? [person.descriptor]
            : [];

      for (const stored of descriptors) {
        const d = faceapi.euclideanDistance(
          new Float32Array(descriptor),
          new Float32Array(stored)
        );

        if (d < bestDistance) {
          bestDistance = d;
          bestPerson = person;
        }
      }
    }

    return { person: bestPerson, distance: bestDistance };
  };

  const getEnrollmentDepth = () => {
    const landmarks = lastMeshResultRef.current?.faceLandmarks?.[0] || lastMeshResultRef.current?.landmarks?.[0];
    return landmarks ? buildDepthProfile(landmarks) : null;
  };

  const resetEnrollment = () => {
    if (enrollmentLoopRef.current) {
      clearInterval(enrollmentLoopRef.current);
      enrollmentLoopRef.current = null;
    }
    enrollmentRef.current = {
      active: false,
      stage: 0,
      stable: 0,
      samples: [],
      firstTurnSign: null,
      lastAt: 0,
      primaryFrontImage: null,
    };

    enrollmentBusyRef.current = false;
    setEnrollStage(0);
    setEnrollStable(0);
    setEnrollCountdown(0);
    setEnrollMessage("");
    setEnrollmentStarted(false);
  };

  // เปิดกล้องอย่างเดียว: ยังไม่เริ่ม Enrollment และห้ามเก็บ/บันทึกข้อมูล
  const openEnrollmentCamera = async () => {
    if (!selectedEmployee?.id) {
      setEnrollMessage("กรุณาเลือกพนักงานจากรายชื่อก่อนเปิดกล้อง");
      return;
    }


    try {
      resetEnrollment();
      lastMeshResultRef.current = null;
      meshLastTimestampRef.current = 0;
      modeRef.current = "enroll";
      setMode("enroll");
      setEnrollMessage("กล้องพร้อมแล้ว • กด “เริ่มบันทึก” เมื่อต้องการเริ่มเก็บใบหน้า");
      await startCamera(cameraId || "");

      if (!cameraOnRef.current) {
        setEnrollMessage(cameraError || "เปิดกล้องไม่สำเร็จ • กรุณาลองอีกครั้ง");
      }
    } catch (error) {
      console.error("Open enrollment camera:", error);
      setEnrollMessage("เปิดกล้องไม่ได้ • กรุณาลองอีกครั้ง");
    }
  };

  // เริ่มบันทึกจริงเฉพาะเมื่อผู้ใช้กดปุ่มนี้
  const startEnrollment = async () => {
    if (!selectedEmployee?.id) {
      setEnrollMessage("กรุณาเลือกพนักงานก่อนเริ่มบันทึก");
      return;
    }

    if (!cameraOnRef.current) {
      setEnrollMessage("กรุณาเปิดกล้องก่อน แล้วกด “เริ่มบันทึก”");
      return;
    }

    try {
      if (enrollmentLoopRef.current) {
        clearInterval(enrollmentLoopRef.current);
        enrollmentLoopRef.current = null;
      }

      // ล้างเฉพาะ state ของรอบบันทึก แต่ไม่ปิดกล้อง
      resetEnrollment();
      lastMeshResultRef.current = null;
      meshLastTimestampRef.current = 0;
      modeRef.current = "enroll";
      setMode("enroll");

      // สำคัญ: ตั้ง active หลังจากกดปุ่มนี้เท่านั้น
      localStorage.removeItem("face_enrollment_draft");
      enrollmentRef.current.active = true;
      enrollmentRef.current.stage = 0;
      enrollmentRef.current.samples = [];
      enrollmentRef.current.firstTurnSign = null;
      enrollmentRef.current.targetLock = null;
      enrollmentRef.current.lastAt = 0;

      setEnrollmentStarted(true);
      setEnrollStage(0);
      setEnrollStable(0);
      setEnrollCountdown(0);
      setEnrollMessage("🔴 เริ่มบันทึก • มองหน้าตรงเข้ากล้อง");
      setScanMessage("บันทึกใบหน้า • หน้าตรง");

      await processEnrollmentFrame(videoRef.current, performance.now());

      if (cameraOnRef.current && modeRef.current === "enroll" && enrollmentRef.current.active) {
        enrollmentLoopRef.current = setInterval(() => {
          if (cameraOnRef.current && modeRef.current === "enroll" && enrollmentRef.current.active) {
            processEnrollmentFrame(videoRef.current, performance.now());
          }
        }, 180);
      }
    } catch (error) {
      console.error("Start enrollment:", error);
      setEnrollmentStarted(false);
      enrollmentRef.current.active = false;
      modeRef.current = "enroll";
      setEnrollMessage("เริ่มระบบบันทึกไม่ได้ • ลองกดเริ่มบันทึกอีกครั้ง");
    }
  };

  const cancelEnrollment = () => {
    resetEnrollment();
    stopCamera("ยกเลิกการลงทะเบียน • กล้องปิดแล้ว");
    modeRef.current = "scan";
    setMode("scan");
    setScanMessage("พร้อมสแกนหลายคนพร้อมกัน");
    setScanActivity("🔍 กำลังรอใบหน้า...");
  };

  const capturePrimaryFrontImage = async () => {
    const state = enrollmentRef.current;
    const video = videoRef.current;

    if (!state.active || state.samples.length !== ENROLL_STAGES.length) {
      setEnrollMessage("กรุณาบันทึกครบทั้ง 3 มุมก่อน");
      return;
    }
    if (!video || video.readyState < 2 || !cameraOnRef.current) {
      setEnrollMessage("กล้องยังไม่พร้อม");
      return;
    }

    try {
      const result = await detectWithMultipleModels(video);
      if (!result) {
        setEnrollMessage("ยังตรวจไม่พบใบหน้า • มองตรงเข้ากล้องก่อน");
        return;
      }
      const yaw = getFaceYaw(result.landmarks);
      if (!Number.isFinite(yaw) || Math.abs(yaw) > 0.35) {
        setEnrollMessage("กรุณากลับมาหน้าตรงและอยู่นิ่ง");
        return;
      }
      const box = result.detection.box;
      if (result.detection.score < 0.10 || box.width < 60 || box.height < 60) {
        setEnrollMessage("ใบหน้ายังเล็กหรือไม่ชัด • ขยับเข้าใกล้กล้องอีกนิด");
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("ไม่สามารถสร้างภาพจากกล้องได้");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      state.primaryFrontImage = canvas.toDataURL("image/jpeg", 0.92);
      safeSetLocalStorage("face_enrollment_draft", {
        employeeId: selectedEmployee?.id || null,
        name: enrollName.trim(),
        samples: state.samples,
        primaryFrontImage: state.primaryFrontImage,
        stage: ENROLL_STAGES.length,
        updatedAt: new Date().toISOString(),
      });
      setEnrollStage(ENROLL_STAGES.length + 1);
      setEnrollStable(0);
      setEnrollMessage("บันทึกภาพหน้าตรงสำเร็จ ✓ • กำลังบันทึกโปรไฟล์");
      await finishEnrollment();
    } catch (error) {
      console.warn("Capture primary front image:", error);
      setEnrollMessage("บันทึกภาพไม่สำเร็จ • ลองใหม่อีกครั้ง");
    }
  };

  const saveEnrollmentToApi = async (person) => {
    if (!selectedEmployee?.id) {
      throw new Error("ยังไม่ได้เลือกพนักงาน");
    }

    // สำคัญ: API ใช้ id ของพนักงานเป็น key จริง และไม่ส่ง name ไปเป็นตัวอ้างอิง
    const payload = {
      id: String(selectedEmployee.id),
      createdAt: person.createdAt,
      depthProfile: person.depthProfile || null,
      depthProfiles: Array.isArray(person.depthProfiles) ? person.depthProfiles : [],
      descriptors: Array.isArray(person.descriptors) ? person.descriptors : [],
      face3DProfile: person.face3DProfile || null,
      faceAngles: Array.isArray(person.faceAngles) ? person.faceAngles : [],
      frontImage: person.frontImage || "",
      landmarks: Array.isArray(person.landmarks) ? person.landmarks : [],
      models: Array.isArray(person.models) ? person.models : [],
      version: person.version || "5.0",
    };

    if (!payload.descriptors.length) {
      throw new Error("ไม่พบ Face Descriptor");
    }

    const invalidDescriptor = payload.descriptors.find(
      (descriptor) => !Array.isArray(descriptor) || descriptor.length !== 128
    );

    if (invalidDescriptor) {
      throw new Error("Face Descriptor ต้องมี 128 ค่า");
    }

    const response = await fetch(FACE_ENROLL_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    const responseText = await response.text();
    let result = null;
    try {
      result = responseText ? JSON.parse(responseText) : null;
    } catch {
      result = null;
    }

    if (!response.ok) {
      const message = result?.message || responseText || `HTTP ${response.status}`;
      throw new Error(message);
    }

    return result || { success: true };
  };

  const finishEnrollment = async () => {
    const state = enrollmentRef.current;

    if (!state.active || state.samples.length !== 3) return;
    if (!state.primaryFrontImage) {
      setEnrollMessage("กลับมาหน้าตรงแล้วอยู่นิ่ง เพื่อบันทึกภาพหลักอัตโนมัติ");
      return;
    }

    // Strong duplicate protection: compare the whole 3-angle enrollment,
    // not only one frame. A person must not be stored twice under a new name.
    let duplicatePerson = null;
    let duplicateScore = Infinity;
    for (const person of savedFacesRef.current) {
      // ถ้าเป็นพนักงานคนเดียวกัน ให้ถือเป็นการลงทะเบียนใหม่/อัปเดต ไม่ใช่ duplicate คนอื่น
      if (String(person?.id || "") === String(selectedEmployee.id)) continue;

      const stored = Array.isArray(person.descriptors) && person.descriptors.length
        ? person.descriptors
        : person.descriptor ? [person.descriptor] : [];
      if (!stored.length) continue;
      const distances = state.samples.map((sample) => {
        let best = Infinity;
        for (const d of stored) {
          best = Math.min(best, faceapi.euclideanDistance(
            new Float32Array(sample.descriptor), new Float32Array(d)
          ));
        }
        return best;
      });
      distances.sort((a, b) => a - b);
      const score = distances.length >= 2
        ? distances[0] * 0.6 + distances[1] * 0.4
        : distances[0];
      if (score < duplicateScore) {
        duplicateScore = score;
        duplicatePerson = person;
      }
    }

    if (duplicatePerson && duplicateScore < 0.42) {
      state.active = false;
      if (enrollmentLoopRef.current) { clearInterval(enrollmentLoopRef.current); enrollmentLoopRef.current = null; }
      setEnrollMessage(`บุคคลนี้มีอยู่แล้ว: ${duplicatePerson.name} • ไม่บันทึกซ้ำ`);
      setScanMessage(`พบข้อมูลเดิม • ${duplicatePerson.name}`);
      return;
    }

    // ภาพหลักต้องมาจากปุ่ม “บันทึกภาพหน้าตรง” เท่านั้น
    const frontFaceImage = state.primaryFrontImage;

    // เก็บเฉพาะข้อมูลที่จำเป็นสำหรับการแสกนหน้าแม่นยำ
    const person = {
      // ใช้ employees.id เป็น ID เดียวกับที่ส่งเข้า Go API
      id: String(selectedEmployee.id),
      // name มีไว้แสดงผลบน frontend เท่านั้น ไม่ใช้เป็น key ของ API/DB
      name: enrollName.trim(),

      // ภาพอ้างอิงหลัก = หน้าตรงเท่านั้น (เพื่อความแม่นยำสูงสุด)
      frontImage: frontFaceImage || state.samples.find((s) => s.stage === "front")?.image || state.samples[0].image,

      // Descriptor แยกตามมุม (เฉพาะ 3 มุมหลัก: front, left, right)
      descriptors: state.samples
        .filter((sample) => ["front", "left", "right"].includes(sample.stage))
        .map((sample) => Array.from(sample.descriptor)),
      
      faceAngles: state.samples
        .filter((sample) => ["front", "left", "right"].includes(sample.stage))
        .map((sample) => sample.stage),
      
      models: state.samples
        .filter((sample) => ["front", "left", "right"].includes(sample.stage))
        .map((sample) => sample.model || "tiny"),

      // เก็บ landmarks เฉพาะ 3 มุมหลักเพื่อใช้เปรียบเทียบมุม
      landmarks: state.samples
        .filter((sample) => ["front", "left", "right"].includes(sample.stage))
        .map((sample) => ({
          stage: sample.stage,
          normalized: sample.landmarks || null,
          yaw: sample.yaw || 0,
          boxRatio: sample.boxRatio || null,
        })),

      // เก็บ depth/geometry เฉพาะ 3 มุมหลักเพื่อเปรียบเทียบ 3D shape
      depthProfiles: state.samples
        .filter((sample) => ["front", "left", "right"].includes(sample.stage))
        .map((sample) => ({
          stage: sample.stage,
          profile: sample.depthProfile || null,
        })),

      // Hybrid 3D Face Identity Profile
      // ไม่สร้าง "หัว 3D ปลอม" แต่เก็บโครงสร้าง 3D จาก Face Landmarker
      // ของแต่ละมุมจริง + index ของ descriptor เพื่อให้ Scan เลือกตัวอย่าง
      // ที่ใกล้กับมุมหน้าปัจจุบันได้โดยตรง
      face3DProfile: {
        version: "1.0",
        source: "mediapipe-face-landmarker",
        descriptorDimension: 128,
        anchors: state.samples
          .filter((sample) => ["front", "left", "right"].includes(sample.stage))
          .map((sample, descriptorIndex) => ({
            stage: sample.stage,
            descriptorIndex,
            yaw: Number((Number(sample.yaw) || 0).toFixed(5)),
            pitch: Number((Number(sample.pitch) || 0).toFixed(5)),
            landmarks: Array.isArray(sample.landmarks) ? sample.landmarks : null,
            depthProfile: sample.depthProfile || null,
            boxRatio: sample.boxRatio || null,
          })),
        createdAt: new Date().toISOString(),
      },

      // Backward compatibility: หน้าตรงเป็น profile หลัก
      depthProfile:
        state.samples.find((s) => s.stage === "front")?.depthProfile ||
        state.samples[0]?.depthProfile ||
        null,

      createdAt: new Date().toISOString(),
      version: "5.0", // เพิ่ม Hybrid 3D Face Identity Profile
    };

    // 1) ส่งข้อมูล enrollment ไป Go API ก่อน
    // ถ้า API ไม่สำเร็จ จะยังไม่เขียน saved_faces เพื่อไม่ให้ frontend กับ DB ไม่ตรงกัน
    try {
      setEnrollMessage(`กำลังส่งข้อมูล ${enrollName.trim()} ไปยังเซิร์ฟเวอร์...`);
      await saveEnrollmentToApi(person);
    } catch (error) {
      console.error("❌ Face enrollment API:", error);
      state.active = false;
      if (enrollmentLoopRef.current) {
        clearInterval(enrollmentLoopRef.current);
        enrollmentLoopRef.current = null;
      }
      setEnrollmentStarted(false);
      setEnrollMessage(`บันทึกฐานข้อมูลไม่สำเร็จ • ${error?.message || "ลองใหม่อีกครั้ง"}`);
      return;
    }

    // 2) API สำเร็จแล้ว จึงอัปเดต LocalStorage โดยแทนข้อมูลของ employee.id เดิม
    const updated = [
      ...savedFacesRef.current.filter(
        (existing) => String(existing?.id || "") !== String(selectedEmployee.id)
      ),
      person,
    ];

    const storageOk = safeSetLocalStorage(FACE_DB_KEY, updated);

    if (!storageOk) {
      setEnrollMessage(
        "ส่งฐานข้อมูลสำเร็จ แต่บันทึก LocalStorage ไม่สำเร็จ • กรุณารีโหลดข้อมูลจากระบบภายหลัง"
      );
      console.error("❌ saved_faces was not persisted after API success");
      return;
    }

    // อัปเดต RAM หลัง LocalStorage สำเร็จ เพื่อให้ Scan ใช้ข้อมูลชุดเดียวกัน
    savedFacesRef.current = updated;
    setSavedFaces(updated);
    updateFaceMatcher(updated);

    // ลบร่างชั่วคราวเมื่อบันทึกจริงสำเร็จ
    try {
      localStorage.removeItem("face_enrollment_draft");
    } catch {}

    state.active = false;
    if (enrollmentLoopRef.current) { clearInterval(enrollmentLoopRef.current); enrollmentLoopRef.current = null; }
    setEnrollStage(ENROLL_STAGES.length + 1);
    setEnrollStable(0);
    setEnrollCountdown(0);
    setEnrollMessage("บันทึกโปรไฟล์สำเร็จ ✓");
    setScanMessage(`บันทึก ${person.name} สำเร็จ`);

    // บันทึกเสร็จแล้วต้องปิดกล้องทันที และกลับไปหน้า UI ลงทะเบียน
    stopCamera(`บันทึก ${person.name} สำเร็จ • กล้องปิดแล้ว`);
  };

  // ล็อกเป้าหมายแบบ "ตาคน": เมื่อเห็นใบหน้าแรกแล้ว จะจำ descriptor + ตำแหน่ง + เวลา
  // และยอมรับเฉพาะใบหน้าที่ต่อเนื่องกับเป้าหมายเดิม เพื่อไม่สลับไปจับคนอื่นกลางการลงทะเบียน
  const updateEnrollmentTargetLock = (result, box, timestamp) => {
    const state = enrollmentRef.current;
    const descriptor = result?.descriptor;
    if (!descriptor || !box) return { locked: false, distance: Infinity, iou: 0 };

    const center = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    if (!state.targetLock) {
      state.targetLock = {
        descriptor: Array.from(descriptor),
        box: { x: box.x, y: box.y, width: box.width, height: box.height },
        center,
        firstSeenAt: timestamp,
        lastSeenAt: timestamp,
        lostFrames: 0,
        hits: 1,
      };
      return { locked: true, distance: 0, iou: 1, justLocked: true };
    }

    const previous = state.targetLock;
    const distance = faceapi.euclideanDistance(
      new Float32Array(descriptor),
      new Float32Array(previous.descriptor)
    );
    const ix1 = Math.max(box.x, previous.box.x);
    const iy1 = Math.max(box.y, previous.box.y);
    const ix2 = Math.min(box.x + box.width, previous.box.x + previous.box.width);
    const iy2 = Math.min(box.y + box.height, previous.box.y + previous.box.height);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const intersection = iw * ih;
    const union = box.width * box.height + previous.box.width * previous.box.height - intersection;
    const iou = union > 0 ? intersection / union : 0;

    // Descriptor เป็นตัวจำ "คน" ส่วน IoU/ตำแหน่งเป็นตัวช่วยจำ "ตัวที่กำลังมองอยู่"
    // ใช้ hysteresis: คนเดิมสามารถขยับหัว/ขยับตำแหน่งได้ แต่คนใหม่จะไม่แย่ง target ง่าย ๆ
    // ตอนหันซ้าย/ขวา descriptor สามารถเปลี่ยนมากกว่าหน้าตรงได้
    // จึงห้ามใช้ threshold เดิม 0.48 เป็น hard gate ไม่เช่นนั้น
    // target เดิมจะหลุดทันทีที่ผู้ใช้หมุนหน้า
    const sameIdentity = distance < 0.70;
    const spatiallyRelated = iou > 0.08 || Math.hypot(
      center.x - previous.center.x, center.y - previous.center.y
    ) < Math.max(box.width, box.height) * 1.60;

    // ในช่วง enrollment ให้ "ตำแหน่งต่อเนื่อง + descriptor ยังอยู่ในช่วงสมเหตุผล"
    // เป็นตัวล็อกหลัก เพื่อให้การหันซ้าย/ขวาไม่ทำให้เป้าหมายหลุด
    const locked = sameIdentity && (
      spatiallyRelated ||
      timestamp - previous.lastSeenAt < 900
    );

    if (locked) {
      // อัปเดต template แบบช้า ๆ เพื่อให้ระบบจดจำใบหน้าเดียวกันได้ดีขึ้นขณะขยับ
      // แต่ไม่แทนที่ด้วย frame เดียว เพื่อป้องกัน descriptor drift
      const alpha = 0.12;
      previous.descriptor = previous.descriptor.map((v, i) =>
        v * (1 - alpha) + Number(descriptor[i] || 0) * alpha
      );
      previous.box = { x: box.x, y: box.y, width: box.width, height: box.height };
      previous.center = center;
      previous.lastSeenAt = timestamp;
      previous.lostFrames = 0;
      previous.hits += 1;
      return { locked: true, distance, iou, justLocked: false };
    }

    previous.lostFrames += 1;
    return { locked: false, distance, iou, justLocked: false };
  };

  const processEnrollmentFrame = async (video, timestamp) => {
    const state = enrollmentRef.current;

    if (
      !state.active ||
      modeRef.current !== "enroll" ||
      enrollmentBusyRef.current ||
      video.readyState < 2 ||
      timestamp - state.lastAt < 120
    ) return;

    enrollmentBusyRef.current = true;
    state.lastAt = timestamp;

    try {
      setEnrollMessage(`🔍 กำลังค้นหาใบหน้า • ${ENROLL_STAGES[state.stage]?.label || "หน้าตรง"}`);

      // Enrollment: Face-API เป็นตัวหลักสำหรับสร้าง descriptor เพราะต้องใช้
      // landmark + descriptor ต่อทันที ส่วน MediaPipe เป็นตัวช่วยเมื่อ Face-API
      // หาใบหน้าในภาพเต็มไม่เจอ โดยเฉพาะกรณีใบหน้าอยู่ไกล/แสงไม่ดี
      let result = null;

      try {
        result = await detectWithMultipleModels(video);
      } catch (faceApiError) {
        console.warn("Enrollment Face-API primary:", faceApiError);
      }

      // Fallback: ให้ MediaPipe ช่วยระบุตำแหน่งหน้า แล้วครอปบริเวณนั้น
      // เพื่อให้ Face-API ทำ landmark/descriptor จากภาพที่ใหญ่ขึ้น
      if (!result) {
        const detector = detectorRef.current;
        if (detector && video.videoWidth && video.videoHeight) {
          try {
            const mpTimestamp = Math.max(
              Number(timestamp) || 0,
              performance.now(),
              state.lastAt + 1
            );
            const mp = detector.detectForVideo(video, mpTimestamp);
            const detections = (mp?.detections || [])
              .filter((d) => d?.boundingBox)
              .sort((a, b) =>
                (b.boundingBox.width * b.boundingBox.height) -
                (a.boundingBox.width * a.boundingBox.height)
              );

            const preferred = detections[0];
            if (preferred?.boundingBox) {
              const b = preferred.boundingBox;
              const vw = video.videoWidth;
              const vh = video.videoHeight;
              const pad = Math.max(b.width, b.height) * 0.60;
              const x = Math.max(0, b.originX - pad);
              const y = Math.max(0, b.originY - pad);
              const right = Math.min(vw, b.originX + b.width + pad);
              const bottom = Math.min(vh, b.originY + b.height + pad);
              const cropW = Math.max(96, right - x);
              const cropH = Math.max(96, bottom - y);

              const crop = document.createElement("canvas");
              crop.width = Math.round(cropW);
              crop.height = Math.round(cropH);
              const cctx = crop.getContext("2d");
              if (cctx) {
                cctx.imageSmoothingEnabled = true;
                cctx.imageSmoothingQuality = "high";
                cctx.drawImage(video, x, y, cropW, cropH, 0, 0, crop.width, crop.height);
                result = await detectWithMultipleModels(crop);

                if (result?.landmarks?.positions) {
                  for (const point of result.landmarks.positions) {
                    point.x += x;
                    point.y += y;
                  }
                }
                if (result?.detection?.box) {
                  result.detection.box.x += x;
                  result.detection.box.y += y;
                  result.detection.box.originX = result.detection.box.x;
                  result.detection.box.originY = result.detection.box.y;
                }
              }
            }
          } catch (mpError) {
            console.warn("Enrollment MediaPipe fallback:", mpError);
          }
        }
      }

      if (!result) {
        const lock = state.targetLock;
        if (lock && timestamp - lock.lastSeenAt < 900) {
          lock.lostFrames += 1;
          setEnrollMessage(`👁️ เห็นเป้าหมายล่าสุด • รอใบหน้าเดิมกลับเข้ากล้อง • ไม่เริ่มคนใหม่`);
        } else {
          state.stable = 0;
          state.countdown = 0;
          state.countdownStartedAt = 0;
          setEnrollStable(0);
          setEnrollCountdown(0);
          if (lock) state.targetLock = null;
          setEnrollMessage(`🔍 กำลังค้นหาใบหน้า • ${ENROLL_STAGES[state.stage]?.label || "หน้าตรง"}`);
        }
        return;
      }

      const box = result.detection.box;
      const score = result.detection.score || 0;

      const target = updateEnrollmentTargetLock(result, box, timestamp);
      if (!target.locked) {
        state.stable = 0;
        state.countdown = 0;
        state.countdownStartedAt = 0;
        setEnrollStable(0);
        setEnrollCountdown(0);
        setEnrollMessage(
          `👁️ ล็อกเป้าหมายอยู่ • ไม่สลับคน • รอคนเดิมกลับมา${target.distance < 0.60 ? ` • ระยะ ${target.distance.toFixed(2)}` : ""}`
        );
        return;
      }

      setEnrollMessage(
        target.justLocked
          ? `🎯 ล็อกเป้าหมายแล้ว • จำใบหน้านี้เป็นคนเดียวสำหรับการลงทะเบียน`
          : `🎯 Target Lock • คนเดิม ✓ • Face Distance ${target.distance.toFixed(2)}`
      );

      // The blue mesh has already confirmed a face visually. Do not use the
      // old strict quality checks here; only reject a genuinely tiny/weak face.
      if (score < 0.03 || box.width < 40 || box.height < 40) {
        state.stable = 0;
        state.countdown = 0;
        state.countdownStartedAt = 0;
        setEnrollStable(0);
        setEnrollCountdown(0);
        setEnrollMessage("ตรวจพบหน้าแล้ว • ขยับเข้าใกล้กล้องอีกนิด");
        return;
      }

      const stage = state.stage;
      setEnrollMessage(`✓ พบใบหน้า • กำลังอ่าน Face Descriptor (${result.model || "face-api"})...`);
      const yaw = getFaceYaw(result.landmarks);

      // ขั้นสุดท้าย = ภาพหน้าตรงหลัก หลังเก็บครบ 3 มุม
      // เท่านั้นที่ใช้ countdown 3-2-1 และถ่ายอัตโนมัติ
      if (stage === ENROLL_STAGES.length) {
        const frontValid = Number.isFinite(yaw) && Math.abs(yaw) < 0.35;

        if (!frontValid) {
          state.stable = 0;
          state.countdown = 0;
          state.countdownStartedAt = 0;
          setEnrollStable(0);
          setEnrollCountdown(0);
          setEnrollMessage("กลับมาหน้าตรงและอยู่นิ่ง");
          return;
        }

        state.stable += 1;
        setEnrollStable(state.stable);

        if (!state.countdownStartedAt) {
          state.countdownStartedAt = performance.now();
          state.countdown = 3;
          setEnrollCountdown(3);
          setEnrollMessage("หน้าตรงและนิ่ง ✓ • ภาพหลักใน 3");
          return;
        }

        const elapsed = performance.now() - state.countdownStartedAt;
        const secondsLeft = Math.max(1, 3 - Math.floor(elapsed / 700));

        if (secondsLeft !== state.countdown) {
          state.countdown = secondsLeft;
          setEnrollCountdown(secondsLeft);
          setEnrollMessage(`หน้าตรงและนิ่ง ✓ • ${secondsLeft}`);
        }

        if (elapsed < 2100) return;

        // ถ่ายภาพหลักอัตโนมัติ ไม่ต้องกดปุ่ม
        // ภาพหลัก = ภาพเต็มจากกล้อง ไม่ใช่ภาพครอปใบหน้า
        state.primaryFrontImage = captureFullFrameImage(video, 720, 0.80);
        state.countdown = 0;
        state.countdownStartedAt = 0;
        setEnrollCountdown(0);
        setEnrollMessage("📸 จับภาพหน้าตรงแล้ว ✓ • 🧠 Face Descriptor พร้อม • กำลังบันทึกโปรไฟล์");

        localStorage.setItem("face_enrollment_draft", JSON.stringify({
          employeeId: selectedEmployee?.id || null,
          name: enrollName.trim(),
          samples: state.samples,
          primaryFrontImage: state.primaryFrontImage,
          stage: ENROLL_STAGES.length + 1,
          updatedAt: new Date().toISOString(),
        }));

        await finishEnrollment();
        return;
      }

      let valid = false;

      if (stage === 0) {
        // หน้าตรง: ให้ตรวจพบหน้าแล้วเก็บได้ ไม่ใช้ yaw เป็น gate แข็ง
        valid = true;
      } else if (stage === 1) {
        // ซ้าย: ผ่อน threshold ลง เพราะค่า yaw จาก Face-API
        // เปลี่ยนตามระยะ/กล้อง/การ mirror ได้
        valid = Number.isFinite(yaw) && Math.abs(yaw) > 0.08;
        if (valid && state.firstTurnSign == null) {
          state.firstTurnSign = yaw > 0 ? 1 : -1;
        }
      } else if (stage === 2) {
        // ขวา: ต้องเป็นคนละทิศกับมุมแรก แต่ไม่บังคับให้หมุนมาก
        valid =
          Number.isFinite(yaw) &&
          state.firstTurnSign != null &&
          yaw * state.firstTurnSign < -0.08;
      }

      if (!valid) {
        state.stable = 0;
        setEnrollStable(0);
        const messages = [
          "หน้าตรง • มองกล้องตรง ๆ",
          "แก้มซ้าย • หันหน้าไปด้านข้างให้เห็นชัด",
          "แก้มขวา • หันหน้าไปอีกด้านให้เห็นชัด",
        ];
        setEnrollMessage(messages[stage] || "จัดใบหน้าให้อยู่ในกรอบ");
        return;
      }

      // มุมทั้ง 3: ตรวจผ่านแล้วบันทึกทันที
      // ไม่มี countdown 3-2-1 ในแต่ละมุม
      state.stable += 1;
      setEnrollStable(state.stable);
      state.countdown = 0;
      state.countdownStartedAt = 0;
      setEnrollCountdown(0);
      setEnrollMessage(`✓ Face ผ่าน • 📸 กำลังจับภาพ ${ENROLL_STAGES[stage].label}...`);

      const faceSnapshot = captureFaceSnapshotData(video, box);
      const image = faceSnapshot.canvas;
      setEnrollMessage(`📸 จับภาพ ${ENROLL_STAGES[stage].label} แล้ว ✓ • 🧠 บันทึก Face Descriptor...`);

      const capturedSample = {
        stage: ENROLL_STAGES[stage].key,

        // Face descriptor = ตัวแทนเอกลักษณ์ใบหน้า
        descriptor: Array.from(result.descriptor),

        // ภาพใบหน้าเต็ม ๆ ของแต่ละมุม
        image: faceSnapshot.image,

        // ภาพช่วงหัว-ไหล่ + fingerprint เพื่อใช้เป็นหลักฐานเสริม
        // ตอนค้นหาจะยังให้ face descriptor เป็นตัวหลัก
        appearanceImage: faceSnapshot.fullImage,
        appearanceFingerprint: faceSnapshot.appearanceFingerprint,

        // รายละเอียดรูปทรงใบหน้า
        landmarks: normalizeEnrollmentLandmarks(result.landmarks, box),
        yaw,
        pitch: Number((getFacePitch(result.landmarks) || 0).toFixed(5)),
        boxRatio: {
          width: Number(box.width.toFixed(2)),
          height: Number(box.height.toFixed(2)),
          aspect: Number((box.width / Math.max(1, box.height)).toFixed(4)),
        },

        // ข้อมูล depth/geometry จาก Face Landmarker
        depthProfile: getEnrollmentDepth(),

        model: result.model || "tiny",
        // Target-lock metadata: ใช้ตรวจสอบว่า sample นี้มาจากคนเดียวกับที่ล็อกไว้
        targetDistance: Number((enrollmentRef.current.targetLock?.descriptor
          ? faceapi.euclideanDistance(new Float32Array(result.descriptor), new Float32Array(enrollmentRef.current.targetLock.descriptor))
          : 0).toFixed(4)),
        targetHits: enrollmentRef.current.targetLock?.hits || 1,
        capturedAt: new Date().toISOString(),
      };

      // IMPORTANT: persist the angle immediately, not only after all 3 angles.
      state.samples.push(capturedSample);
      safeSetLocalStorage("face_enrollment_draft", {
        employeeId: selectedEmployee?.id || null,
        name: enrollName.trim(),
        samples: state.samples,
        stage,
        updatedAt: new Date().toISOString(),
      });

      const completedLabel = ENROLL_STAGES[stage].label;
      state.stage = stage + 1;
      state.stable = 0;
      state.countdown = 0;
      state.countdownStartedAt = 0;
      setEnrollStage(state.stage);
      setEnrollStable(0);
      setEnrollCountdown(0);

      if (state.stage >= ENROLL_STAGES.length) {
        // เข้าสู่ขั้นตอนสุดท้าย: กลับมาหน้าตรงเพื่อถ่าย "ภาพหลัก"
        // ภาพหลักจะมี countdown 3-2-1 เพียงครั้งเดียว
        state.stage = ENROLL_STAGES.length;
        state.countdown = 0;
        state.countdownStartedAt = 0;
        setEnrollStage(ENROLL_STAGES.length);
        setEnrollCountdown(0);
        setEnrollMessage("ครบ 3 มุมแล้ว ✓ • กลับมาหน้าตรงและอยู่นิ่ง");
        setScanMessage("กลับมาหน้าตรงเพื่อถ่ายภาพหลัก");
      } else {
        const nextLabel = ENROLL_STAGES[state.stage].label;
        setEnrollMessage(`บันทึก ${completedLabel} แล้ว ✓ • ต่อไป ${nextLabel}`);
        setScanMessage(`บันทึกแล้ว • ${nextLabel}`);
      }
    } catch (error) {
      console.warn("Enrollment frame:", error);
      state.stable = 0;
      setEnrollStable(0);
      setEnrollMessage("ตรวจจับใบหน้าไม่สำเร็จ • ลองมองเข้ากล้องอีกครั้ง");
    } finally {
      enrollmentBusyRef.current = false;
    }
  };

  // ฟังก์ชันสกัดใบหน้าแบบ Asynchronous เบื้องหลัง ไม่ทำให้วิดีโอกระตุก
  // ยืนยันตัวตนหลายเฟรมก่อนล็อกชื่อ
  // หลักการสำคัญ: "ไม่แน่ใจ = ไม่ระบุชื่อ" ดีกว่าระบุผิดคน
  // ต้องยืนยันคนเดิมหลายเฟรมก่อนขึ้นชื่อสีเขียว
  // ถ้าไม่มั่นใจจะไม่ฝืนระบุชื่อ
  // Recognition แบบเร็วสำหรับหลายคนพร้อมกัน:
  // ไม่ต้องรอ 3/3 เฟรม เมื่อมั่นใจแล้วให้ล็อกชื่อทันที
  // 1) ถ้า session นี้เคยยืนยันคนนี้แล้ว ให้เทียบกับ descriptor ที่ cache ไว้ก่อน
  // 2) ถ้ายังไม่เคยเจอ ค่อยค้นกับฐานข้อมูลทั้งหมด
  // 3) เมื่อมั่นใจแล้ว "ล็อกทันที" ไม่ต้องรอ 3 เฟรม
  const recognizeFace = async (track, video, box) => {
    recognitionActiveRef.current += 1;

    try {
      track.statusLabel = "📸 จับภาพใบหน้า...";
      track.color = "#38bdf8";
      setScanActivity(`📸 จับภาพ Face • คนที่ ${track.id.substring(0, 5)}`);
      await new Promise(requestAnimationFrame);

      const querySnapshot = captureFaceSnapshotData(video, box);
      track.statusLabel = "🧠 สร้าง Face Descriptor...";
      setScanActivity(`🧠 วิเคราะห์ Face Descriptor • คนที่ ${track.id.substring(0, 5)}`);
      const detection = await detectWithMultipleModels(querySnapshot.canvas);

      if (!detection) {
        setScanActivity("⚠️ พบกรอบหน้า แต่สร้าง Face Descriptor ไม่สำเร็จ • กำลังลองใหม่");
        track.statusLabel = "⚠️ อ่าน Face ไม่สำเร็จ • ลองใหม่...";
        track.color = "#f59e0b";
        track.lastRecognition = 0;
        return;
      }

      detection.appearanceFingerprint = querySnapshot.appearanceFingerprint;

      const currentDescriptor = Array.from(detection.descriptor);
      const people = savedFacesRef.current || [];
      setScanActivity(`🔎 ค้นฐานข้อมูลใบหน้า ${people.length} โปรไฟล์...`);

      if (!people.length) {
        setScanActivity("⚠️ ตรวจพบใบหน้าแล้ว แต่ฐานข้อมูลยังไม่มีบุคคล");
        track.statusLabel = "ยังไม่มีข้อมูลในระบบ";
        track.color = "#ef4444";
        track.matchDistance = null;
        track.found = false;
        track.lockedResult = false;
        track.lastRecognition = performance.now();
        return;
      }

      let result = null;

      // ----- ขั้นที่ 1: session memory แบบผูกกับ TRACK เท่านั้น -----
      // สำคัญ: ห้ามเอา cache ของคนที่ 1 ไปใช้ตัดสินใบหน้าคนที่ 2/3
      // เพราะใบหน้าคนที่อยู่ด้านหลังอาจมี descriptor ใกล้พอจนถูกเรียกชื่อผิดได้
      // เราจะใช้ memory ได้ก็ต่อเมื่อเป็น track เดิม และตำแหน่งใบหน้ายังต่อเนื่องกัน
      if (track.recognizedPersonId) {
        const cached = recognizedSessionRef.current.get(track.recognizedPersonId);
        const previousBox = track.lastRecognizedBox;
        const currentBox = box;

        const boxCenter = (b) => ({
          x: b.originX + b.width / 2,
          y: b.originY + b.height / 2,
        });
        const centerA = previousBox ? boxCenter(previousBox) : null;
        const centerB = boxCenter(currentBox);
        const spatialDistance = centerA
          ? Math.hypot(centerA.x - centerB.x, centerA.y - centerB.y)
          : Infinity;
        const spatialLimit = Math.max(
          previousBox?.width || 0,
          previousBox?.height || 0,
          currentBox.width,
          currentBox.height
        ) * 0.75;

        if (cached && previousBox && spatialDistance <= spatialLimit) {
          const distances = (cached.descriptors || [])
            .map((vector) =>
              faceapi.euclideanDistance(
                detection.descriptor,
                new Float32Array(vector)
              )
            )
            .sort((x, y) => x - y);

          const cachedBest = distances[0];
          const cachedSecond = distances[1] ?? Infinity;

          // Memory เป็นเพียงการยืนยันคนเดิม ไม่ใช่ใบอนุญาตให้ตั้งชื่อคนอื่น
          // จึงใช้ threshold เข้มกว่าการค้นฐานข้อมูลเต็ม
          if (
            Number.isFinite(cachedBest) &&
            cachedBest <= 0.40 &&
            cachedSecond - cachedBest >= 0.025
          ) {
            result = {
              person: cached.person,
              distance: cachedBest,
              margin: cachedSecond - cachedBest,
              support: 1,
              matchedImage: null,
              matchedAngle: null,
              source: "track-session-cache",
              confidentEnough: true,
            };
          }
        }
      }

      // ----- ขั้นที่ 2: ค้นฐานข้อมูลเต็ม เฉพาะเมื่อ cache ใช้ไม่ได้ -----
      if (!result) {
        result = await recognizeWithSavedImages(detection, people);
      }

      if (!result?.person || !result.confidentEnough) {
        setScanActivity(result?.ambiguous ? "⚠️ หน้าคล้ายหลายคน • ไม่ฟันธง" : "⚠️ ไม่พบคนที่ตรงกัน • พร้อมสแกนใหม่");
        track.pendingPersonId = null;
        track.pendingHits = 0;
        track.statusLabel = result?.ambiguous
          ? "หน้าคล้ายหลายคน • ไม่ฟันธง"
          : result
            ? "ไม่พบคนที่ตรงกัน"
            : "อ่านใบหน้าได้ แต่ยังระบุคนไม่ได้";
        track.color = "#f59e0b";
        track.matchDistance = Number.isFinite(result?.distance)
          ? result.distance
          : null;
        track.found = false;
        track.lockedResult = false;
        track.recognitionMethod = "strict";
        track.lastRecognition = performance.now();
        return;
      }

      // ----- เจอแล้ว = ล็อกทันที -----
      setScanActivity(`✓ พบ ${result.person.name} • กำลังบันทึกการตรวจสอบ`);
      track.statusLabel = `✓ ${result.person.name}`;
      track.color = "#22c55e";
      track.found = true;
      track.lockedResult = true;
      track.matchedAngle = result.matchedAngle || null;
      track.matchDistance = result.distance;
      track.recognitionMethod = result.source || "all-images";
      track.recognizedPersonId = result.person.id;
      track.lastRecognizedBox = { ...box };
      track.lastRecognition = performance.now();

      // จำผลไว้เฉพาะ session ปัจจุบัน
      // เมื่อ refresh หน้า React/Javascript context จะหายเอง
      recognizedSessionRef.current.set(result.person.id, {
        person: result.person,
        descriptors: [
          ...(result.person.descriptors || []),
          ...(
            Array.isArray(result.person.scanGallery)
              ? result.person.scanGallery.map((item) => item.descriptor)
              : []
          ),
          currentDescriptor,
        ],
        firstSeenAt:
          recognizedSessionRef.current.get(result.person.id)?.firstSeenAt ||
          new Date().toISOString(),
        lastSeenAt: new Date().toISOString(),
        lastDistance: result.distance,
      });

      // เก็บภาพค้นหาที่ "ยืนยันแล้ว" เพิ่มครั้งเดียวเป็นตัวอย่างเสริม
      // ไม่บันทึกซ้ำทุกเฟรม
      const cachedPerson = recognizedSessionRef.current.get(result.person.id);
      if (cachedPerson && !cachedPerson.savedScanImage) {
        cachedPerson.savedScanImage = true;

        try {
          const confirmedSnapshot = querySnapshot;

          const updatedPeople = savedFacesRef.current.map((person) => {
            if (person.id !== result.person.id) return person;

            const gallery = Array.isArray(person.scanGallery)
              ? [...person.scanGallery]
              : [];

            const alreadySimilar = gallery.some((item) => {
              if (!Array.isArray(item.descriptor)) return false;
              return (
                faceapi.euclideanDistance(
                  currentDescriptor,
                  new Float32Array(item.descriptor)
                ) < 0.08
              );
            });

            if (!alreadySimilar) {
              gallery.push({
                id: crypto.randomUUID(),
                image: confirmedSnapshot.image,
                fullImage: confirmedSnapshot.fullImage,
                descriptor: currentDescriptor,
                appearanceFingerprint:
                  confirmedSnapshot.appearanceFingerprint || null,
                capturedAt: new Date().toISOString(),
              });
            }

            return {
              ...person,
              scanGallery: gallery.slice(-MAX_SCAN_GALLERY_PER_PERSON),
            };
          });

          savedFacesRef.current = updatedPeople;
          setSavedFaces(updatedPeople);

          const storageOk = safeSetLocalStorage(FACE_DB_KEY, updatedPeople);
          if (storageOk) updateFaceMatcher(updatedPeople);

          saveScanSnapshot({
            image: confirmedSnapshot.image,
            fullImage: confirmedSnapshot.fullImage,
            appearanceFingerprint:
              confirmedSnapshot.appearanceFingerprint || null,
            descriptor: currentDescriptor,
            personId: result.person.id,
            personName: result.person.name,
            distance: result.distance,
            source: "confirmed-match",
          });
          setScanActivity(`✓ ${result.person.name} • บันทึกการตรวจสอบแล้ว`);
        } catch (galleryError) {
          console.warn("Save confirmed scan:", galleryError);
        }
      }

      const profile = getEnrollmentDepth();
      if (profile) {
        setFaceGeometry({
          current: profile,
          saved: result.person.depthProfile || null,
          score: compareDepthProfiles(profile, result.person.depthProfile),
        });
      }
    } catch (error) {
      console.warn("Face recognition retry:", error);
      setScanActivity(`❌ Face recognition error • ${error?.message || "ลองใหม่"}`);
      track.isRecognizing = false;
      track.lastRecognition = 0;
      track.statusLabel = "ตรวจสอบใหม่...";
      track.color = "#f59e0b";
      track.found = false;
      track.lockedResult = false;
    } finally {
      track.isRecognizing = false;
      recognitionActiveRef.current = Math.max(
        0,
        recognitionActiveRef.current - 1
      );
    }
  };


  // ตรวจสอบว่า bounding box จาก FaceDetector สอดคล้องกับโครงหน้า 478 จุดหรือไม่
  // ใช้ Face Landmarker เป็นชั้นยืนยัน (verification) ก่อนส่งเข้า tracking/recognition
  const verifyDetectionWithLandmarks = (detection, faceLandmarks, width, height) => {
    const box = detection?.boundingBox;
    if (!box || !faceLandmarks?.length) return { ok: false, score: 0, landmarks: null };
    if (faceLandmarks.length < 300) return { ok: false, score: 0, landmarks: null };

    const points = faceLandmarks.filter(Boolean);
    if (points.length < 300) return { ok: false, score: 0, landmarks: null };

    const xs = points.map((p) => p.x * width);
    const ys = points.map((p) => p.y * height);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const landmarkBox = {
      x: minX,
      y: minY,
      width: Math.max(1, maxX - minX),
      height: Math.max(1, maxY - minY),
    };

    const detectorCx = box.originX + box.width / 2;
    const detectorCy = box.originY + box.height / 2;
    const landmarkCx = landmarkBox.x + landmarkBox.width / 2;
    const landmarkCy = landmarkBox.y + landmarkBox.height / 2;
    const centerDistance = Math.hypot(detectorCx - landmarkCx, detectorCy - landmarkCy);
    const centerTolerance = Math.max(box.width, box.height) * 0.65;

    const ix1 = Math.max(box.originX, landmarkBox.x);
    const iy1 = Math.max(box.originY, landmarkBox.y);
    const ix2 = Math.min(box.originX + box.width, landmarkBox.x + landmarkBox.width);
    const iy2 = Math.min(box.originY + box.height, landmarkBox.y + landmarkBox.height);
    const intersection = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    const minArea = Math.max(1, Math.min(box.width * box.height, landmarkBox.width * landmarkBox.height));
    const overlap = intersection / minArea;

    // ตรวจ landmark สำคัญ: ตาซ้าย/ขวา จมูก ปาก และคาง ต้องอยู่ในบริเวณ detection
    const keyIds = [33, 263, 1, 13, 14, 152];
    let keyInside = 0;
    for (const id of keyIds) {
      const point = faceLandmarks[id];
      if (!point) continue;
      const x = point.x * width;
      const y = point.y * height;
      const inside =
        x >= box.originX - box.width * 0.18 &&
        x <= box.originX + box.width * 1.18 &&
        y >= box.originY - box.height * 0.18 &&
        y <= box.originY + box.height * 1.18;
      if (inside) keyInside += 1;
    }

    const eyeLeft = faceLandmarks[33];
    const eyeRight = faceLandmarks[263];
    const nose = faceLandmarks[1];
    const eyeDistance = eyeLeft && eyeRight
      ? Math.hypot((eyeLeft.x - eyeRight.x) * width, (eyeLeft.y - eyeRight.y) * height)
      : 0;

    const geometryOk =
      landmarkBox.width >= box.width * 0.22 &&
      landmarkBox.height >= box.height * 0.22 &&
      eyeDistance >= Math.max(3, box.width * 0.12) &&
      !!nose;

    const centerScore = Math.max(0, 1 - centerDistance / Math.max(1, centerTolerance));
    const overlapScore = Math.min(1, overlap / 0.65);
    const keyScore = keyInside / keyIds.length;
    const geometryScore = geometryOk ? 1 : 0;
    const score = Number((
      centerScore * 0.30 +
      overlapScore * 0.35 +
      keyScore * 0.20 +
      geometryScore * 0.15
    ).toFixed(3));

    return {
      ok: score >= 0.50 && keyInside >= 5 && geometryOk,
      score,
      landmarks: faceLandmarks,
    };
  };

  const getBestLandmarksForDetection = (detection, landmarkSets, width, height, used) => {
    const box = detection?.boundingBox;
    if (!box || !landmarkSets?.length) return null;

    const cx = box.originX + box.width / 2;
    const cy = box.originY + box.height / 2;
    let best = null;
    let bestDistance = Infinity;

    landmarkSets.forEach((landmarks, index) => {
      if (used.has(index) || !landmarks?.length) return;
      const points = landmarks.filter(Boolean);
      if (points.length < 300) return;
      const lx = points.reduce((s, p) => s + p.x * width, 0) / points.length;
      const ly = points.reduce((s, p) => s + p.y * height, 0) / points.length;
      const distance = Math.hypot(lx - cx, ly - cy);
      const maxDistance = Math.max(box.width, box.height) * 0.9;
      if (distance <= maxDistance && distance < bestDistance) {
        bestDistance = distance;
        best = { index, landmarks };
      }
    });

    return best;
  };

  // Smooth detection loop with throttled MediaPipe inference
  function detectFaces(timestamp = 0) {
    const frameTimestamp = Math.max(timestamp, lastFrameTimestampRef.current + 1);
    lastFrameTimestampRef.current = frameTimestamp;
    
    if (!videoRef.current || !canvasRef.current || !detectorRef.current || !cameraOnRef.current) return;
    if (modeRef.current !== "scan" && modeRef.current !== "enroll") return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = detectorRef.current;

    if (video.readyState < 2) {
      animationRef.current = requestAnimationFrame(detectFaces);
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
      animationRef.current = requestAnimationFrame(detectFaces);
      return;
    }

    // Do not run the synchronous MediaPipe detector at 60fps.
    // Running it around 12-15fps keeps the UI responsive and avoids apparent freezes.
    if (frameTimestamp < nextDetectionAtRef.current) {
      animationRef.current = requestAnimationFrame(detectFaces);
      return;
    }
    lastDetectorRunRef.current = frameTimestamp;
    nextDetectionAtRef.current = frameTimestamp + 120;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const result = detector.detectForVideo(video, frameTimestamp);
    const rawDetections = result.detections || [];

    // Face Landmarker ทำงานทั้ง Scan และ Enrollment เพื่อให้โครงหน้าจุดฟ้ายังแสดง
    // จำกัดความถี่ไว้เพื่อไม่ให้ main thread หนักจนภาพค้างหรือกระพริบ
    let meshResult = null;
    if (
      faceLandmarkerRef.current &&
      frameTimestamp - meshLastTimestampRef.current >= 180
    ) {
      meshLastTimestampRef.current = frameTimestamp;
      try {
        meshResult = faceLandmarkerRef.current.detectForVideo(video, frameTimestamp);
        if (meshResult) {
          lastMeshResultRef.current = meshResult;
          const allLandmarks = meshResult.faceLandmarks || [];
          const firstProfile = allLandmarks[0]
            ? buildDepthProfile(allLandmarks[0])
            : null;
          if (firstProfile) {
            setFaceGeometry((prev) => ({ ...(prev || {}), current: firstProfile }));
          }
        }
      } catch (meshError) {
        console.warn("Face Landmarker frame error:", meshError);
      }
    }

    // แสดงจุด/เส้นโครงหน้าสีฟ้าเฉพาะหน้า "ลงทะเบียน"
    // หน้า Scan ใช้ Landmarker ภายใน แต่ไม่แสดง mesh ให้ผู้ใช้เห็น
    if (modeRef.current === "enroll" && meshResult?.faceLandmarks?.length) {
      for (const landmarks of meshResult.faceLandmarks) {
        drawFaceMesh(ctx, landmarks, width, height, "#22d3ee");
      }
    }

    // MediaPipe FaceDetector สามารถคืนใบหน้าหลายคนในเฟรมเดียว
    // เก็บไว้สูงสุด 7 คน และเรียงจากใบหน้าใหญ่ -> เล็ก
    // เพื่อให้คนที่อยู่ใกล้กล้องถูกประมวลผลก่อน แต่ไม่ทิ้งคนที่อยู่ไกล
    // หากใบหน้ายังมีขนาด/ความมั่นใจเพียงพอ
    const landmarkSets = meshResult?.faceLandmarks || [];
    const usedLandmarkIndexes = new Set();

    const validDetections = rawDetections
      .map((detection) => {
        const box = detection.boundingBox;
        const score = detection.categories?.[0]?.score || 0;
        if (!box || score < 0.30) return null; // ให้ detector จับหน้าได้ก่อน แล้วค่อยให้ Face-API ตัดสินตัวตน

        let verification = { ok: false, score: 0, landmarks: null };
        let landmarkIndex = -1;

        if (landmarkSets.length) {
          const best = getBestLandmarksForDetection(
            detection,
            landmarkSets,
            width,
            height,
            usedLandmarkIndexes
          );
          if (best) {
            verification = verifyDetectionWithLandmarks(
              detection,
              best.landmarks,
              width,
              height
            );
            landmarkIndex = best.index;
          }
        }

        // Landmarker เป็นข้อมูลเสริมเท่านั้น ห้ามเป็น hard gate ของ Scan
        // เพราะ Detector กับ Landmarker ทำงานคนละรอบเวลา ถ้าเอามาบังคับ
        // จะเกิดอาการ "เห็นหน้าแล้วแต่ไม่ไปต่อ" เมื่อ mesh ยังไม่ทันอัปเดต
        if (landmarkIndex >= 0 && verification.ok) {
          usedLandmarkIndexes.add(landmarkIndex);
        }

        const areaRatio = (box.width * box.height) / (width * height);
        const aspectRatio = box.width / Math.max(1, box.height);
        if (
          box.width < MIN_FACE_WIDTH ||
          box.height < MIN_FACE_HEIGHT ||
          areaRatio < 0.000025 ||
          aspectRatio < 0.35 ||
          aspectRatio > 1.8
        ) return null;

        return {
          detection,
          meshScore: verification.score,
          landmarks: verification.landmarks,
        };
      })
      .filter(Boolean)
      .sort((a, b) => {
        const aBox = a.detection.boundingBox;
        const bBox = b.detection.boundingBox;
        return (bBox.width * bBox.height) - (aBox.width * aBox.height);
      })
      .slice(0, MAX_PEOPLE);
    
    // If a face is found, briefly pause the detector. If nothing is found,
    // retry soon. This pulse strategy reduces CPU/GPU load without making the UI feel stuck.
    nextDetectionAtRef.current = rawDetections.length ? frameTimestamp + 140 : frameTimestamp + 100;

    // Dedupe detector results BEFORE creating/updating tracks.
    // FaceDetector + landmark verification can occasionally return two boxes for one face.
    // Keep the strongest/largest box so one real person gets one track and one recognition job.
    const dedupedValidDetections = [];
    for (const item of validDetections) {
      const box = item.detection?.boundingBox;
      if (!box) continue;

      const duplicateIndex = dedupedValidDetections.findIndex((kept) => {
        const kb = kept.detection.boundingBox;
        const ix1 = Math.max(box.originX, kb.originX);
        const iy1 = Math.max(box.originY, kb.originY);
        const ix2 = Math.min(box.originX + box.width, kb.originX + kb.width);
        const iy2 = Math.min(box.originY + box.height, kb.originY + kb.height);
        const iw = Math.max(0, ix2 - ix1);
        const ih = Math.max(0, iy2 - iy1);
        const intersection = iw * ih;
        const union = box.width * box.height + kb.width * kb.height - intersection;
        const iou = union > 0 ? intersection / union : 0;
        const c1x = box.originX + box.width / 2;
        const c1y = box.originY + box.height / 2;
        const c2x = kb.originX + kb.width / 2;
        const c2y = kb.originY + kb.height / 2;
        const centerDistance = Math.hypot(c1x - c2x, c1y - c2y);
        const centerLimit = Math.max(box.width, box.height, kb.width, kb.height) * 0.45;
        return iou >= 0.35 || centerDistance <= centerLimit;
      });

      if (duplicateIndex === -1) {
        dedupedValidDetections.push(item);
      } else {
        const kept = dedupedValidDetections[duplicateIndex];
        const keptBox = kept.detection.boundingBox;
        const keptArea = keptBox.width * keptBox.height;
        const currentArea = box.width * box.height;
        if (currentArea > keptArea) dedupedValidDetections[duplicateIndex] = item;
      }
    }

    const tracks = tracksRef.current;
    const seenTrackIds = new Set();
    const confirmedDetections = [];

    dedupedValidDetections.forEach((item) => {
      const detection = item.detection;
      const box = detection.boundingBox;
      const centerX = box.originX + box.width / 2;
      const centerY = box.originY + box.height / 2;
      
      let track = tracks
        .filter((candidate) => !seenTrackIds.has(candidate.id))
        .map((candidate) => ({
          candidate,
          distance: Math.hypot(candidate.centerX - centerX, candidate.centerY - centerY),
        }))
        .filter(({ candidate, distance }) => distance < Math.max(box.width, candidate.width) * 0.7) // จับ track เดิมให้ต่อเนื่อง ไม่สร้างกรอบซ้ำ
        .sort((left, right) => left.distance - right.distance)[0]?.candidate;

      if (!track) {
        track = { 
          id: crypto.randomUUID(), 
          frames: 0, 
          statusLabel: "กำลังตรวจสอบ", 
          color: "#f59e0b", // ยังไม่ยืนยันตัวตน = เหลือง
          isProcessed: false,
          isRecognizing: false,
          lastRecognition: 0,
          matchDistance: null,
          found: false,
          lockedResult: false,
          stability: 0,
          pendingPersonId: null,
          pendingHits: 0,
          lastSnapshotSavedAt: 0,
          recognizedPersonId: null,
          lastRecognizedBox: null
        };
        tracks.push(track);
      }

      track.frames += 1;
      track.centerX = centerX;
      track.centerY = centerY;
      track.width = box.width;
      track.detection = detection;
      track.meshScore = item.meshScore;
      track.landmarks = item.landmarks;
      track.lastSeen = frameTimestamp;
      seenTrackIds.add(track.id);
      
      // Apply temporal smoothing for stable boxes
      const smoothedBox = smoothBox(track.id, box, width, height);
      track.smoothedBox = smoothedBox;
      
      // Initialize stability counter
      if (!stabilityCounterRef.current[track.id]) {
        stabilityCounterRef.current[track.id] = 0;
      }
      
      // Increment stability if box is stable
      const positionChange = Math.hypot(
        smoothedBox.originX - track.centerX + smoothedBox.width / 2,
        smoothedBox.originY - track.centerY + smoothedBox.height / 2
      );
      
      if (positionChange < 5) {
        stabilityCounterRef.current[track.id] = Math.min(10, stabilityCounterRef.current[track.id] + 1);
      } else {
        stabilityCounterRef.current[track.id] = Math.max(0, stabilityCounterRef.current[track.id] - 1);
      }
      
      track.stability = stabilityCounterRef.current[track.id];

      // เริ่มวิเคราะห์ทันทีตั้งแต่เฟรมแรกที่ตรวจพบหน้า
      if (track.frames >= 1) {
        confirmedDetections.push(track);
        
        if (
          modeRef.current === "scan" &&
          !track.isRecognizing &&
          !track.lockedResult &&
          recognitionActiveRef.current < MAX_CONCURRENT_RECOGNITION &&
          frameTimestamp - (track.lastRecognition || 0) >= 300
        ) {
          track.isRecognizing = true;
          track.statusLabel = "กำลังค้นหา...";
          track.color = "#f59e0b";
          recognizeFace(track, video, box);
        }
      } else if (track.frames >= 1) {
        // แสดงกรอบตั้งแต่เฟรมแรกที่ผ่าน detector
        confirmedDetections.push(track);
      }
    });

    tracksRef.current = tracks.filter((track) => frameTimestamp - track.lastSeen < 500);

    // รักษาลำดับใหญ่ -> เล็กในผลลัพธ์ทุกเฟรม
    const orderedDetections = confirmedDetections
      .sort((a, b) => {
        const aBox = a.detection?.boundingBox;
        const bBox = b.detection?.boundingBox;
        return (bBox.width * bBox.height) - (aBox.width * aBox.height);
      })
      .slice(0, MAX_PEOPLE);

    // Final visual guard: never render two nearby tracks as two faces.
    const deduplicatedDetections = [];
    for (const track of orderedDetections) {
      const duplicateIndex = deduplicatedDetections.findIndex((kept) => {
        const distance = Math.hypot(kept.centerX - track.centerX, kept.centerY - track.centerY);
        const size = Math.max(kept.width, kept.height, track.width, track.height);
        return distance < size * 0.45;
      });

      if (duplicateIndex === -1) {
        deduplicatedDetections.push(track);
      } else {
        const kept = deduplicatedDetections[duplicateIndex];
        // Prefer the locked/recognized track; otherwise prefer the larger face.
        const keepCurrent =
          (track.lockedResult && !kept.lockedResult) ||
          (track.found && !kept.found) ||
          (track.width * track.height > kept.width * kept.height);
        if (keepCurrent) deduplicatedDetections[duplicateIndex] = track;
      }
    }

    setFaceCount(deduplicatedDetections.length);
    setDetectedFaces(deduplicatedDetections.map((trk, index) => ({
      id: trk.id.substring(0, 5),
      label: trk.statusLabel,
      box: trk.smoothedBox || trk.detection.boundingBox,
      order: index + 1,
      area: trk.detection.boundingBox.width * trk.detection.boundingBox.height,
      meshScore: trk.meshScore ?? null,
      verifiedByMesh: Boolean(trk.landmarks),
      stability: trk.stability || 0,
      matchedAngle: trk.matchedAngle || null,
      found: Boolean(trk.found),
      lockedResult: Boolean(trk.lockedResult),
      matchDistance: trk.matchDistance ?? null,
      recognitionMethod: trk.recognitionMethod || null,
      recognitionStage: trk.statusLabel || "กำลังตรวจจับใบหน้า...",
    })));
    if (modeRef.current === "scan") {
      if (!orderedDetections.length) {
        setScanMessage("ยังไม่พบใบหน้า • กำลังค้นหา...");
        setScanActivity("🔍 กำลังค้นหาใบหน้าในภาพ...");
      } else {
        setScanMessage(`พบ ${orderedDetections.length} คน • AI กำลังวิเคราะห์`);
      }
    }

    // Detection boxes are rendered by the React overlay below.
    // The canvas is reserved for the 3D mesh, so there is no duplicate box.

    if (cameraOnRef.current) {
      animationRef.current = requestAnimationFrame(detectFaces);
    }
  }
  return (
    <div className={cameraOn ? "fixed inset-0 z-[9999] h-[100dvh] w-screen overflow-hidden bg-black" : "min-h-[calc(100dvh-120px)] space-y-3 overflow-x-hidden"}>
      <main className={cameraOn ? "relative h-[100dvh] w-full overflow-hidden" : "w-full space-y-3 p-3 pb-4 sm:space-y-4 sm:p-4"}>
        <section
          className={
            cameraOn
              ? "camera-panel absolute inset-0 h-[100dvh] w-full overflow-hidden bg-black"
              : "fixed -left-[10000px] top-0 h-px w-px overflow-hidden opacity-0 pointer-events-none"
          }
        >
            <video
              ref={videoRef}
              muted
              playsInline
              autoPlay
              className={`absolute inset-0 h-full w-full object-cover ${cameraMirrored ? "-scale-x-100" : ""}`}
            />

            <canvas
              ref={canvasRef}
              className={`pointer-events-none absolute inset-0 h-full w-full ${cameraMirrored ? "-scale-x-100" : ""}`}
            />

            <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-2">
              <div className="rounded-full bg-black/70 px-3 py-1.5 text-[10px] font-bold backdrop-blur">
                {enrollMessage || "กำลังค้นหาใบหน้า..."}
              </div>
              <div className="rounded-full bg-green-500/10 px-2 py-1 text-[9px] text-green-300 backdrop-blur">
                LIVE
              </div>
            </div>

            {cameraOn && enrollmentRef.current.active && (
              <div className="absolute top-4 left-3 right-3 rounded-xl border border-cyan-400/30 bg-black/90 p-4 backdrop-blur-xl">
                <div className="text-center">
                  <div className="text-lg font-black text-cyan-300">
                    {ENROLL_STAGES[enrollStage]?.label || "เตรียมตัว"}
                  </div>
                 
                  <div className="mt-3 flex items-center justify-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-cyan-400 animate-pulse" />
                    <div className="text-xs text-cyan-200">
                      {Math.min(2, enrollStable || 0)}/2 เฟรมยืนยัน
                    </div>
                  </div>
                </div>
              </div>
            )}

            {cameraError && (
              <div className="absolute bottom-12 left-3 right-3 rounded-lg border border-red-500/30 bg-red-950/90 p-2 text-center text-[10px] text-red-200">
                {cameraError}
              </div>
            )}
          </section>

        {cameraOn && (
          <div className="absolute bottom-[max(1rem,env(safe-area-inset-bottom))] left-0 right-0 z-40 flex flex-wrap items-center justify-center gap-2 px-3">
            {!enrollmentRef.current.active ? (
              <button
                type="button"
                onClick={startEnrollment}
                disabled={!selectedEmployee?.id || loading}
                className="min-h-11 rounded-xl bg-cyan-400 px-5 py-2.5 text-xs font-black text-black disabled:opacity-40"
              >
                ▶ เริ่มบันทึกใบหน้า
              </button>
            ) : (
              <div className="min-h-11 rounded-xl bg-cyan-400 px-5 py-2.5 text-xs font-black text-black">
                ● กำลังบันทึกใบหน้า
              </div>
            )}

            <button
              type="button"
              onClick={toggleCamera}
              className="min-h-11 rounded-xl bg-white/10 px-4 py-2.5 text-xs font-black text-white"
            >
              ✕ ปิดกล้อง
            </button>

            <button
              type="button"
              onClick={switchCamera}
              className="min-h-11 rounded-xl bg-white/5 px-4 py-2.5 text-xs font-bold text-white ring-1 ring-white/10"
            >
              ↻ กล้องหน้า / หลัง
            </button>

            <span className="max-w-[220px] truncate text-[9px] text-gray-500">
              {cameraLabel}
            </span>
          </div>
        )}

        <section className={cameraOn ? "hidden" : "rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-cyan-950/30 to-black/50 p-6 shadow-2xl shadow-cyan-500/10"}>
          <div className="flex items-center justify-between gap-3 mb-6">
            <div>
              <div className="flex items-center gap-2">
                <div className="text-[15px] font-black tracking-[0.2em] text-cyan-300">
                  บันทึกใบหน้า
                </div>
                <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
              </div>
              {/* <h2 className="mt-2 text-xl font-black text-white">
                บันทึกใบหน้า
              </h2> */}
              {/* <p className="text-[11px] text-gray-400 mt-1">
                ค้นหาพนักงาน → เปิดกล้อง → กด “เริ่มบันทึก” เมื่อต้องการเก็บข้อมูลใบหน้า
              </p> */}
            </div>

            <div className="rounded-full bg-gradient-to-br from-cyan-400/20 to-cyan-600/20 px-3 py-1.5 text-[9px] font-bold text-cyan-300 border border-cyan-400/30">
              AI
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-[10px] font-bold text-gray-400 mb-2">
                ค้นหาพนักงานจากระบบ
              </label>

              <div className="relative">
                <input
                  value={employeeSearch}
                  onChange={(e) => {
                    const value = e.target.value;
                    setEmployeeSearch(value);
                    setEnrollMessage("");

                    if (selectedEmployee && value !== selectedEmployee.name) {
                      setSelectedEmployee(null);
                      setEnrollName("");
                    }
                  }}
                  disabled={enrollmentRef.current.active || employeeLoading}
                  placeholder={employeeLoading ? "กำลังโหลดรายชื่อพนักงาน..." : "พิมพ์ชื่อหรือรหัสพนักงาน..."}
                  className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 pr-10 text-sm text-white outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 disabled:opacity-40 transition-all"
                />

                <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-500">
                  🔎
                </div>

                {!enrollmentRef.current.active && employeeSearch.trim() && (
                  <div className="absolute left-0 right-0 top-full z-50 mt-2 max-h-64 overflow-y-auto rounded-xl border border-white/10 bg-slate-950/95 p-1 shadow-2xl backdrop-blur-xl">
                    {employees
                      .filter((employee) => {
                        const q = employeeSearch.trim().toLowerCase();
                        return (
                          employee.name.toLowerCase().includes(q) ||
                          employee.employee_code.toLowerCase().includes(q) ||
                          employee.first_name_th.toLowerCase().includes(q) ||
                          employee.last_name_th.toLowerCase().includes(q)
                        );
                      })
                      .slice(0, 30)
                      .map((employee) => (
                        <button
                          key={employee.id}
                          type="button"
                          onClick={() => {
                            setSelectedEmployee(employee);
                            setEnrollName(employee.name);
                            setEmployeeSearch(employee.name);
                            setEnrollMessage(`เลือก ${employee.name} • รหัส ${employee.employee_code || "-"}`);
                          }}
                          className="flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-cyan-400/10 transition-colors"
                        >
                          <div className="min-w-0">
                            <div className="truncate text-sm font-bold text-white">{employee.name}</div>
                            <div className="mt-0.5 truncate text-[10px] text-gray-500">
                              รหัส {employee.employee_code || "-"} • ID {employee.id}
                            </div>
                          </div>
                          <span className="shrink-0 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-2 py-1 text-[9px] font-bold text-cyan-300">
                            เลือก
                          </span>
                        </button>
                      ))}

                    {!employees.some((employee) => {
                      const q = employeeSearch.trim().toLowerCase();
                      return (
                        employee.name.toLowerCase().includes(q) ||
                        employee.employee_code.toLowerCase().includes(q) ||
                        employee.first_name_th.toLowerCase().includes(q) ||
                        employee.last_name_th.toLowerCase().includes(q)
                      );
                    }) && (
                      <div className="px-3 py-4 text-center text-xs text-gray-500">ไม่พบพนักงาน</div>
                    )}
                  </div>
                )}
              </div>

              {employeeError && (
                <div className="mt-2 flex items-center justify-between gap-2 rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-[10px] text-red-200">
                  <span>{employeeError}</span>
                  <button
                    type="button"
                    onClick={loadEmployees}
                    className="shrink-0 rounded-md bg-red-400/10 px-2 py-1 font-bold text-red-200 hover:bg-red-400/20"
                  >
                    โหลดใหม่
                  </button>
                </div>
              )}

              {selectedEmployee && (
                <div className="mt-2 rounded-xl border border-cyan-400/20 bg-cyan-400/5 px-3 py-2.5">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-black text-cyan-200">{selectedEmployee.name}</div>
                      <div className="mt-0.5 truncate text-[10px] text-gray-400">
                        รหัสพนักงาน: {selectedEmployee.employee_code || "-"} • employee_id: {selectedEmployee.id}
                      </div>
                    </div>
                    {!enrollmentRef.current.active && (
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedEmployee(null);
                          setEnrollName("");
                          setEmployeeSearch("");
                          setEnrollMessage("");
                        }}
                        className="shrink-0 rounded-lg bg-white/5 px-2.5 py-1.5 text-[10px] font-bold text-gray-300 hover:bg-white/10"
                      >
                        เปลี่ยน
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>

            {!enrollmentRef.current.active ? (
              <button
                type="button"
                onClick={openEnrollmentCamera}
                disabled={!selectedEmployee?.id || loading}
                className="w-full rounded-xl bg-gradient-to-r from-cyan-400 to-cyan-500 px-4 py-4 text-sm font-black text-black disabled:opacity-30 hover:from-cyan-300 hover:to-cyan-400 transition-all shadow-lg shadow-cyan-500/20"
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 002 2v8a2 2 0 002 2z" />
                  </svg>
                  {selectedEmployee?.id ? "เปิดกล้อง" : "เลือกพนักงานก่อนเปิดกล้อง"}
                </span>
              </button>
            ) : (
              <button
                type="button"
                onClick={cancelEnrollment}
                className="w-full rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-4 text-sm font-bold text-red-300 hover:bg-red-500/20 transition-all"
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  ยกเลิก
                </span>
              </button>
            )}

            {enrollmentStarted && enrollMessage && (
              <div className="rounded-xl bg-black/40 border border-cyan-400/20 p-4 text-center">
                <div className="flex items-center justify-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
                  <p className="text-xs text-cyan-200">{enrollMessage}</p>
                </div>
              </div>
            )}

            <div className="grid grid-cols-3 gap-3">
              {ENROLL_STAGES.map((stage, index) => (
                <div
                  key={stage.key}
                  className={`rounded-xl border p-4 text-center transition-all ${enrollStage > index
                      ? "border-green-400/40 bg-green-400/10 shadow-lg shadow-green-400/10"
                      : enrollStage === index
                        ? "border-cyan-400/40 bg-cyan-400/10 shadow-lg shadow-cyan-400/10"
                        : "border-white/5 bg-black/30"
                    }`}
                >
                  <div className={`flex items-center justify-center w-8 h-8 mx-auto rounded-full mb-2 ${enrollStage > index
                      ? "bg-green-400 text-black"
                      : enrollStage === index
                        ? "bg-cyan-400 text-black"
                        : "bg-white/10 text-gray-400"
                    }`}>
                    {enrollStage > index ? (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="text-sm font-black">{index + 1}</span>
                    )}
                  </div>

                  <div className="text-[10px] font-bold text-gray-300">
                    {stage.label}
                  </div>

                  {enrollStage === index && (
                    <div className="mt-2 text-[9px] text-cyan-300 font-mono">
                      {Math.min(2, enrollStable || 0)}/2
                    </div>
                  )}
                </div>
              ))}
            </div>

            {enrollmentRef.current.active && (
              <div className="mt-4 rounded-xl border border-cyan-400/20 bg-cyan-950/20 p-4">
                <div className="text-center">
                  <div className="text-lg font-black text-cyan-300">
                    {ENROLL_STAGES[enrollStage]?.label || "เตรียมตัว"}
                  </div>
                  <div className="mt-2 text-xs text-gray-300">
                    {enrollStage === 0 && "มองตรงเข้ากล้อง ให้ใบหน้าชัดและอยู่นิ่ง"}
                    {enrollStage === 1 && "หันหน้าไปทางซ้าย ให้เห็นแก้มซ้ายชัดเจน"}
                    {enrollStage === 2 && "หันหน้าไปทางขวา ให้เห็นแก้มขวาชัดเจน"}
                  </div>
                  <div className="mt-3 flex items-center justify-center gap-2">
                    <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
                    <div className="text-[10px] text-cyan-200">
                      {Math.min(2, enrollStable || 0)}/2 เฟรมยืนยัน
                    </div>
                  </div>
                </div>
              </div>
            )}

            {enrollmentStarted && enrollStage >= ENROLL_STAGES.length && (
              <div className="mt-4 rounded-xl border border-green-400/20 bg-green-950/20 p-4">
                <div className="text-center">
                  <div className="text-lg font-black text-green-300">
                    ✓ บันทึกสำเร็จ
                  </div>
                  <div className="mt-2 text-xs text-gray-300">
                    บันทึกข้อมูลใบหน้า {enrollName} เรียบร้อยแล้ว
                    <div className="mt-1 text-[10px] text-gray-500">
                      employee_id: {selectedEmployee?.id || "-"}
                    </div>
                  </div>
                  <div className="mt-3 flex items-center justify-center gap-2">
                    <div className="h-2 w-2 rounded-full bg-green-400" />
                    <div className="text-[10px] text-green-200">
                      พร้อมใช้งานในระบบตรวจสอบ
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="rounded-xl border border-white/5 bg-black/30 p-4">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                </div>
                <div className="text-[10px] text-gray-400 leading-relaxed">
                  <div className="font-bold text-gray-300 mb-1">ขั้นตอนการลงทะเบียน</div>
                  <div>1. ค้นหาพนักงาน → 2. หน้าตรง → 3. แก้มซ้าย → 4. แก้มขวา</div>
                  <div className="mt-1">3 มุมหลักเพียงพอสำหรับความแม่นยำสูงสุด</div>
                </div>
              </div>
            </div>

            {/* <div className="grid grid-cols-3 gap-3">
              <div className="rounded-xl bg-black/30 p-3 border border-white/5">
                <div className="text-2xl font-black text-cyan-300">{faceCount}</div>
                <div className="text-[9px] text-gray-400 mt-1">ใบหน้าที่พบ</div>
              </div>

              <div className="rounded-xl bg-black/30 p-3 border border-white/5">
                <div className="text-2xl font-black text-cyan-300">
                  {Math.min(enrollStage, ENROLL_STAGES.length)}/{ENROLL_STAGES.length}
                </div>
                <div className="text-[9px] text-gray-400 mt-1">มุมที่บันทึก</div>
              </div>

              <div className="rounded-xl bg-black/30 p-3 border border-white/5">
                <div className="text-2xl font-black text-cyan-300">
                  {Math.min(2, enrollStable || 0)}
                </div>
                <div className="text-[9px] text-gray-400 mt-1">เฟรมยืนยัน</div>
              </div>
            </div> */}

            {/* ปิด div หลักของส่วนลงทะเบียน */}
          </div>
        </section>

      </main>
    </div>
  );
}
