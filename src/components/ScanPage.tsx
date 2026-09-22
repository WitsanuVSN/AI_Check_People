// @ts-nocheck
import React, { useEffect, useRef, useState } from "react";
import { FaceDetector, FaceLandmarker, FilesetResolver, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import * as faceapi from "@vladmandic/face-api";

type LandmarkPoint = NormalizedLandmark;

const WASM_URL = "/wasm";
const MODEL_URL = "/models/face_detector_full_range.tflite";
const FACE_LANDMARKER_REMOTE = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const DEPTH_GRID_SIZE = 7;
const MAX_PEOPLE = 7;
const MIN_FACE_WIDTH = 16;
const MIN_FACE_HEIGHT = 16;
const FACE_PEOPLE_API = "/api/v1/face/people";
const FACE_GROUPS_API = "/api/v1/face/groups";
const FACE_GROUP_CREATE_API = "/api/v1/face/groups/create";
const FACE_GROUP_RENAME_API = "/api/v1/face/groups/rename";
const FACE_GROUP_DELETE_API = "/api/v1/face/groups/delete";
const FACE_GROUP_MEMBERS_API = "/api/v1/face/groups/members";
const FACE_GROUP_MEMBERS_SAVE_API = "/api/v1/face/groups/members/save";
const FACE_GROUP_MEMBER_DELETE_API = "/api/v1/face/groups/member/delete";
const FACE_GROUP_MEMBERS_CLEAR_API = "/api/v1/face/groups/members/clear";
const FACE_GROUP_DETECTION_API = "/api/v1/face/groups/detection";
const FACE_SCAN_HISTORY_KEY = "face_scan_history_v1";
const FACE_SCAN_SELECTION_KEY = "face_scan_selection_v2";
const MAX_SCAN_HISTORY = 40;
const MAX_SCAN_GALLERY_PER_PERSON = 3;
const MAX_APPEARANCE_IMAGES_PER_PERSON = 5;
const MAX_CONCURRENT_RECOGNITION = 4;
const SCAN_SNAPSHOT_INTERVAL_MS = 2200;
const LOCKED_IDENTITY_VERIFY_INTERVAL_MS = 900;
const LOCKED_IDENTITY_MAX_DISTANCE = 0.48;
const DUPLICATE_EXCLUSION_MS = 900;
const LOCKED_TRACK_MAX_GAP_MS = 1500;
const UNLOCKED_TRACK_MAX_GAP_MS = 900;
const LOCKED_IDENTITY_MAX_MISSES = 3;
const REPLACEMENT_SEARCH_MAX_DISTANCE = 0.50;
const REPLACEMENT_SEARCH_MAX_FACE_SCORE = 0.52;
const REPLACEMENT_SEARCH_MIN_MARGIN = 0.05;

type ScanPageProps = {
  onCameraStateChange?: (active: boolean) => void;
};

export default function ScanPage({ onCameraStateChange }: ScanPageProps) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const meshLastTimestampRef = useRef(0);
  const lastMeshResultRef = useRef(null);
  const depthProfilesRef = useRef({});
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const aiInitStartedRef = useRef(false);
  const aiInitPromiseRef = useRef(null);
  const tracksRef = useRef([]);
  const lastFrameTimestampRef = useRef(0);
  const lastDetectorRunRef = useRef(0);
  const recognitionActiveRef = useRef(0);
  const nextDetectionAtRef = useRef(0);
  const groupFacesRef = useRef([]);
  const faceMatcherRef = useRef(null);
  const faceSearchIndexRef = useRef([]);
  const scanHistoryRef = useRef([]);
  const recognizedSessionRef = useRef(new Map());
  const recentRecognizedPeopleRef = useRef([]);
  const RECOGNIZED_HOLD_MS = 30000;
  const cameraOnRef = useRef(false);
  const facingModeRef = useRef("environment");
  const smoothedBoxesRef = useRef({});
  const lastDetectedBoxesRef = useRef([]);
  const detectionHistoryRef = useRef({});
  const stabilityCounterRef = useRef({});
  const detectionPostRef = useRef(new Map());

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraId, setCameraId] = useState("");
  const [cameraLabel, setCameraLabel] = useState("กล้อง");
  const [loading, setLoading] = useState(true);
  const [faceCount, setFaceCount] = useState(0);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [groupFaces, setGroupFaces] = useState([]);
  const [facePeople, setFacePeople] = useState([]);
  const [facePeopleLoading, setFacePeopleLoading] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [scanMessage, setScanMessage] = useState("กำลังเตรียม AI ตรวจจับใบหน้า...");
  const [scanActivity, setScanActivity] = useState("กำลังเริ่มระบบตรวจจับใบหน้า...");
  const [cameraQuality, setCameraQuality] = useState("");
  const [cameraMirrored, setCameraMirrored] = useState(false);
  const [meshReady, setMeshReady] = useState(false);
  const [faceGeometry, setFaceGeometry] = useState(null);

  // กำหนดบุคคลตามลำดับในแถว: index 0 = คนที่ 1, index 1 = คนที่ 2 ...
  const [rowSlots, setRowSlots] = useState(3);
  const [selectedRowPeople, setSelectedRowPeople] = useState(() => Array(MAX_PEOPLE).fill(""));
  const [rowSearch, setRowSearch] = useState(() => Array(MAX_PEOPLE).fill(""));
  const [activeRowSearch, setActiveRowSearch] = useState(null);
  const [scanGroups, setScanGroups] = useState([]);
  const [selectedGroupId, setSelectedGroupId] = useState("");
  const [newGroupName, setNewGroupName] = useState("");
  const [groupSearch, setGroupSearch] = useState("");
  const [showFacePeopleModal, setShowFacePeopleModal] = useState(false);

  // แจ้ง App เมื่อเปิด/ปิดกล้อง เพื่อซ่อน BottomNav ขณะใช้กล้อง
  useEffect(() => {
    onCameraStateChange?.(cameraOn);
    return () => {
      onCameraStateChange?.(false);
    };
  }, [cameraOn, onCameraStateChange]);

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

  // เก็บ "กลุ่มที่เลือก + บุคคลที่เลือกพร้อม Face Descriptor" ไว้ใน LocalStorage
  // อ่านจาก LocalStorage ตอนเลือกกลุ่ม/เริ่มกล้องเท่านั้น ไม่อ่านทุก frame
  const loadFaceScanSelectionCache = () => {
    try {
      const raw = localStorage.getItem(FACE_SCAN_SELECTION_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return null;
      return {
        groupId: parsed.groupId ? String(parsed.groupId) : "",
        group: parsed.group || null,
        memberIds: Array.isArray(parsed.memberIds) ? parsed.memberIds.map(String) : [],
        people: Array.isArray(parsed.people) ? parsed.people.map(normalizePersonImages).filter((p) => p?.id && p?.descriptors?.length) : [],
        savedAt: parsed.savedAt || null,
      };
    } catch (error) {
      console.warn("⚠️ อ่าน Face scan cache ไม่สำเร็จ:", error);
      return null;
    }
  };

  const saveFaceScanSelectionCache = (group, people) => {
    try {
      const safePeople = (Array.isArray(people) ? people : [])
        .map(normalizePersonImages)
        .filter((person) => person?.id && person?.descriptors?.length)
        .map((person) => ({
          id: String(person.id),
          employee_code: person.employee_code || "",
          first_name_th: person.first_name_th || "",
          last_name_th: person.last_name_th || "",
          name: person.name || `${person.first_name_th || ""} ${person.last_name_th || ""}`.trim(),
          descriptors: person.descriptors,
          faceAngles: Array.isArray(person.faceAngles) ? person.faceAngles : [],
          // เก็บภาพใบหน้าที่ลงทะเบียนไว้ด้วย เพื่อให้ Scan ใช้งานจาก LocalStorage ได้
          // จำกัดจำนวนภาพเพื่อไม่ให้ quota ของ iPhone/Safari เต็มเร็วเกินไป
          faceImages: Array.isArray(person.faceImages)
            ? person.faceImages
                .filter((item) => item?.image)
                .slice(0, MAX_APPEARANCE_IMAGES_PER_PERSON)
                .map((item) => ({
                  stage: item.stage || item.angle || "front",
                  image: item.image,
                }))
            : [],
        }));

      const payload = {
        groupId: group?.id ? String(group.id) : "",
        group: group ? {
          id: String(group.id || ""),
          name: group.name || "",
          description: group.description || "",
        } : null,
        memberIds: safePeople.map((person) => String(person.id)),
        people: safePeople,
        savedAt: new Date().toISOString(),
      };

      localStorage.setItem(FACE_SCAN_SELECTION_KEY, JSON.stringify(payload));
      console.log("✅ Face scan selection saved:", {
        groupId: payload.groupId,
        people: payload.people.length,
      });
      return true;
    } catch (error) {
      console.warn("⚠️ บันทึก Face scan cache ไม่สำเร็จ:", error);
      return false;
    }
  };

  const clearFaceScanSelectionCache = () => {
    try {
      localStorage.removeItem(FACE_SCAN_SELECTION_KEY);
    } catch (error) {
      console.warn("⚠️ ลบ Face scan cache ไม่สำเร็จ:", error);
    }
  };

  const persistScanHistory = (items) => {
    const trimmed = items.slice(-MAX_SCAN_HISTORY);
    scanHistoryRef.current = trimmed;

    // ถ้า quota เต็ม ให้ลดภาพเก่าก่อน แต่ไม่แตะ group_faces
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

  // แปลง descriptor จาก localStorage ให้เป็น number[] เสมอ
  // รองรับทั้ง Array, Float32Array และ object ที่ JSON.stringify() ของ TypedArray สร้างไว้
  const toDescriptorArray = (value) => {
    if (!value) return null;

    if (Array.isArray(value)) {
      const arr = value.map(Number);
      return arr.length >= 64 && arr.every(Number.isFinite) ? arr : null;
    }

    if (ArrayBuffer.isView(value)) {
      const arr = Array.from(value, Number);
      return arr.length >= 64 && arr.every(Number.isFinite) ? arr : null;
    }

    if (typeof value === "object") {
      const keys = Object.keys(value)
        .filter((key) => /^\d+$/.test(key))
        .sort((a, b) => Number(a) - Number(b));

      if (keys.length >= 64) {
        const arr = keys.map((key) => Number(value[key]));
        return arr.every(Number.isFinite) ? arr : null;
      }
    }

    return null;
  };

  const normalizePersonImages = (person) => {
    const rawDescriptors =
      Array.isArray(person?.descriptors) && person.descriptors.length
        ? person.descriptors
        : person?.descriptor
          ? [person.descriptor]
          : [];

    const descriptors = rawDescriptors
      .map(toDescriptorArray)
      .filter(Boolean);

    let faceImages = Array.isArray(person?.faceImages)
      ? person.faceImages
        .filter((item) => item?.image)
        .map((item, index) => ({
          ...item,
          descriptor: toDescriptorArray(item?.descriptor) || descriptors[index] || null,
        }))
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

    const frontImage =
      person?.frontImage ||
      faceImages.find((item) => String(item?.stage || "").toLowerCase() === "front")?.image ||
      faceImages[0]?.image ||
      person?.image ||
      "";

    return {
      ...person,
      descriptors,
      faceImages,
      frontImage,
      image: person?.image || frontImage,
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


  useEffect(() => {
    let mounted = true;

    const loadInitialData = async () => {
      try {
        const [peopleResponse, groupsResponse] = await Promise.all([
          fetch(FACE_PEOPLE_API, { headers: { Accept: "application/json" }, cache: "no-store" }),
          fetch(FACE_GROUPS_API, { headers: { Accept: "application/json" }, cache: "no-store" }),
        ]);

        if (!peopleResponse.ok) throw new Error(`Face people HTTP ${peopleResponse.status}`);
        if (!groupsResponse.ok) throw new Error(`Groups HTTP ${groupsResponse.status}`);

        const peoplePayload = await peopleResponse.json();
        const groupsPayload = await groupsResponse.json();
        const peopleRows = Array.isArray(peoplePayload)
          ? peoplePayload
          : Array.isArray(peoplePayload?.data) ? peoplePayload.data : [];
        const groupsRows = Array.isArray(groupsPayload)
          ? groupsPayload
          : Array.isArray(groupsPayload?.data) ? groupsPayload.data : [];

        const apiFaces = peopleRows
          .map(normalizePersonImages)
          .filter((person) => person?.id && person?.descriptors?.length);
        const apiGroups = groupsRows
          .map(normalizeGroup)
          .filter((group) => group?.id);

        if (!mounted) return;
        setFacePeople(apiFaces);
        setScanGroups(apiGroups);
        scanHistoryRef.current = loadScanHistory();

        if (apiGroups.length) {
          const cachedSelection = loadFaceScanSelectionCache();
          const cachedGroup = cachedSelection?.groupId
            ? apiGroups.find((group) => String(group.id) === String(cachedSelection.groupId))
            : null;
          const initialGroup = cachedGroup || apiGroups[0];

          setSelectedGroupId(String(initialGroup.id));

          // ถ้ามี cache ของกลุ่มเดิม ให้เอา Face Data จาก LocalStorage เข้า memory ก่อน
          // เพื่อให้เปิดกล้องได้ทันทีโดยไม่ต้องยิง API ซ้ำ
          if (cachedGroup && cachedSelection?.people?.length) {
            const cachedPeople = cachedSelection.people
              .map((cachedPerson) => {
                const apiPerson = apiFaces.find((person) => String(person.id) === String(cachedPerson.id));
                // API เป็นแหล่งข้อมูลล่าสุด แต่ถ้า API ไม่มีรูป ให้ใช้รูปที่ cache ไว้
                return normalizePersonImages({
                  ...(apiPerson || {}),
                  ...cachedPerson,
                  descriptors: apiPerson?.descriptors?.length ? apiPerson.descriptors : cachedPerson.descriptors,
                  faceImages: apiPerson?.faceImages?.length ? apiPerson.faceImages : cachedPerson.faceImages,
                });
              })
              .filter((person) => person?.id && person?.descriptors?.length);
            const cachedIds = cachedPeople.map((person) => String(person.id));
            const cachedGroupWithMembers = { ...cachedGroup, memberIds: cachedIds, scanResults: {} };
            setScanGroups((prev) => prev.map((item) => String(item.id) === String(initialGroup.id) ? cachedGroupWithMembers : item));
            groupFacesRef.current = cachedPeople;
            setGroupFaces(cachedPeople);
            applyGroupToRow(cachedGroupWithMembers, cachedPeople);
            updateFaceMatcher(cachedPeople);
            setScanMessage(`โหลดกลุ่ม "${initialGroup.name}" จาก LocalStorage • Face Data ${cachedPeople.length} คนพร้อมค้นหา`);
          } else {
            await loadGroupMembersFromAPI(initialGroup.id, apiFaces, initialGroup);
          }
        } else {
          groupFacesRef.current = [];
          setGroupFaces([]);
        }
      } catch (error) {
        console.error("❌ Load face/group API failed:", error);
        if (!mounted) return;
        setFacePeople([]);
        setScanGroups([]);
        setGroupFaces([]);
        groupFacesRef.current = [];
      }
    };

    scanHistoryRef.current = loadScanHistory();
    loadInitialData();

    async function initAI() {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.40,
      });
      const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_LANDMARKER_REMOTE, delegate: "GPU" },
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
        updateFaceMatcher(groupFacesRef.current);
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
    const safeFaces = Array.isArray(faces)
      ? faces.map(normalizePersonImages).filter(Boolean)
      : [];

    if (!safeFaces.length) {
      faceMatcherRef.current = null;
      faceSearchIndexRef.current = [];
      return;
    }

    const labeled = [];
    const searchIndex = [];

    for (const person of safeFaces) {
      const descriptors = Array.isArray(person.descriptors)
        ? person.descriptors
          .map(toDescriptorArray)
          .filter(Boolean)
        : [];

      const faceImages = Array.isArray(person.faceImages)
        ? person.faceImages
        : [];

      const scanGallery = Array.isArray(person.scanGallery)
        ? person.scanGallery
        : [];

      // Descriptor จากหน้าที่ลงทะเบียน
      descriptors.forEach((descriptor, index) => {
        const imageEntry = faceImages[index];

        searchIndex.push({
          person,
          personId: String(person.id),
          vector: new Float32Array(descriptor),
          image: imageEntry?.image || null,
          angle: imageEntry?.stage || person.faceAngles?.[index] || null,
          source: "enrollment",
        });
      });

      const personVectors = searchIndex
        .filter((item) => item.personId === String(person.id))
        .map((item) => item.vector);

      if (personVectors.length) {
        labeled.push(
          new faceapi.LabeledFaceDescriptors(
            String(person.id),
            personVectors
          )
        );
      }
    }

    faceSearchIndexRef.current = searchIndex;

    // ใช้ไว้เป็น fallback เท่านั้น แต่การค้นหาหลักด้านล่างจะอ่าน index โดยตรง
    faceMatcherRef.current = labeled.length
      ? new faceapi.FaceMatcher(labeled, 0.60)
      : null;

    console.log("✅ Face search index ready:", {
      people: safeFaces.length,
      vectors: searchIndex.length,
      names: safeFaces.map((p) => p.name),
    });
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

    // Preserve where the original MediaPipe/Detector target face center landed inside the crop.
    // This is critical when another person's face is also visible inside the enlarged crop.
    canvas.__targetFaceCenter = {
      x: ((cx - x) / Math.max(1, side)) * output,
      y: ((cy - y) / Math.max(1, side)) * output,
    };

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
  // Face API ต้องเลือก "หน้าที่อยู่ตรงกลาง crop" ไม่ใช่ใช้ detectSingleFace()
  // เพราะ crop ของคนหนึ่งอาจมีใบหน้าคนอื่นติดเข้ามาด้วย และ detectSingleFace()
  // อาจเลือกคนอื่นแทนเป้าหมายได้
  const detectWithMultipleModels = async (source) => {
    try {
      let input = source;
      let targetFaceCenter = source?.__targetFaceCenter || null;

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
            ctx.drawImage(source, 0, 0, upscaled.width, upscaled.height);
            if (targetFaceCenter) {
              targetFaceCenter = {
                x: targetFaceCenter.x * (upscaled.width / Math.max(1, source.width)),
                y: targetFaceCenter.y * (upscaled.height / Math.max(1, source.height)),
              };
              upscaled.__targetFaceCenter = targetFaceCenter;
            }
            input = upscaled;
          }
        }
      }

      const pickCenterFace = (results) => {
        if (!results?.length) return null;

        const cx = targetFaceCenter?.x ?? (input.width || 640) / 2;
        const cy = targetFaceCenter?.y ?? (input.height || 480) / 2;
        const maxCenterDistance = Math.min(input.width || 640, input.height || 480) * 0.42;

        const ranked = results
          .map((item) => {
            const box = item.detection?.box;
            if (!box) return null;
            const bx = box.x + box.width / 2;
            const by = box.y + box.height / 2;
            const centerDistance = Math.hypot(bx - cx, by - cy);
            const normalizedDistance = centerDistance / Math.max(1, Math.min(input.width || 640, input.height || 480));
            const centerInside =
              cx >= box.x - box.width * 0.35 &&
              cx <= box.x + box.width * 1.35 &&
              cy >= box.y - box.height * 0.35 &&
              cy <= box.y + box.height * 1.35;
            const confidencePenalty = 1 - (item.detection?.score || 0);
            return {
              item,
              centerDistance,
              centerInside,
              rankScore:
                (centerInside ? 0 : 1.0) +
                normalizedDistance * 1.4 +
                confidencePenalty * 0.08,
            };
          })
          .filter(Boolean)
          .sort((a, b) => {
            if (a.centerDistance !== b.centerDistance) {
              return a.centerDistance - b.centerDistance;
            }
            return a.rankScore - b.rankScore;
          });

        const winner = ranked[0];
        if (!winner || winner.centerDistance > maxCenterDistance) return null;
        return winner.item;
      };

      const runDetector = async (options, model) => {
        const results = await faceapi
          .detectAllFaces(input, options)
          .withFaceLandmarks()
          .withFaceDescriptors();

        if (results.length > 1) {
          console.warn("⚠️ Face-API crop มีหลายใบหน้า เลือกหน้าที่ใกล้ target center", results.map((item) => ({
            score: Number((item.detection?.score || 0).toFixed(3)),
            x: Math.round(item.detection?.box?.x || 0),
            y: Math.round(item.detection?.box?.y || 0),
            w: Math.round(item.detection?.box?.width || 0),
            h: Math.round(item.detection?.box?.height || 0),
          })));
        }

        const best = pickCenterFace(results);
        return best ? { ...best, model } : null;
      };

      const tiny = await runDetector(
        new faceapi.TinyFaceDetectorOptions({
          inputSize: 416,
          scoreThreshold: 0.03,
        }),
        "tiny"
      );

      if (tiny) return tiny;

      const ssd = await runDetector(
        new faceapi.SsdMobilenetv1Options({
          minConfidence: 0.15,
        }),
        "ssd"
      );

      if (ssd) return ssd;
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
  // ข้อมูลกลุ่มมาจาก API และถูกเก็บไว้ใน Memory ระหว่าง Scan

  const recognizeWithSavedImages = async (currentDetection, people, excludedPersonIds = new Set()) => {
    if (!currentDetection || !people.length) return null;

    const query = toDescriptorArray(currentDetection.descriptor);
    const index = faceSearchIndexRef.current;

    if (!query || query.length !== 128) {
      console.warn("⚠️ Query descriptor ไม่ใช่ 128 ค่า:", query?.length);
      return null;
    }

    if (!index.length) {
      console.warn("⚠️ ไม่มี descriptor ที่ใช้งานได้ใน API group face data");
      return null;
    }

    const byPerson = new Map();

    for (const item of index) {
      if (!item?.vector || item.vector.length !== query.length) continue;
      const distance = faceapi.euclideanDistance(
        query,
        item.vector
      );
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
      if (excludedPersonIds?.has(String(group.person?.id))) continue;
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
      const aggregateScore = faceScore;

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

    if (!scores.length) return null;

    const winner = scores[0];
    const runner = scores[1] ?? null;
    const margin = runner ? runner.aggregateScore - winner.aggregateScore : Infinity;

    return {
      person: winner.person,
      distance: winner.best,
      secondDistance: winner.second,
      faceScore: winner.faceScore,
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

      // ระบุตัวตนจาก Face Descriptor ใน LocalStorage โดยตรง
      // ถ้ามีคนเดียว ไม่ต้องใช้ margin เพราะไม่มีคนอื่นให้เปรียบเทียบ
      // ถ้ามีหลายคน ต้องมีระยะห่างขั้นต่ำเพื่อไม่เอาคนด้านหลังไปชื่อผิด
      confidentEnough:
        winner.best <= 0.55 &&
        winner.faceScore <= 0.57 &&
        (!runner || margin >= 0.025) &&
        winner.support >= 1,
    };
  };


  const fetchJSON = async (url, options = {}) => {
    const response = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload?.message || payload?.error || `HTTP ${response.status}`);
    }
    return payload;
  };

  const getPayloadRows = (payload) =>
    Array.isArray(payload)
      ? payload
      : (Array.isArray(payload?.data) ? payload.data : []);

  // Normalize Go API JSON to the frontend's internal camelCase shape.
  // รองรับทั้ง JSON ที่มี json tags แล้ว (id/name) และ struct JSON default (ID/Name).
  const normalizeGroup = (group) => {
    if (!group) return null;

    return {
      ...group,
      id: String(group?.id ?? group?.ID ?? ""),
      name: String(group?.name ?? group?.Name ?? ""),
      description: String(group?.description ?? group?.Description ?? ""),
      createdAt: group?.createdAt ?? group?.CreatedAt ?? null,
      updatedAt: group?.updatedAt ?? group?.UpdatedAt ?? null,
      memberIds: Array.isArray(group?.memberIds)
        ? group.memberIds.map(String)
        : Array.isArray(group?.MemberIDs)
          ? group.MemberIDs.map(String)
          : [],
      scanResults: group?.scanResults || {},
    };
  };

  const normalizeGroupMember = (member) => {
    if (!member) return null;
    return {
      ...member,
      id: String(member?.id ?? member?.ID ?? ""),
      group_id: String(member?.group_id ?? member?.GroupID ?? ""),
      employee_id: String(member?.employee_id ?? member?.EmployeeID ?? ""),
      employee_code: member?.employee_code ?? member?.EmployeeCode ?? "",
      first_name_th: member?.first_name_th ?? member?.FirstNameTH ?? "",
      last_name_th: member?.last_name_th ?? member?.LastNameTH ?? "",
      name: member?.name ?? member?.Name ?? "",
      joined_at: member?.joined_at ?? member?.JoinedAt ?? null,
      last_detected_at: member?.last_detected_at ?? member?.LastDetectedAt ?? null,
      daily_detections: member?.daily_detections ?? member?.DailyDetections ?? {},
    };
  };

  const loadGroupsFromAPI = async () => {
    const payload = await fetchJSON(FACE_GROUPS_API);
    const groups = getPayloadRows(payload)
      .map(normalizeGroup)
      .filter((group) => group?.id);
    setScanGroups(groups);
    return groups;
  };

  const loadGroupMembersFromAPI = async (groupId, peopleOverride = null, groupOverride = null) => {
    if (!groupId) {
      groupFacesRef.current = [];
      setGroupFaces([]);
      updateFaceMatcher([]);
      return [];
    }

    try {
      const payload = await fetchJSON(`${FACE_GROUP_MEMBERS_API}?group_id=${encodeURIComponent(groupId)}`);
      const rows = getPayloadRows(payload)
        .map(normalizeGroupMember)
        .filter(Boolean);
      const people = Array.isArray(peopleOverride) && peopleOverride.length ? peopleOverride : facePeople;
      const byId = new Map(people.map((person) => [String(person.id), person]));
      const members = rows
        .map((member) => byId.get(String(member.employee_id)) || byId.get(String(member.id)))
        .filter(Boolean)
        .map(normalizePersonImages);

      const memberIds = rows
        .map((member) => String(member.employee_id || member.id))
        .filter(Boolean);
      const group = groupOverride || scanGroups.find((item) => String(item.id) === String(groupId));
      const nextGroup = group
        ? { ...group, memberIds, scanResults: group.scanResults || {} }
        : { id: String(groupId), memberIds, scanResults: {} };

      setScanGroups((prev) => prev.map((item) => String(item.id) === String(groupId) ? nextGroup : item));
      groupFacesRef.current = members;
      setGroupFaces(members);
      applyGroupToRow(nextGroup, members);
      updateFaceMatcher(members);
      return members;
    } catch (error) {
      console.error("❌ Load group members failed:", error);
      groupFacesRef.current = [];
      setGroupFaces([]);
      updateFaceMatcher([]);
      return [];
    }
  };

  const applyGroupToRow = (group, peopleOverride = null) => {
    const ids = Array(MAX_PEOPLE).fill("");
    const names = Array(MAX_PEOPLE).fill("");
    const people = Array.isArray(peopleOverride) ? peopleOverride : groupFacesRef.current;
    (group?.memberIds || []).slice(0, MAX_PEOPLE).forEach((id, index) => {
      const person = people.find((p) => String(p.id) === String(id));
      ids[index] = String(id);
      names[index] = person?.name || "";
    });
    setSelectedRowPeople(ids);
    setRowSearch(names);
    setRowSlots(Math.max(1, Math.min(MAX_PEOPLE, group?.memberIds?.length || 1)));
  };

  const createScanGroup = async () => {
    const name = newGroupName.trim();
    if (!name) return;
    try {
      const payload = await fetchJSON(FACE_GROUP_CREATE_API, {
        method: "POST",
        body: JSON.stringify({ name, description: "" }),
      });
      const createdRaw = payload?.data || payload?.group || payload;
      const created = normalizeGroup(createdRaw);
      if (!created?.id) throw new Error("API ไม่ได้ส่ง group id กลับมา");
      const group = { ...created, memberIds: [], scanResults: {} };
      setScanGroups((prev) => [...prev, group]);
      setSelectedGroupId(String(group.id));
      setNewGroupName("");
      setGroupFaces([]);
      groupFacesRef.current = [];
      updateFaceMatcher([]);
      applyGroupToRow(group, []);
      saveFaceScanSelectionCache(group, []);
      setScanMessage(`สร้างกลุ่ม "${group.name}" แล้ว • บันทึกกลุ่มลง LocalStorage`);
    } catch (error) {
      console.error("❌ Create group failed:", error);
      setScanMessage(`สร้างกลุ่มไม่สำเร็จ: ${error.message}`);
    }
  };

  const selectScanGroup = async (groupId) => {
    const group = scanGroups.find((item) => String(item.id) === String(groupId));
    const nextId = group ? String(group.id) : "";
    setSelectedGroupId(nextId);
    setGroupSearch("");
    if (!nextId) {
      setSelectedRowPeople(Array(MAX_PEOPLE).fill(""));
      setRowSearch(Array(MAX_PEOPLE).fill(""));
      setRowSlots(1);
      groupFacesRef.current = [];
      setGroupFaces([]);
      updateFaceMatcher([]);
      return;
    }
    setScanMessage(`กำลังโหลดสมาชิกกลุ่ม "${group.name}"...`);
    const members = await loadGroupMembersFromAPI(nextId);
    saveFaceScanSelectionCache({ ...group, memberIds: members.map((person) => String(person.id)) }, members);
    setScanMessage(`เลือกกลุ่ม "${group.name}" • บันทึก Face Data ${members.length} คนลง LocalStorage แล้ว`);
  };

  const deleteScanGroup = async (groupId) => {
    if (!groupId) return;
    if (!confirm("ต้องการลบกลุ่มนี้หรือไม่? สมาชิกใน Face Database จะไม่ถูกลบ")) return;
    try {
      await fetchJSON(FACE_GROUP_DELETE_API, {
        method: "DELETE",
        body: JSON.stringify({ group_id: groupId }),
      });
      setScanGroups((prev) => prev.filter((item) => String(item.id) !== String(groupId)));
      if (String(selectedGroupId) === String(groupId)) {
        setSelectedGroupId("");
        setGroupFaces([]);
        groupFacesRef.current = [];
        setSelectedRowPeople(Array(MAX_PEOPLE).fill(""));
        setRowSearch(Array(MAX_PEOPLE).fill(""));
        setRowSlots(1);
        updateFaceMatcher([]);
        clearFaceScanSelectionCache();
      }
    } catch (error) {
      console.error("❌ Delete group failed:", error);
      setScanMessage(`ลบกลุ่มไม่สำเร็จ: ${error.message}`);
    }
  };

  const saveGroupMemberIds = async (groupId, memberIds) => {
    const ids = Array.from(new Set((memberIds || []).map(String))).slice(0, MAX_PEOPLE);
    await fetchJSON(FACE_GROUP_MEMBERS_SAVE_API, {
      method: "POST",
      body: JSON.stringify({ group_id: groupId, employee_ids: ids }),
    });
    return ids;
  };

  const removePersonFromSelectedGroup = async (personId) => {
    if (!selectedGroupId || !personId) return;
    try {
      await fetchJSON(FACE_GROUP_MEMBER_DELETE_API, {
        method: "DELETE",
        body: JSON.stringify({ group_id: selectedGroupId, employee_id: personId }),
      });
      const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
      const nextIds = (group?.memberIds || []).filter((id) => String(id) !== String(personId));
      const nextGroup = { ...(group || { id: selectedGroupId }), memberIds: nextIds, scanResults: {} };
      setScanGroups((prev) => prev.map((item) => String(item.id) === String(selectedGroupId) ? nextGroup : item));
      const nextFaces = groupFacesRef.current.filter((person) => String(person.id) !== String(personId));
      groupFacesRef.current = nextFaces;
      setGroupFaces(nextFaces);
      applyGroupToRow(nextGroup, nextFaces);
      updateFaceMatcher(nextFaces);
      saveFaceScanSelectionCache(nextGroup, nextFaces);
    } catch (error) {
      console.error("❌ Delete group member failed:", error);
      setScanMessage(`ลบบุคคลไม่สำเร็จ: ${error.message}`);
    }
  };

  const clearCurrentGroupMembers = async () => {
    if (!selectedGroupId) return;
    if (!confirm("ต้องการเคลียร์สมาชิกทั้งหมดในกลุ่มนี้หรือไม่?")) return;
    try {
      await fetchJSON(FACE_GROUP_MEMBERS_CLEAR_API, {
        method: "DELETE",
        body: JSON.stringify({ group_id: selectedGroupId }),
      });
      setScanGroups((prev) => prev.map((item) => String(item.id) === String(selectedGroupId) ? { ...item, memberIds: [], scanResults: {} } : item));
      groupFacesRef.current = [];
      setGroupFaces([]);
      setSelectedRowPeople(Array(MAX_PEOPLE).fill(""));
      setRowSearch(Array(MAX_PEOPLE).fill(""));
      setRowSlots(1);
      updateFaceMatcher([]);
      const clearedGroup = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
      if (clearedGroup) saveFaceScanSelectionCache({ ...clearedGroup, memberIds: [] }, []);
    } catch (error) {
      console.error("❌ Clear group members failed:", error);
      setScanMessage(`เคลียร์สมาชิกไม่สำเร็จ: ${error.message}`);
    }
  };

  const addPersonToSelectedGroup = async (personId) => {
    if (!selectedGroupId) return;
    const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
    if (!group) return;
    const ids = (group.memberIds || []).map(String);
    if (ids.includes(String(personId)) || ids.length >= MAX_PEOPLE) return;
    const apiPerson = facePeople.find((person) => String(person.id) === String(personId));
    if (!apiPerson) return;
    try {
      const nextIds = await saveGroupMemberIds(selectedGroupId, [...ids, String(personId)]);
      const nextGroup = { ...group, memberIds: nextIds, scanResults: {} };
      setScanGroups((prev) => prev.map((item) => String(item.id) === String(group.id) ? nextGroup : item));
      const nextFaces = nextIds.map((id) => facePeople.find((person) => String(person.id) === String(id))).filter(Boolean).map(normalizePersonImages);
      groupFacesRef.current = nextFaces;
      setGroupFaces(nextFaces);
      applyGroupToRow(nextGroup, nextFaces);
      setActiveRowSearch(null);
      updateFaceMatcher(nextFaces);
      saveFaceScanSelectionCache(nextGroup, nextFaces);
    } catch (error) {
      console.error("❌ Add group member failed:", error);
      setScanMessage(`เพิ่มบุคคลไม่สำเร็จ: ${error.message}`);
    }
  };

  const moveGroupMember = async (personId, direction) => {
    if (!selectedGroupId) return;
    const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
    if (!group) return;
    const ids = [...(group.memberIds || [])];
    const index = ids.findIndex((id) => String(id) === String(personId));
    const target = index + direction;
    if (index < 0 || target < 0 || target >= ids.length) return;
    [ids[index], ids[target]] = [ids[target], ids[index]];
    try {
      await saveGroupMemberIds(selectedGroupId, ids);
      const nextGroup = { ...group, memberIds: ids, scanResults: {} };
      setScanGroups((prev) => prev.map((item) => String(item.id) === String(group.id) ? nextGroup : item));
      applyGroupToRow(nextGroup, groupFacesRef.current);
      saveFaceScanSelectionCache(nextGroup, groupFacesRef.current);
    } catch (error) {
      console.error("❌ Reorder group member failed:", error);
    }
  };

  const renameScanGroup = async (groupId, name) => {
    const trimmed = String(name || "").trim();
    if (!trimmed) return;
    try {
      await fetchJSON(FACE_GROUP_RENAME_API, {
        method: "POST",
        body: JSON.stringify({ group_id: groupId, name: trimmed }),
      });
      const renamedGroup = scanGroups.find((item) => String(item.id) === String(groupId));
      const nextGroup = renamedGroup
        ? { ...renamedGroup, name: trimmed }
        : { id: String(groupId), name: trimmed, memberIds: [] };
      setScanGroups((prev) => prev.map((item) => String(item.id) === String(groupId) ? { ...item, name: trimmed } : item));
      if (String(selectedGroupId) === String(groupId)) {
        saveFaceScanSelectionCache(nextGroup, groupFacesRef.current);
      }
    } catch (error) {
      console.error("❌ Rename group failed:", error);
    }
  };

  const updateGroupScanResult = (personId, status, detectedName = null, box = null) => {
    if (!selectedGroupId || !personId) return;
    const key = String(personId);
    const now = new Date().toISOString();
    setScanGroups((prev) => prev.map((group) => {
      if (String(group.id) !== String(selectedGroupId)) return group;
      const previous = group.scanResults?.[key] || null;
      return {
        ...group,
        scanResults: {
          ...(group.scanResults || {}),
          [key]: {
            status,
            personId: key,
            detectedName: detectedName || previous?.detectedName || null,
            checked: status === "matched",
            checkedAt: status === "matched" ? (previous?.checkedAt || now) : previous?.checkedAt || null,
            lastSeenAt: now,
            lockedAt: status === "matched" ? (previous?.lockedAt || now) : previous?.lockedAt || null,
            lastBox: box ? { originX: Number(box.originX || 0), originY: Number(box.originY || 0), width: Number(box.width || 0), height: Number(box.height || 0) } : previous?.lastBox || null,
          },
        },
      };
    }));

    if (status === "matched") {
      const lastPosted = detectionPostRef.current.get(key) || 0;
      if (Date.now() - lastPosted >= DUPLICATE_EXCLUSION_MS) {
        detectionPostRef.current.set(key, Date.now());
        fetchJSON(FACE_GROUP_DETECTION_API, {
          method: "POST",
          body: JSON.stringify({ group_id: selectedGroupId, employee_id: key, detected_at: now }),
        }).catch((error) => console.warn("⚠️ บันทึก detection API ไม่สำเร็จ:", error));
      }
    }
  };

  const resetSelectedGroupScanResults = () => {
    if (!selectedGroupId) return;
    setScanGroups((prev) => prev.map((item) => String(item.id) === String(selectedGroupId) ? { ...item, scanResults: {} } : item));
  };

  const updateName = (id, newName) => {
    const updated = groupFaces.map(f => f.id === id ? { ...f, name: newName } : f);
    groupFacesRef.current = updated;
    setGroupFaces(updated);
    updateFaceMatcher(updated);
  };

  const clearCurrentGroupFaces = clearCurrentGroupMembers;

  const formatScanTime = (iso) => {
    if (!iso) return '--:--:--';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '--:--:--';
    return date.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  };


  async function refreshCameraDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
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

    // เปิดกล้องแล้วใช้ Face Data จาก LocalStorage เป็นฐานค้นหา
    // โหลดเข้า memory ครั้งเดียว ไม่อ่าน LocalStorage ทุก frame
    const cachedSelection = loadFaceScanSelectionCache();
    if (selectedGroupId && cachedSelection && String(cachedSelection.groupId) === String(selectedGroupId)) {
      const cachedPeople = (cachedSelection.people || [])
        .map(normalizePersonImages)
        .filter((person) => person?.id && person?.descriptors?.length);
      groupFacesRef.current = cachedPeople;
      setGroupFaces(cachedPeople);
      updateFaceMatcher(cachedPeople);
      setScanMessage(`📦 ใช้ LocalStorage • ${cachedPeople.length} คนพร้อมค้นหา`);
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
          width: { ideal: 1280, max: 1280, min: 640 },
          height: { ideal: 720, max: 720, min: 360 },
          frameRate: { ideal: 24, max: 24 },
        }
        : {
          facingMode: { ideal: facingModeRef.current },
          width: { ideal: 1280, max: 1280, min: 640 },
          height: { ideal: 720, max: 720, min: 360 },
          frameRate: { ideal: 24, max: 24 },
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
    if (cameraOnRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }

      cameraOnRef.current = false;
      setCameraOn(false);
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
      setScanMessage("AI พร้อมใช้งาน กดที่ภาพเพื่อเริ่มสแกน");
      setScanActivity("กล้องปิด • พร้อมเริ่มสแกนใหม่");

      const ctx = canvasRef.current?.getContext("2d");
      if (ctx && canvasRef.current) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      return;
    }

    await startCamera(cameraId || "");
  }



  // เมื่อเจอคนแล้ว ให้จำผลระดับ session และแสดงชื่อ/กรอบเขียวต่อเนื่อง
  // แม้ detector จะหลุดชั่วคราว โดยไม่ต้องส่งคนเดิมกลับไปค้นใหม่
  const rememberRecognizedPerson = (person, box, distance = null, source = "group_faces") => {
    if (!person?.id || !box) return;
    const now = performance.now();
    const centerX = (box.originX || 0) + (box.width || 0) / 2;
    const centerY = (box.originY || 0) + (box.height || 0) / 2;
    const existing = recentRecognizedPeopleRef.current.find(
      (item) => String(item.personId) === String(person.id)
    );
    const value = {
      personId: person.id,
      name: person.name,
      box: { ...box },
      centerX,
      centerY,
      width: box.width,
      height: box.height,
      lastSeen: now,
      distance,
      source,
      descriptors: (Array.isArray(person.descriptors) ? person.descriptors : [])
        .map(toDescriptorArray)
        .filter(Boolean),
    };
    if (existing) Object.assign(existing, value);
    else recentRecognizedPeopleRef.current.push(value);
  };


  // IMPORTANT: never restore an identity from position alone.
  // The new face must match the remembered person's enrollment descriptors first.
  const findRecentRecognizedPersonByDescriptor = (
    descriptor,
    box,
    usedPersonIds = new Set()
  ) => {
    const query = toDescriptorArray(descriptor);
    if (!query || query.length !== 128) return null;

    const now = performance.now();
    recentRecognizedPeopleRef.current = recentRecognizedPeopleRef.current.filter(
      (item) => now - item.lastSeen <= RECOGNIZED_HOLD_MS
    );

    const candidates = recentRecognizedPeopleRef.current
      .filter((item) => !usedPersonIds.has(String(item.personId)))
      .map((item) => {
        const distances = (item.descriptors || [])
          .map(toDescriptorArray)
          .filter((vector) => vector && vector.length === query.length)
          .map((vector) => faceapi.euclideanDistance(query, new Float32Array(vector)))
          .sort((a, b) => a - b);
        return {
          item,
          bestDistance: distances[0] ?? Infinity,
          secondDistance: distances[1] ?? Infinity,
        };
      })
      .filter((candidate) =>
        Number.isFinite(candidate.bestDistance) &&
        candidate.bestDistance <= LOCKED_IDENTITY_MAX_DISTANCE &&
        (
          !Number.isFinite(candidate.secondDistance) ||
          candidate.secondDistance - candidate.bestDistance >= 0.025
        )
      )
      .sort((a, b) => a.bestDistance - b.bestDistance);

    const winner = candidates[0];
    if (!winner) return null;

    winner.item.lastSeen = now;
    if (box) {
      winner.item.box = { ...box };
      winner.item.centerX = (box.originX || 0) + (box.width || 0) / 2;
      winner.item.centerY = (box.originY || 0) + (box.height || 0) / 2;
      winner.item.width = box.width;
      winner.item.height = box.height;
    }

    return {
      ...winner.item,
      distance: winner.bestDistance,
    };
  };

  const findRecentRecognizedPerson = (box, usedPersonIds = new Set()) => {
    if (!box) return null;
    const now = performance.now();
    const centerX = (box.originX || 0) + (box.width || 0) / 2;
    const centerY = (box.originY || 0) + (box.height || 0) / 2;

    recentRecognizedPeopleRef.current = recentRecognizedPeopleRef.current.filter(
      (item) => now - item.lastSeen <= RECOGNIZED_HOLD_MS
    );

    const candidates = recentRecognizedPeopleRef.current
      .filter((item) => !usedPersonIds.has(String(item.personId)))
      .map((item) => ({
        item,
        distance: Math.hypot(centerX - item.centerX, centerY - item.centerY),
      }))
      .filter(({ item, distance }) => {
        const size = Math.max(
          item.width || 0,
          item.height || 0,
          box.width || 0,
          box.height || 0,
          1
        );
        return distance <= size * 1.5;
      })
      .sort((a, b) => a.distance - b.distance);

    const winner = candidates[0]?.item || null;
    if (!winner) return null;

    winner.lastSeen = now;
    winner.box = { ...box };
    winner.centerX = centerX;
    winner.centerY = centerY;
    winner.width = box.width;
    winner.height = box.height;
    return winner;
  };


  // ตรวจว่ากรอบ 2 กรอบน่าจะเป็นใบหน้าเดียวกันหรือไม่
  // ใช้เฉพาะเพื่อจัดการกรณี detector สร้างหน้าเดียวซ้ำในเฟรมเดียว
  const boxesLikelySameFace = (a, b) => {
    if (!a || !b) return false;

    const ax1 = a.originX || 0;
    const ay1 = a.originY || 0;
    const ax2 = ax1 + (a.width || 0);
    const ay2 = ay1 + (a.height || 0);
    const bx1 = b.originX || 0;
    const by1 = b.originY || 0;
    const bx2 = bx1 + (b.width || 0);
    const by2 = by1 + (b.height || 0);

    const intersection = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1)) *
      Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
    const areaA = Math.max(1, (a.width || 0) * (a.height || 0));
    const areaB = Math.max(1, (b.width || 0) * (b.height || 0));
    const smallerArea = Math.min(areaA, areaB);
    const largerArea = Math.max(areaA, areaB);
    const union = areaA + areaB - intersection;
    const iou = union > 0 ? intersection / union : 0;
    const containment = intersection / smallerArea;

    const acx = ax1 + (a.width || 0) / 2;
    const acy = ay1 + (a.height || 0) / 2;
    const bcx = bx1 + (b.width || 0) / 2;
    const bcy = by1 + (b.height || 0) / 2;
    const centerDistance = Math.hypot(acx - bcx, acy - bcy);
    const centerLimit = Math.max(1, Math.min(a.width || 0, a.height || 0, b.width || 0, b.height || 0)) * 0.65;

    return (
      iou >= 0.25 ||
      containment >= 0.60 ||
      (centerDistance <= centerLimit && smallerArea / largerArea >= 0.55)
    );
  };

  // การกัน personId ซ้ำใช้ชั่วคราวเท่านั้น
  // ห้ามจำ exclusion ถาวร เพราะเมื่อคนเดิมกลับเข้ากล้องต้องค้นหาเจออีกครั้ง
  const expireTrackExclusions = (track) => {
    if (!track?.excludedPersonIds) return;
    if (performance.now() >= (track.excludedPersonIdsUntil || 0)) {
      track.excludedPersonIds.clear();
      track.excludedPersonIdsUntil = 0;
    }
  };

  // ถ้า detector สร้างหน้าเดียวกันซ้ำ และทั้งสอง track ได้ชื่อเดียวกัน
  // ให้เลือกผลที่มี distance ต่ำกว่า (ตรงกับ group_faces มากกว่า)
  // อีก track ถูกปลดผลแล้วกลับไปค้นใหม่
  const resolveDuplicateRecognition = (track) => {
    if (!track?.recognizedPersonId) return true;

    const currentBox = track.lastRecognizedBox || track.detection?.boundingBox;
    if (!currentBox) return true;

    const sameIdentityTracks = tracksRef.current.filter((other) => {
      if (!other || other.id === track.id) return false;
      if (!other.recognizedPersonId) return false;
      if (String(other.recognizedPersonId) !== String(track.recognizedPersonId)) return false;
      if (performance.now() - (other.lastSeen || 0) > RECOGNIZED_HOLD_MS) return false;

      const otherBox = other.lastRecognizedBox || other.detection?.boundingBox;
      return boxesLikelySameFace(currentBox, otherBox);
    });

    if (!sameIdentityTracks.length) return true;

    const candidates = [track, ...sameIdentityTracks];
    candidates.sort((a, b) => {
      const ad = Number.isFinite(a.matchDistance) ? a.matchDistance : Infinity;
      const bd = Number.isFinite(b.matchDistance) ? b.matchDistance : Infinity;
      if (ad !== bd) return ad - bd;
      return (b.width || 0) * (b.height || 0) - (a.width || 0) * (a.height || 0);
    });

    const winner = candidates[0];

    for (const loser of candidates.slice(1)) {
      loser.found = false;
      loser.lockedResult = false;
      loser.recognizedPersonId = null;
      loser.identityName = null;
      loser.matchedAngle = null;
      loser.matchDistance = null;
      loser.recognitionMethod = null;
      loser.statusLabel = "กำลังค้นหาใหม่...";
      loser.color = "#f59e0b";
      loser.lastRecognition = 0;
      loser.lastRecognizedBox = null;
      loser.isRecognizing = false;
      if (!loser.excludedPersonIds) loser.excludedPersonIds = new Set();
      loser.excludedPersonIds.add(String(winner.recognizedPersonId));
      loser.excludedPersonIdsUntil = performance.now() + DUPLICATE_EXCLUSION_MS;
      loser.replacementSearch = true;
    }

    return winner.id === track.id;
  };


  // จำกัดการค้นหาเฉพาะสมาชิกของกลุ่มที่เลือก
  // ไม่ฟิกตำแหน่งล่วงหน้า: เจอใครตรงไหน ให้ล็อกคนนั้นไว้กับ Track ตรงนั้น
  const getScopedRecognitionPeople = (allPeople, excludedIds = new Set()) => {
    const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
    if (!group) return { groupPeople: [] };

    const memberIds = new Set((group.memberIds || []).map(String));
    const groupPeople = allPeople.filter((person) =>
      memberIds.has(String(person.id)) && !excludedIds.has(String(person.id))
    );

    return { groupPeople };
  };

  const prepareTrackForRescan = (track, wrongPersonId = null) => {
    if (!track) return;
    if (!track.excludedPersonIds) track.excludedPersonIds = new Set();
    if (wrongPersonId) track.excludedPersonIds.add(String(wrongPersonId));

    track.recognizedPersonId = null;
    track.identityName = null;
    track.identityDescriptor = null;
    track.lastRecognizedBox = null;
    track.matchDistance = null;
    track.matchedAngle = null;
    track.recognitionMethod = null;
    track.found = false;
    track.lockedResult = false;
    track.replacementSearch = true;
    track.identityMisses = 0;
    track.pendingPersonId = null;
    track.pendingHits = 0;
    track.statusLabel = "🔄 สแกนใหม่...";
    track.color = "#f59e0b";
    track.lastRecognition = 0;
    track.isRecognizing = false;
    track.rescanAttempts = Math.min(5, Number(track.rescanAttempts || 0) + 1);
  };

  const recognizeFace = async (track, video, box) => {
    // จัดตำแหน่งในแถวตามแกน X: ซ้าย -> ขวา
    const candidates = tracksRef.current
      .filter((candidate) => candidate?.detection?.boundingBox && (performance.now() - (candidate.lastSeen || 0)) < 500)
      .sort((a, b) => {
        const ab = a.detection.boundingBox;
        const bb = b.detection.boundingBox;
        return ((ab.originX || 0) + (ab.width || 0) / 2) - ((bb.originX || 0) + (bb.width || 0) / 2);
      });
    const candidateIndex = candidates.findIndex((candidate) => String(candidate.id) === String(track.id));
    track.rowOrder = candidateIndex >= 0 ? candidateIndex + 1 : 1;

    recognitionActiveRef.current += 1;
    const silentIdentityGuard = Boolean(track.lockedResult && track.recognizedPersonId);

    try {
      if (!silentIdentityGuard) {
        track.statusLabel = "📸 จับภาพใบหน้า...";
        track.color = "#38bdf8";
        setScanActivity(`📸 จับภาพ Face • คนที่ ${track.id.substring(0, 5)}`);
      }
      await new Promise(requestAnimationFrame);

      const querySnapshot = captureFaceSnapshotData(video, box);
      if (!silentIdentityGuard) {
        track.statusLabel = "🧠 สร้าง Face Descriptor...";
        setScanActivity(`🧠 วิเคราะห์ Face Descriptor • คนที่ ${track.id.substring(0, 5)}`);
      }
      const detection = await detectWithMultipleModels(querySnapshot.canvas);

      if (!detection) {
        // ถ้าเคยยืนยันตัวตนแล้ว ห้ามเปลี่ยนเป็นเหลืองเพียงเพราะ Face API
        // อ่าน descriptor ของเฟรมนี้ไม่สำเร็จ เพราะ MediaPipe ยังตรวจพบหน้าอยู่
        if (silentIdentityGuard && track.recognizedPersonId) {
          track.identityMisses = Math.min(
            LOCKED_IDENTITY_MAX_MISSES,
            (track.identityMisses || 0) + 1
          );
          track.statusLabel = track.identityName || "ยืนยันแล้ว";
          track.found = true;
          track.lockedResult = true;
          track.color = "#22c55e";
          track.lastIdentityVerifiedAt = performance.now();
          return;
        }

        setScanActivity("⚠️ พบกรอบหน้า แต่สร้าง Face Descriptor ไม่สำเร็จ • กำลังลองใหม่");
        track.statusLabel = "⚠️ อ่าน Face ไม่สำเร็จ • ลองใหม่...";
        track.color = "#f59e0b";
        track.lastRecognition = 0;
        setDetectedFaces((prev) =>
          prev.map((face) =>
            face.id === track.id.substring(0, 5)
              ? { ...face, label: track.statusLabel, found: false, lockedResult: false }
              : face
          )
        );
        return;
      }

      detection.appearanceFingerprint = querySnapshot.appearanceFingerprint;

      const currentDescriptor = Array.from(detection.descriptor);
      track.currentDescriptor = currentDescriptor;
      if (!track.excludedPersonIds) track.excludedPersonIds = new Set();
      if (!track.excludedPersonIdsUntil) track.excludedPersonIdsUntil = 0;
      if (!track.replacementSearch) track.replacementSearch = false;
      expireTrackExclusions(track);

      // ค้นเฉพาะ Face Data ของกลุ่มที่เลือก
      // Face Matcher ใช้ memory ที่โหลดมาจาก LocalStorage ตอนเลือกกลุ่ม/เริ่มกล้อง
      // ไม่อ่าน LocalStorage ทุก frame เพราะจะทำให้ iPhone/Safari หน่วง
      const people = groupFacesRef.current || [];

      setScanActivity(`🔎 ค้นฐานข้อมูลใบหน้า ${people.length} โปรไฟล์...`);

      if (!people.length) {
        if (silentIdentityGuard && track.recognizedPersonId) {
          track.identityMisses = Math.min(LOCKED_IDENTITY_MAX_MISSES, (track.identityMisses || 0) + 1);
          track.statusLabel = track.identityName || "ยืนยันแล้ว";
          track.found = true;
          track.lockedResult = true;
          track.color = "#22c55e";
          track.lastIdentityVerifiedAt = performance.now();
          return;
        }

        setScanActivity("⚠️ ตรวจพบใบหน้าแล้ว แต่ฐานข้อมูลยังไม่มีบุคคล");
        track.statusLabel = "ยังไม่มีข้อมูลในระบบ";
        track.color = "#ef4444";
        track.matchDistance = null;
        track.found = false;
        track.lockedResult = false;
        track.lastRecognition = performance.now();
        setDetectedFaces((prev) =>
          prev.map((face) =>
            face.id === track.id.substring(0, 5)
              ? { ...face, label: track.statusLabel, found: false, lockedResult: false }
              : face
          )
        );
        return;
      }

      let result = null;
      const scoped = getScopedRecognitionPeople(people, track.excludedPersonIds || new Set());
      const groupPeople = scoped.groupPeople;
      const primaryPeople = groupPeople;

      // ----- ขั้นที่ 1: session memory ต้องผ่าน Face Descriptor เท่านั้น -----
      // ห้ามใช้แค่ตำแหน่งของกรอบ เพราะคนใหม่/คนอื่นอาจเข้ามายืนตำแหน่งเดิมได้
      if (track.recognizedPersonId && !track.excludedPersonIds?.has(String(track.recognizedPersonId))) {
        const cachedPerson = groupPeople.find(
          (person) => String(person.id) === String(track.recognizedPersonId)
        );
        const referenceVectors = [
          ...(track.identityDescriptor ? [track.identityDescriptor] : []),
          ...((cachedPerson?.descriptors || []).map(toDescriptorArray).filter(Boolean)),
        ];

        const distances = referenceVectors
          .map(toDescriptorArray)
          .filter((vector) => vector && vector.length === currentDescriptor.length)
          .map((vector) =>
            faceapi.euclideanDistance(
              new Float32Array(currentDescriptor),
              new Float32Array(vector)
            )
          )
          .sort((a, b) => a - b);

        const cachedBest = distances[0];
        const cachedSecond = distances[1] ?? Infinity;

        if (
          Number.isFinite(cachedBest) &&
          cachedBest <= LOCKED_IDENTITY_MAX_DISTANCE &&
          (
            !Number.isFinite(cachedSecond) ||
            cachedSecond - cachedBest >= 0.025
          )
        ) {
          result = {
            person: cachedPerson,
            distance: cachedBest,
            margin: cachedSecond - cachedBest,
            support: 1,
            matchedImage: null,
            matchedAngle: null,
            source: "descriptor-session-cache",
            confidentEnough: Boolean(cachedPerson),
          };
        } else if (track.recognizedPersonId) {
          // Descriptor ของเฟรมเดียวไม่ควรล้าง identity ที่ล็อกไว้ทันที
          // เก็บชื่อเดิมไว้ก่อน และให้ guard ตรวจซ้ำอีกหลายครั้ง
          track.identityMisses = Math.min(
            LOCKED_IDENTITY_MAX_MISSES,
            (track.identityMisses || 0) + 1
          );
          track.lastIdentityVerifiedAt = performance.now();

          if ((track.identityMisses || 0) < LOCKED_IDENTITY_MAX_MISSES) {
            track.statusLabel = track.identityName || "ยืนยันแล้ว";
            track.found = true;
            track.lockedResult = true;
            track.color = "#22c55e";
            return;
          }

          // หลุดจริงเมื่อไม่ตรงกันต่อเนื่องหลายครั้งเท่านั้น
          const replacedPersonId = String(track.recognizedPersonId);
          track.excludedPersonIds.add(replacedPersonId);
          track.excludedPersonIdsUntil = performance.now() + DUPLICATE_EXCLUSION_MS;
          track.replacementSearch = true;
          track.recognizedPersonId = null;
          track.identityName = null;
          track.identityDescriptor = null;
          track.lastRecognizedBox = null;
          track.found = false;
          track.lockedResult = false;
          track.matchDistance = null;
          track.recognitionMethod = null;
          track.statusLabel = "กำลังค้นหาใหม่...";
          track.color = "#f59e0b";
          track.identityMisses = 0;
        }
      }

      // ถ้าเป็น track ใหม่ที่เพิ่งเกิดใกล้คนเดิม ก็ต้องตรวจ descriptor ก่อนคืนชื่อ
      if (!result && !track.recognizedPersonId && groupPeople.length) {
        const remembered = findRecentRecognizedPersonByDescriptor(
          currentDescriptor,
          box,
          track.excludedPersonIds || new Set()
        );
        if (remembered) {
          const person = groupPeople.find(
            (item) => String(item.id) === String(remembered.personId)
          );
          if (person) {
            result = {
              person,
              distance: remembered.distance,
              margin: Infinity,
              support: 1,
              matchedImage: null,
              matchedAngle: null,
              source: "descriptor-recent-cache",
              confidentEnough: true,
            };
          }
        }
      }

      // ----- ขั้นที่ 2: ค้นฐานข้อมูลเต็ม เฉพาะเมื่อ cache ใช้ไม่ได้ -----
      if (!result && primaryPeople.length) {
        result = await recognizeWithSavedImages(
          detection,
          primaryPeople,
          track.excludedPersonIds || new Set()
        );
      }


      // เมื่อใบหน้าใหม่เข้ามาแทน track ที่เคยล็อกชื่อไว้
      // ต้องใช้เกณฑ์เข้มกว่าปกติ เพื่อไม่ให้คนใหม่ถูกยัดชื่อของคนเดิมจาก match ที่ก้ำกึ่ง
      if (track.replacementSearch && result?.person) {
        const replacementAccepted =
          Number.isFinite(result.distance) &&
          result.distance <= REPLACEMENT_SEARCH_MAX_DISTANCE &&
          Number.isFinite(result.faceScore) &&
          result.faceScore <= REPLACEMENT_SEARCH_MAX_FACE_SCORE &&
          (!Number.isFinite(result.margin) || result.margin >= REPLACEMENT_SEARCH_MIN_MARGIN);

        if (!replacementAccepted) {
          console.warn("⛔ Replacement match rejected:", {
            candidate: result.person?.name,
            distance: result.distance,
            faceScore: result.faceScore,
            margin: result.margin,
          });
          result = null;
        }
      }

      // ขณะเป็นสีเขียว ห้ามเปลี่ยนชื่อเพราะผลจากเฟรมเดียว
      // ถ้า full-search เสนอคนอื่น ให้ถือเป็น verification miss และรักษา identity เดิมไว้
      if (
        silentIdentityGuard &&
        track.recognizedPersonId &&
        result?.person &&
        String(result.person.id) !== String(track.recognizedPersonId)
      ) {
        track.identityMisses = Math.min(
          LOCKED_IDENTITY_MAX_MISSES,
          (track.identityMisses || 0) + 1
        );
        track.lastIdentityVerifiedAt = performance.now();

        if ((track.identityMisses || 0) < LOCKED_IDENTITY_MAX_MISSES) {
          track.statusLabel = track.identityName || "ยืนยันแล้ว";
          track.found = true;
          track.lockedResult = true;
          track.color = "#22c55e";
          return;
        }

        // ครบจำนวน miss แล้วค่อยยอมให้ระบบกลับไปค้นหาคนใหม่
        result = null;
        track.identityMisses = 0;
        track.recognizedPersonId = null;
        track.identityName = null;
        track.identityDescriptor = null;
        track.lastRecognizedBox = null;
        track.found = false;
        track.lockedResult = false;
        track.color = "#f59e0b";
        track.statusLabel = "กำลังค้นหาใหม่...";
      }

      console.log("🔍 Face match result:", result
        ? {
          name: result.person?.name,
          distance: result.distance,
          confidentEnough: result.confidentEnough,
          margin: result.margin,
          source: result.source,
        }
        : null
      );

      if (!result?.person || !result.confidentEnough) {
        if (silentIdentityGuard && track.recognizedPersonId) {
          track.identityMisses = Math.min(
            LOCKED_IDENTITY_MAX_MISSES,
            (track.identityMisses || 0) + 1
          );
          track.lastIdentityVerifiedAt = performance.now();

          if ((track.identityMisses || 0) < LOCKED_IDENTITY_MAX_MISSES) {
            track.statusLabel = track.identityName || "ยืนยันแล้ว";
            track.found = true;
            track.lockedResult = true;
            track.color = "#22c55e";
            return;
          }

          // ไม่ตรงต่อเนื่องครบจำนวนที่กำหนด จึงค่อยปลดล็อก
          track.identityMisses = 0;
        }

        setScanActivity(
          result?.ambiguous
            ? "⚠️ หน้าคล้ายหลายคน • ไม่ฟันธง"
            : `⚠️ ไม่พบคนที่ตรงกัน${Number.isFinite(result?.distance) ? ` (distance ${result.distance.toFixed(3)})` : ""} • พร้อมสแกนใหม่`
        );
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
        track.identityDescriptor = null;
        track.lastIdentityVerifiedAt = performance.now();
        track.recognitionMethod = "strict";
        track.lastRecognition = performance.now();

        // อัปเดต UI ทันทีว่าค้นเสร็จแล้วแต่ไม่พบ ไม่ปล่อยคำว่า "กำลังระบุตัวตน" ค้าง
        setDetectedFaces((prev) =>
          prev.map((face) =>
            face.id === track.id.substring(0, 5)
              ? {
                ...face,
                label: track.statusLabel,
                found: false,
                lockedResult: false,
                matchDistance: track.matchDistance,
                recognitionMethod: "strict",
              }
              : face
          )
        );
        return;
      }

      // ----- เจอคนในกลุ่ม = ล็อกทันทีตรง Track ปัจจุบัน -----
      setScanActivity(`✓ พบ ${result.person.name}`);
      track.statusLabel = result.person.name;
      track.identityName = result.person.name;
      track.color = "#22c55e";
      track.found = true;
      track.lockedResult = true;
      track.matchedAngle = result.matchedAngle || null;
      track.matchDistance = result.distance;
      track.recognitionMethod = result.source || "all-images";
      track.recognizedPersonId = result.person.id;
      track.replacementSearch = false;

      // เจอใครตรง Track ไหน ให้ล็อกบุคคลนั้นไว้กับ Track ตรงนั้นทันที
      updateGroupScanResult(result.person.id, "matched", result.person.name, box);
      if (track.excludedPersonIds) track.excludedPersonIds.clear();
      track.excludedPersonIdsUntil = 0;
      track.identityDescriptor = currentDescriptor.slice();
      track.identityMisses = 0;
      track.lastIdentityVerifiedAt = performance.now();
      track.lastRecognizedBox = { ...box };
      track.lastRecognition = performance.now();

      // ถ้ามี track อื่นในเฟรมเดียวกันที่เป็นคนเดียวกัน
      // เลือกเฉพาะผลที่ตรง group_faces มากที่สุด ส่วน track ที่แพ้กลับไปค้นใหม่
      const isBestDuplicateMatch = resolveDuplicateRecognition(track);
      if (!isBestDuplicateMatch) {
        setScanActivity("🔄 ตรวจพบกรอบซ้ำ • ใช้ผลที่ตรงที่สุด และส่งอีกกรอบกลับไปค้นใหม่");
        setDetectedFaces((prev) =>
          prev.map((face) =>
            face.id === track.id.substring(0, 5)
              ? {
                ...face,
                name: null,
                label: "กำลังค้นหาใหม่...",
                found: false,
                lockedResult: false,
                matchDistance: null,
                recognitionMethod: null,
              }
              : face
          )
        );
        return;
      }

      rememberRecognizedPerson(result.person, box, result.distance, result.source || "group_faces");

      // อัปเดต UI ทันที ไม่ต้องรอ detector รอบถัดไป
      setDetectedFaces((prev) =>
        prev.map((face) =>
          face.id === track.id.substring(0, 5)
            ? {
              ...face,
              name: result.person.name,
              label: result.person.name,
              found: true,
              lockedResult: true,
              matchDistance: track.matchDistance,
              recognitionMethod: track.recognitionMethod,
              matchedAngle: track.matchedAngle || null,
            }
            : face
        )
      );

      // จำผลไว้เฉพาะ session ปัจจุบัน
      // ใช้เฉพาะ descriptor จาก group_faces + descriptor ของการจับครั้งนี้
      // ห้ามดึงประวัติ face_scan_history_v1 มาเป็นฐานค้นหาคน
      recognizedSessionRef.current.set(result.person.id, {
        person: result.person,
        descriptors: [
          ...(result.person.descriptors || []),
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

          const updatedPeople = groupFacesRef.current.map((person) => {
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

          groupFacesRef.current = updatedPeople;
          setGroupFaces(updatedPeople);
          updateFaceMatcher(updatedPeople);

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

      const currentLandmarks = lastMeshResultRef.current?.faceLandmarks?.[0];
      const profile = currentLandmarks ? buildDepthProfile(currentLandmarks) : null;
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

      if (silentIdentityGuard && track.recognizedPersonId) {
        track.identityMisses = Math.min(
          LOCKED_IDENTITY_MAX_MISSES,
          (track.identityMisses || 0) + 1
        );
        track.statusLabel = track.identityName || "ยืนยันแล้ว";
        track.color = "#22c55e";
        track.found = true;
        track.lockedResult = true;
        track.lastIdentityVerifiedAt = performance.now();
      } else {
        track.statusLabel = "ตรวจสอบใหม่...";
        track.color = "#f59e0b";
        track.found = false;
        track.lockedResult = false;
        track.identityDescriptor = null;
      }
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
    if (!cameraOnRef.current) return;

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

    // ไม่แสดงจุด/เส้น mesh บนใบหน้าในหน้า Scan

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
    const usedRecentPersonIds = new Set();
    const confirmedDetections = [];

    dedupedValidDetections.forEach((item) => {
      const detection = item.detection;
      const box = detection.boundingBox;
      const centerX = box.originX + box.width / 2;
      const centerY = box.originY + box.height / 2;

      let track = tracks
        .filter((candidate) => !seenTrackIds.has(candidate.id))
        .map((candidate) => {
          const gap = candidate.lastSeen ? frameTimestamp - candidate.lastSeen : Infinity;
          const centerDistance = Math.hypot(
            candidate.centerX - centerX,
            candidate.centerY - centerY
          );
          const candidateBox = candidate.lastRecognizedBox || candidate.smoothedBox || candidate.detection?.boundingBox;
          const sameFaceGeometry = boxesLikelySameFace(candidateBox, box);

          // IMPORTANT: a locked identity is never transferred to a new face merely
          // because the new face enters the same area after a brief disappearance.
          // After a gap, create a new track and let group_faces identify it.
          const lockedTrackContinuityOk = !candidate.lockedResult || (
            gap <= LOCKED_TRACK_MAX_GAP_MS &&
            sameFaceGeometry
          );

          return { candidate, distance: centerDistance, lockedTrackContinuityOk };
        })
        .filter(({ candidate, distance, lockedTrackContinuityOk }) =>
          lockedTrackContinuityOk &&
          distance < Math.max(box.width, candidate.width) * 0.7
        )
        .sort((left, right) => left.distance - right.distance)[0]?.candidate;

      if (!track) {
        track = {
          id: crypto.randomUUID(),
          frames: 0,
          statusLabel: "กำลังตรวจสอบ",
          identityName: null,
          color: "#f59e0b",
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
          lastRecognizedBox: null,
          recognitionMethod: null,
          excludedPersonIds: new Set(),
          excludedPersonIdsUntil: 0,
          replacementSearch: false,
          currentDescriptor: null,
          identityDescriptor: null,
          lastIdentityVerifiedAt: 0,
          identityMisses: 0
        };
        tracks.push(track);
      }

      track.frames += 1;
      if (!track.excludedPersonIds) track.excludedPersonIds = new Set();
      if (!Object.prototype.hasOwnProperty.call(track, "excludedPersonIdsUntil")) {
        track.excludedPersonIdsUntil = 0;
      }
      if (!Object.prototype.hasOwnProperty.call(track, "replacementSearch")) {
        track.replacementSearch = false;
      }
      expireTrackExclusions(track);
      if (!Object.prototype.hasOwnProperty.call(track, "identityDescriptor")) {
        track.identityDescriptor = null;
      }
      if (!Object.prototype.hasOwnProperty.call(track, "lastIdentityVerifiedAt")) {
        track.lastIdentityVerifiedAt = 0;
      }
      if (!Object.prototype.hasOwnProperty.call(track, "identityMisses")) {
        track.identityMisses = 0;
      }
      const previousLastSeen = track.lastSeen || 0;
      const gapSincePreviousDetection = previousLastSeen
        ? frameTimestamp - previousLastSeen
        : 0;

      track.centerX = centerX;
      track.centerY = centerY;
      track.width = box.width;
      track.detection = detection;

      // IMPORTANT: once this track has been recognized, the identity is final
      // for this track. Never put it back into "กำลังระบุตัวตน" or start a new
      // recognition job for the same locked track.
      if (track.lockedResult && track.recognizedPersonId) {
        const person = groupFacesRef.current.find(
          (item) => String(item.id) === String(track.recognizedPersonId)
        );
        if (person) {
          track.statusLabel = person.name;
          track.identityName = person.name;
          track.found = true;
          track.lockedResult = true;
          track.color = "#22c55e";
        }
      }
      track.meshScore = item.meshScore;
      track.landmarks = item.landmarks;
      track.lastSeen = frameTimestamp;
      seenTrackIds.add(track.id);

      if (track.lockedResult && track.recognizedPersonId) {
        const person = groupFacesRef.current.find(
          (item) => String(item.id) === String(track.recognizedPersonId)
        );
        if (person) {
          rememberRecognizedPerson(
            person,
            box,
            track.matchDistance,
            track.recognitionMethod || "session-memory"
          );
        }
      }

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

        const lockedIdentityNeedsGuard =
          track.lockedResult &&
          track.recognizedPersonId &&
          (
            frameTimestamp - (track.lastIdentityVerifiedAt || 0) >= LOCKED_IDENTITY_VERIFY_INTERVAL_MS ||
            gapSincePreviousDetection >= 220
          );

        if (
          !track.isRecognizing &&
          recognitionActiveRef.current < MAX_CONCURRENT_RECOGNITION &&
          (
            lockedIdentityNeedsGuard ||
            (
              !track.lockedResult &&
              !track.found &&
              !track.recognizedPersonId &&
              frameTimestamp - (track.lastRecognition || 0) >= 300
            )
          )
        ) {
          track.isRecognizing = true;
          // Locked names stay visible while the silent guard checks the new descriptor.
          if (!lockedIdentityNeedsGuard) {
            track.statusLabel = "กำลังค้นหา...";
            track.color = "#f59e0b";
          }
          recognizeFace(track, video, box);
        }
      } else if (track.frames >= 1) {
        // แสดงกรอบตั้งแต่เฟรมแรกที่ผ่าน detector
        confirmedDetections.push(track);
      }
    });

    tracksRef.current = tracks.filter((track) => {
      const gap = frameTimestamp - track.lastSeen;
      return gap < (
        track.lockedResult
          ? LOCKED_TRACK_MAX_GAP_MS
          : UNLOCKED_TRACK_MAX_GAP_MS
      );
    });

    // ลำดับสำหรับ "แถว" = ซ้าย -> ขวา ตามตำแหน่งใบหน้าในภาพ
    const orderedDetections = confirmedDetections
      .sort((a, b) => {
        const aBox = a.detection?.boundingBox;
        const bBox = b.detection?.boundingBox;
        const aCenterX = (aBox?.originX || 0) + (aBox?.width || 0) / 2;
        const bCenterX = (bBox?.originX || 0) + (bBox?.width || 0) / 2;
        return aCenterX - bCenterX;
      })
      .slice(0, MAX_PEOPLE);

    // Final visual guard:
    // 1) ถ้าหลาย track ได้ personId เดียวกันในเฟรมเดียว ให้เหลือชื่อไว้เพียง track เดียว
    //    โดยเลือก distance ต่ำสุด; track อื่นจะถูกปลดชื่อและบล็อก personId นี้เพื่อค้นหาคนอื่นใหม่
    // 2) ถ้ายังไม่รู้ชื่อและกรอบซ้ำ ให้รวมตาม geometry
    const deduplicatedDetections = [];
    for (const track of orderedDetections) {
      if (track.recognizedPersonId) {
        const sameIdentityIndex = deduplicatedDetections.findIndex(
          (kept) =>
            String(kept.recognizedPersonId) === String(track.recognizedPersonId)
        );

        if (sameIdentityIndex !== -1) {
          const kept = deduplicatedDetections[sameIdentityIndex];
          const keptDistance = Number.isFinite(kept.matchDistance) ? kept.matchDistance : Infinity;
          const currentDistance = Number.isFinite(track.matchDistance) ? track.matchDistance : Infinity;
          const currentIsBetter =
            currentDistance < keptDistance ||
            (currentDistance === keptDistance &&
              track.width * track.height > kept.width * kept.height);

          const winner = currentIsBetter ? track : kept;
          const loser = currentIsBetter ? kept : track;

          loser.found = false;
          loser.lockedResult = false;
          loser.recognizedPersonId = null;
          loser.identityName = null;
          loser.matchedAngle = null;
          loser.matchDistance = null;
          loser.recognitionMethod = null;
          loser.statusLabel = "กำลังค้นหาใหม่...";
          loser.color = "#f59e0b";
          loser.lastRecognition = 0;
          loser.isRecognizing = false;
          loser.lastRecognizedBox = null;
          if (!loser.excludedPersonIds) loser.excludedPersonIds = new Set();
          loser.excludedPersonIds.add(String(winner.recognizedPersonId));
          loser.excludedPersonIdsUntil = performance.now() + DUPLICATE_EXCLUSION_MS;
          loser.replacementSearch = true;

          deduplicatedDetections[sameIdentityIndex] = winner;

          if (loser === track) {
            // Keep the losing track visible so the next frame can search for another person.
            deduplicatedDetections.push(loser);
          } else {
            // The previous kept track lost; keep the current winner and append the previous loser.
            deduplicatedDetections.push(loser);
          }
          continue;
        }
      }

      const duplicateIndex = deduplicatedDetections.findIndex((kept) => {
        if (kept.recognizedPersonId || track.recognizedPersonId) return false;
        return boxesLikelySameFace(
          kept.lastRecognizedBox || kept.smoothedBox || kept.detection?.boundingBox,
          track.lastRecognizedBox || track.smoothedBox || track.detection?.boundingBox
        );
      });

      if (duplicateIndex === -1) {
        deduplicatedDetections.push(track);
        continue;
      }

      const kept = deduplicatedDetections[duplicateIndex];
      const keptArea = kept.width * kept.height;
      const currentArea = track.width * track.height;
      if (currentArea > keptArea) deduplicatedDetections[duplicateIndex] = track;
    }

    setFaceCount(deduplicatedDetections.length);
    setDetectedFaces(deduplicatedDetections.map((trk, index) => ({
      id: trk.id.substring(0, 5),
      personId: trk.recognizedPersonId ? String(trk.recognizedPersonId) : null,
      recognizedPersonId: trk.recognizedPersonId ? String(trk.recognizedPersonId) : null,
      label: trk.identityName || trk.statusLabel,
      name: trk.identityName || null,
      box: trk.smoothedBox || trk.detection.boundingBox,
      order: index + 1,
      rowOrder: index + 1,
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
    {
      if (!orderedDetections.length) {
        setScanMessage("ยังไม่พบใบหน้า • กำลังค้นหา...");
        setScanActivity("🔍 กำลังค้นหาใบหน้าในภาพ...");
      } else {
        const recognizedCount = orderedDetections.filter((track) =>
          Boolean(track.found || track.lockedResult || /^✓\s/.test(track.statusLabel || ""))
        ).length;
        const activeRecognitionCount = orderedDetections.filter((track) => track.isRecognizing).length;
        const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
        const groupMemberIds = new Set((group?.memberIds || []).map(String));
        const foundGroupIds = new Set(
          Object.values(group?.scanResults || {})
            .filter((result) => result?.status === "matched" && result?.personId && groupMemberIds.has(String(result.personId)))
            .map((result) => String(result.personId))
        );
        orderedDetections
          .filter((track) => track.found && track.recognizedPersonId && groupMemberIds.has(String(track.recognizedPersonId)))
          .forEach((track) => foundGroupIds.add(String(track.recognizedPersonId)));
        const foundGroupCount = foundGroupIds.size;
        const totalGroupCount = groupMemberIds.size;

        if (totalGroupCount > 0 && foundGroupCount >= totalGroupCount) {
          setScanMessage(`✅ ครบ ${foundGroupCount}/${totalGroupCount} คน • พบสมาชิกครบแล้ว`);
        } else if (totalGroupCount > 0) {
          setScanMessage(`👥 พบ ${foundGroupCount}/${totalGroupCount} คนในกลุ่ม • เห็นหน้า ${orderedDetections.length} • กำลังค้นหา ${activeRecognitionCount} คน`);
        } else if (activeRecognitionCount > 0) {
          setScanMessage(`พบ ${orderedDetections.length} คน • AI กำลังระบุตัวตน`);
        } else {
          setScanMessage(`พบ ${orderedDetections.length} คน • เลือกกลุ่มเพื่อเริ่มตรวจสมาชิก`);
        }
      }
    }

    // Detection boxes are rendered by the React overlay below.
    // The canvas is reserved for the 3D mesh, so there is no duplicate box.

    if (cameraOnRef.current) {
      animationRef.current = requestAnimationFrame(detectFaces);
    }
  }


  const getExpectedPersonForSlot = (slotIndex) => {
    const personId = selectedRowPeople[slotIndex] || "";
    if (!personId) return null;
    return groupFacesRef.current.find((person) => String(person.id) === String(personId)) || null;
  };

  const getFaceRowStatus = (face) => {
    const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
    const memberIds = new Set((group?.memberIds || []).map(String));
    const personId = face?.personId || face?.recognizedPersonId || "";

    if (personId && memberIds.has(String(personId)) && face?.found) {
      const savedResult = group?.scanResults?.[String(personId)];
      return {
        key: "green",
        label: `✓ ${face.name || savedResult?.detectedName || "พบแล้ว"}`,
        className: "border-green-300 shadow-[0_0_28px_rgba(34,197,94,0.7)]",
      };
    }

    if (personId && !memberIds.has(String(personId))) {
      return {
        key: "red",
        label: `✕ ${face.name || "บุคคลนอกกลุ่ม"} • ไม่อยู่ในกลุ่ม`,
        className: "border-red-400 shadow-[0_0_24px_rgba(248,113,113,0.65)]",
      };
    }

    return {
      key: "yellow",
      label: face?.label || "🟡 กำลังค้นหา...",
      className: "border-yellow-300 shadow-[0_0_18px_rgba(250,204,21,0.35)]",
    };
  };

  const selectRowPerson = async (slotIndex, personId) => {
    const person = groupFacesRef.current.find((item) => String(item.id) === String(personId))
      || facePeople.find((item) => String(item.id) === String(personId));
    if (selectedGroupId) {
      const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
      if (group) {
        const ids = [...(group.memberIds || [])].filter((id) => String(id) !== String(personId));
        ids.splice(slotIndex, 0, String(personId));
        const nextIds = ids.slice(0, MAX_PEOPLE);
        try {
          await saveGroupMemberIds(selectedGroupId, nextIds);
          const nextGroup = { ...group, memberIds: nextIds, scanResults: {} };
          const nextFaces = nextIds.map((id) => facePeople.find((p) => String(p.id) === String(id))).filter(Boolean).map(normalizePersonImages);
          setScanGroups((prev) => prev.map((item) => String(item.id) === String(group.id) ? nextGroup : item));
          groupFacesRef.current = nextFaces;
          setGroupFaces(nextFaces);
          applyGroupToRow(nextGroup, nextFaces);
          setActiveRowSearch(null);
          updateFaceMatcher(nextFaces);
          saveFaceScanSelectionCache(nextGroup, nextFaces);
        } catch (error) {
          console.error("❌ Save selected row person failed:", error);
        }
        return;
      }
    }
    setSelectedRowPeople((prev) => { const next = [...prev]; next[slotIndex] = String(personId); return next; });
    setRowSearch((prev) => { const next = [...prev]; next[slotIndex] = person?.name || ""; return next; });
    setActiveRowSearch(null);
  };

  const clearRowPerson = (slotIndex) => {
    setSelectedRowPeople((prev) => {
      const next = [...prev];
      next[slotIndex] = "";
      return next;
    });
    setRowSearch((prev) => {
      const next = [...prev];
      next[slotIndex] = "";
      return next;
    });
  };

  const selectedScanGroup = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
  const expectedPeopleCount = (selectedScanGroup?.memberIds || []).length;
  const configuredPeopleIds = new Set((selectedScanGroup?.memberIds || []).map(String));

  const getCameraStatusText = () => {
    if (!cameraOn) return "AI VISION";

    const faces = Array.isArray(detectedFaces) ? detectedFaces : [];

    if (!faces.length) {
      return "🔍 กำลังค้นหาใบหน้า...";
    }

    const recognized = faces.filter((face) => Boolean(face.name));
    const notFound = faces.filter((face) =>
      /ไม่พบ|ยังไม่มีข้อมูล|ไม่สำเร็จ|ไม่ฟันธง/.test(face.label || "")
    );
    const searching = faces.length - recognized.length - notFound.length;

    if (recognized.length === faces.length) {
      const names = recognized.map((face) => face.name).filter(Boolean);
      return names.length === 1
        ? `✓ ${names[0]}`
        : `✓ ระบุแล้ว ${names.length} คน`;
    }

    if (recognized.length > 0 && searching > 0) {
      return `✓ พบแล้ว ${recognized.length} คน • 🔎 กำลังค้นหา ${searching} คน`;
    }

    if (recognized.length > 0 && notFound.length > 0) {
      return `✓ พบแล้ว ${recognized.length} คน • ⚠ ไม่พบ ${notFound.length} คน`;
    }

    if (searching > 0) {
      return `🔎 กำลังระบุตัวตน ${searching} คน`;
    }

    return `⚠ ไม่พบคนที่ตรงกัน ${notFound.length} คน`;
  };

  const loadFacePeopleForModal = async () => {
    setFacePeopleLoading(true);
    try {
      const response = await fetch(FACE_PEOPLE_API, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      const rows = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.data)
          ? payload.data
          : [];

      // Modal ใช้ข้อมูลจาก API โดยตรงเท่านั้น
      const apiPeople = rows
        .map(normalizePersonImages)
        .filter((person) => person?.id && person?.descriptors?.length);

      setFacePeople(apiPeople);
    } catch (error) {
      setFacePeople([]);
    } finally {
      setFacePeopleLoading(false);
    }
  };

  const renderFacePeopleModalContent = () => {
    const currentGroup = scanGroups.find(
      (item) => String(item.id) === String(selectedGroupId)
    );

    const memberIds = new Set(
      (currentGroup?.memberIds || []).map(String)
    );

    const query = groupSearch.trim().toLowerCase();

    // สำคัญ: รายชื่อใน Modal ต้องมาจาก /api/v1/face/people เท่านั้น
    // ห้ามอ่านรายชื่อจาก savedFaces / localStorage เพื่อแสดงผล Modal
    const people = facePeople.filter((person) => {
      if (memberIds.has(String(person.id))) return false;
      if (!query) return true;

      const name = String(person.name || "").toLowerCase();
      const employeeCode = String(person.employee_code || "").toLowerCase();
      return name.includes(query) || employeeCode.includes(query);
    });

    if (!people.length) {
      return (
        <div className="flex min-h-40 items-center justify-center rounded-xl border border-dashed border-white/10">
          <div className="text-center text-xs text-gray-500">
            {query ? "ไม่พบรายชื่อที่ค้นหา" : "ไม่มีบุคคลที่สามารถเพิ่มได้"}
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-1">
        {people.map((person) => {
          const previewImage =
            person.frontImage ||
            person.faceImages?.find(
              (item) => String(item?.stage || "").toLowerCase() === "front"
            )?.image ||
            person.faceImages?.[0]?.image ||
            person.image ||
            "";

          return (
            <div
              key={person.id}
              className="flex items-center gap-3 rounded-xl border border-white/5 bg-white/[0.025] p-2.5 hover:border-cyan-400/20 hover:bg-cyan-400/5"
            >
              {previewImage ? (
                <img
                  src={previewImage}
                  alt={person.name || ""}
                  className="h-11 w-11 shrink-0 rounded-xl object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
              ) : (
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/5 text-lg">
                  👤
                </div>
              )}

              <div className="min-w-0 flex-1">
                <div className="truncate text-xs font-bold text-white">
                  {person.name || "ไม่มีชื่อ"}
                </div>
                <div className="mt-0.5 text-[9px] text-gray-500">
                  {person.employee_code || "ไม่มีรหัสพนักงาน"}
                </div>
              </div>

              <button
                type="button"
                disabled={!selectedGroupId || (currentGroup?.memberIds || []).length >= MAX_PEOPLE}
                onClick={() => {
                  addPersonToSelectedGroup(person.id);
                  setShowFacePeopleModal(false);
                  setGroupSearch("");
                }}
                className="shrink-0 rounded-xl bg-cyan-400 px-3 py-2 text-[10px] font-black text-black disabled:cursor-not-allowed disabled:opacity-30"
              >
                เลือก
              </button>
            </div>
          );
        })}
      </div>
    );
  };

  const cameraStatusText = getCameraStatusText();

  return (
    <div className={cameraOn ? "fixed inset-0 left-1/2 z-[9999] h-[100dvh] w-full max-w-[430px] -translate-x-1/2 overflow-hidden overscroll-none bg-black" : "min-h-[calc(100dvh-120px)] space-y-3 overflow-x-hidden"}>
      <main className={cameraOn ? "relative h-full w-full overflow-hidden" : "w-full space-y-3 p-3 pb-4 sm:space-y-4 sm:p-4"}>
        <section
          className={cameraOn ? "camera-panel absolute inset-0 h-full w-full overflow-hidden bg-black" : "camera-panel relative aspect-[3/4] w-full overflow-hidden rounded-2xl border border-white/10 bg-black sm:aspect-video"}
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

          {cameraOn && [...detectedFaces].sort((a, b) => (a.order ?? 999) - (b.order ?? 999)).map((face) => {
            const b = face.box;
            if (!b) return null;
            return (
              <div
                key={face.id}
                className={`pointer-events-none absolute z-20 rounded-2xl border-2 ${getFaceRowStatus(face).className}`}
                style={(() => {
                  const vw = Math.max(1, videoRef.current?.videoWidth || 1);
                  const vh = Math.max(1, videoRef.current?.videoHeight || 1);
                  const cw = Math.max(1, videoRef.current?.clientWidth || 1);
                  const ch = Math.max(1, videoRef.current?.clientHeight || 1);

                  // Match the browser's object-cover transform exactly.
                  // This keeps the AI box attached to the face in both portrait
                  // and landscape instead of stretching by raw video percentages.
                  const coverScale = Math.max(cw / vw, ch / vh);
                  const renderedW = vw * coverScale;
                  const renderedH = vh * coverScale;
                  const offsetX = (cw - renderedW) / 2;
                  const offsetY = (ch - renderedH) / 2;

                  // Make the detection box a TRUE SQUARE around the whole face.
                  const side = Math.min(
                    Math.min(vw, vh),
                    Math.max(b.width, b.height) * 1.25
                  );
                  let x = b.originX + b.width / 2 - side / 2;
                  let y = b.originY + b.height / 2 - side / 2;
                  x = Math.max(0, Math.min(x, vw - side));
                  y = Math.max(0, Math.min(y, vh - side));

                  let displayX = x * coverScale + offsetX;
                  const displayY = y * coverScale + offsetY;
                  const displaySide = side * coverScale;

                  // Front camera is mirrored, so mirror the overlay by the same amount.
                  if (cameraMirrored) {
                    displayX = cw - (displayX + displaySide);
                  }

                  return {
                    left: `${displayX}px`,
                    top: `${displayY}px`,
                    width: `${displaySide}px`,
                    height: `${displaySide}px`,
                  };
                })()}
              >
                <span
                  className={`absolute -top-6 left-0 rounded-md px-2 py-1 text-[9px] font-black shadow-lg backdrop-blur ${(() => { const st = getFaceRowStatus(face); return st.key === "green" ? "bg-green-400 text-black" : st.key === "red" ? "bg-red-400 text-black" : "bg-yellow-300 text-black"; })()}`}
                >
                  {getFaceRowStatus(face).label}
                </span>
              </div>
            );
          })}

          {!cameraOn && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/75">
              {loading ? (
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-700 border-t-cyan-400" />
              ) : (
                <button
                  type="button"
                  onClick={toggleCamera}
                  className="flex flex-col items-center justify-center touch-manipulation"
                  aria-label="เปิดกล้อง"
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-cyan-400/10 text-3xl text-cyan-300 ring-1 ring-cyan-400/20 transition-transform active:scale-95">
                    ◉
                  </div>

                  <div className="mt-3 text-sm font-bold text-white">
                    แตะเพื่อเปิดกล้อง
                  </div>
                </button>
              )}
            </div>
          )}

          {cameraOn && (
            <>
              {selectedGroupId && (() => {
                const group = scanGroups.find((g) => String(g.id) === String(selectedGroupId));
                const members = (group?.memberIds || [])
                  .map((id) => groupFaces.find((person) => String(person.id) === String(id)))
                  .filter(Boolean);
                const foundIds = new Set(
                  Object.values(group?.scanResults || {})
                    .filter((result) => result?.status === "matched" && result?.personId)
                    .map((result) => String(result.personId))
                );
                detectedFaces.forEach((face) => {
                  if (face.found && face.personId) foundIds.add(String(face.personId));
                });
                const foundCount = members.filter((person) => foundIds.has(String(person.id))).length;
                const totalCount = members.length;
                const complete = totalCount > 0 && foundCount >= totalCount;

                return (
                  <div className="pointer-events-none absolute left-3 top-3 z-40 w-[min(240px,calc(100%-5rem))] rounded-xl border border-white/10 bg-black/75 p-2.5 text-[9px] shadow-2xl backdrop-blur-md">
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0 truncate font-black text-white">👥 {group?.name || "กลุ่ม"}</div>
                      <div className={`shrink-0 rounded-full px-2 py-0.5 font-black ${complete ? "bg-green-400 text-black" : "bg-yellow-300 text-black"}`}>
                        {complete ? "ครบ" : "ยังไม่ครบ"}
                      </div>
                    </div>
                    <div className="mt-1 font-black text-cyan-200">พบแล้ว {foundCount}/{totalCount} คน</div>
                    <div className="mt-1.5 max-h-24 space-y-0.5 overflow-hidden">
                      {members.slice(0, 6).map((person) => {
                        const found = foundIds.has(String(person.id));
                        const result = group?.scanResults?.[String(person.id)];
                        return (
                          <div key={person.id} className="flex items-center justify-between gap-2">
                            <span className={`truncate ${found ? "text-green-300" : "text-gray-400"}`}>{found ? "✓" : "○"} {person.name || "ไม่มีชื่อ"}</span>
                            <span className="shrink-0 text-[8px] text-gray-500">{found ? formatScanTime(result?.checkedAt) : "รอพบ"}</span>
                          </div>
                        );
                      })}
                      {members.length > 6 && <div className="text-gray-500">+ อีก {members.length - 6} คน</div>}
                    </div>
                  </div>
                );
              })()}

              <div className={`pointer-events-none absolute left-1/2 top-4 z-30 -translate-x-1/2 rounded-full px-4 py-2 text-center text-[11px] font-black shadow-xl backdrop-blur-md ${/^(✓|✓ พบ)/.test(cameraStatusText)
                ? "border border-green-400/40 bg-green-500/20 text-green-100"
                : /⚠/.test(cameraStatusText)
                  ? "border border-red-400/40 bg-red-500/20 text-red-100"
                  : "border border-yellow-400/30 bg-black/65 text-yellow-100"
                }`}>
                {cameraStatusText}
              </div>

              <div className="absolute bottom-3 right-3 flex items-center gap-2 rounded-full bg-black/70 px-2 py-1 text-[9px] text-green-300 backdrop-blur">
                <span>{scanActivity || "AI VISION"}</span>
                <span>LIVE</span>
              </div>
            </>
          )}

          {cameraError && (
            <div className="absolute bottom-12 left-3 right-3 rounded-lg border border-red-500/30 bg-red-950/90 p-2 text-center text-[10px] text-red-200">
              {cameraError}
            </div>
          )}
          {cameraOn && (
            <>
              {/* ปุ่มปิดกล้อง — มุมบนขวา */}
              <button
                type="button"
                onClick={toggleCamera}
                className="absolute right-3 top-3 z-50 flex h-11 w-11 touch-manipulation items-center justify-center rounded-full bg-black/60 text-lg font-black text-white shadow-lg ring-1 ring-white/20 backdrop-blur-md"
                aria-label="ปิดกล้อง"
              >
                ✕
              </button>

              {/* ปุ่มควบคุมด้านล่าง */}
              <div className="absolute bottom-0 left-0 right-0 z-40 flex flex-wrap items-center justify-center gap-2 bg-gradient-to-t from-black/95 via-black/65 to-transparent px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-10">

                <button
                  type="button"
                  onClick={switchCamera}
                  disabled={!cameraOn}
                  className="min-h-11 touch-manipulation rounded-xl bg-white/10 px-4 py-2.5 text-xs font-bold text-white ring-1 ring-white/10 disabled:opacity-30"
                >
                  ↻
                </button>

              </div>
            </>
          )}
        </section>

        {/* <div className={cameraOn ? "hidden" : "flex flex-wrap items-center justify-center gap-2"}>
          <button
            type="button"
            onClick={toggleCamera}
            className="min-h-11 touch-manipulation rounded-xl bg-cyan-400 px-4 py-2.5 text-xs font-black text-black shadow-lg"
          >
            เปิดกล้อง
          </button>
        </div> */}
        <div className={cameraOn ? "hidden" : "space-y-4"}>

          <section className="rounded-2xl border border-white/10 bg-gray-900/60 p-3 sm:p-4">
            {/* <div className="mb-3 flex items-center justify-between gap-2">
              <span className="rounded-full bg-cyan-400/10 px-2 py-1 text-[9px] text-cyan-300">
                {expectedPeopleCount} คน
              </span>
            </div> */}

            <div className="flex gap-2">
              <select
                value={selectedGroupId}
                onChange={(e) => selectScanGroup(e.target.value)}
                className="min-w-0 flex-1 rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-xs text-white outline-none focus:border-cyan-400"
              >
                <option value="">เลือกกลุ่มสำหรับสแกน...</option>
                {scanGroups.map((group) => (
                  <option key={group.id} value={group.id}>{group.name} · {(group.memberIds || []).length} คน</option>
                ))}
              </select>
              {selectedGroupId && (
                <button type="button" onClick={() => deleteScanGroup(selectedGroupId)} className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 text-[10px] font-bold text-red-300">ลบกลุ่ม</button>
              )}
            </div>

            <div className="mt-2 flex gap-2">
              <input
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") createScanGroup(); }}
                placeholder="ชื่อกลุ่มใหม่ เช่น กะ A / แถวผลิต 1"
                className="min-w-0 flex-1 rounded-lg border border-gray-700 bg-black/30 px-3 py-2 text-xs text-white outline-none placeholder:text-gray-600 focus:border-cyan-400"
              />
              <button type="button" onClick={createScanGroup} disabled={!newGroupName.trim()} className="rounded-lg bg-cyan-400 px-3 py-2 text-[10px] font-black text-black disabled:opacity-30">+ สร้างกลุ่ม</button>
            </div>

            {selectedGroupId && (() => {
              const group = scanGroups.find((item) => String(item.id) === String(selectedGroupId));
              const members = (group?.memberIds || []).map((id) => groupFaces.find((p) => String(p.id) === String(id))).filter(Boolean);
              return (
                <div className="mt-3 rounded-xl border border-cyan-400/10 bg-black/20 p-2">
                  <div className="mb-2 flex items-center gap-2">
                    <input
                      value={group?.name || ""}
                      onChange={(e) => renameScanGroup(selectedGroupId, e.target.value)}
                      className="min-w-0 flex-1 rounded-lg border border-gray-700 bg-black/30 px-2 py-1.5 text-xs font-bold text-white outline-none focus:border-cyan-400"
                    />
                    <button type="button" onClick={resetSelectedGroupScanResults} className="shrink-0 rounded-lg border border-yellow-400/20 bg-yellow-400/10 px-2 py-1 text-[9px] font-bold text-yellow-300">รีเซ็ตผล</button>
                  </div>

                  <div className="mb-3 flex items-center justify-between gap-2">
                    {/* <span className="text-[8px] text-gray-500">ล็อกตาม Track จริงบนกล้อง</span>   */}
                    <button
                      type="button"
                      onClick={async () => {
                        setGroupSearch("");
                        setShowFacePeopleModal(true);
                        await loadFacePeopleForModal();
                      }}
                      className="w-full shrink-0 rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-3 py-2 text-[10px] font-black text-cyan-300 shadow-[0_0_16px_rgba(34,211,238,0.08)] hover:bg-cyan-400/20"
                    >
                      ＋ เพิ่มสมาชิก
                    </button>
                  </div>

                  <div className="space-y-2">
                    {members.map((person, index) => {
                      const result = group?.scanResults?.[String(person.id)] || null;
                      const status = result?.status || "pending";
                      const statusClass = status === "matched" ? "border-green-400/30 bg-green-500/10" : status === "wrong" ? "border-red-400/30 bg-red-500/10" : status === "unknown" ? "border-yellow-400/30 bg-yellow-500/10" : "border-white/5 bg-white/[0.035]";
                      return (
                        <div key={person.id} className={`flex items-center gap-2 rounded-lg border p-2 ${statusClass}`}>
                          <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sm font-black ${status === "matched" ? "bg-green-400 text-black" : status === "wrong" ? "bg-red-400 text-black" : status === "unknown" ? "bg-yellow-300 text-black" : "bg-cyan-400/10 text-cyan-300"}`}>{status === "matched" ? "✓" : status === "wrong" ? "✕" : status === "unknown" ? "?" : index + 1}</div>
                          {person.frontImage || person.image ? <img src={person.frontImage || person.image} alt="" className="h-8 w-8 rounded-lg object-cover" /> : <div className="h-8 w-8 rounded-lg bg-white/5" />}
                          <div className="min-w-0 flex-1"><div className="truncate text-xs font-bold text-white">{person.name || "ไม่มีชื่อ"}</div><div className="text-[9px] text-gray-400">{status === "matched" ? `ตรวจแล้ว ${formatScanTime(result?.checkedAt)}` : status === "wrong" ? `ตรวจพบแต่ไม่ใช่สมาชิกกลุ่ม • ${formatScanTime(result?.lastSeenAt || result?.checkedAt)}` : status === "unknown" ? `ระบุไม่ได้ • ${formatScanTime(result?.lastSeenAt || result?.checkedAt)}` : "ยังไม่ตรวจ"}</div></div>
                          <button type="button" onClick={() => removePersonFromSelectedGroup(person.id)} className="rounded bg-red-500/10 px-2 py-1 text-[10px] text-red-300">ลบ</button>
                        </div>
                      );
                    })}
                    {!members.length && <div className="p-3 text-center text-[10px] text-gray-500">กลุ่มนี้ยังไม่มีสมาชิก</div>}
                  </div>
                </div>
              );
            })()}

            {!selectedGroupId && <div className="mt-3 rounded-xl border border-dashed border-white/10 p-4 text-center text-[10px] text-gray-500">สร้างหรือเลือกกลุ่มก่อนเริ่มสแกน</div>}

          </section>

          {/* 3D FACE SCAN · 478 จุด
              ซ่อนจากหน้าจอเพื่อลดความรก แต่ยังคง logic ของ depth profile
              ไว้สำหรับการค้นหา/เทียบใบหน้า หากระบบต้องใช้ภายใน
          */}
          <section className="hidden rounded-2xl border border-cyan-400/15 bg-cyan-950/10 p-3 sm:p-4">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-black text-cyan-300">
                  3D FACE SCAN · 478 จุด
                </h2>
                <p className="text-[9px] text-gray-500">
                  ตารางค่า Z สัมพัทธ์ + รูปทรงใบหน้า
                </p>
              </div>
              <span className="rounded-full bg-cyan-400/10 px-2 py-1 text-[9px] text-cyan-300">
                {meshReady ? "READY" : "LOADING"}
              </span>
            </div>

            {faceGeometry?.current ? (
              <>
                <div className="grid grid-cols-3 gap-2">
                  {faceGeometry.current.regions.map((region) => (
                    <div
                      key={region.name}
                      className="rounded-xl bg-black/25 p-2"
                    >
                      <div className="text-[8px] text-gray-500">
                        {region.name}
                      </div>
                      <div className="mt-1 font-mono text-xs text-cyan-200">
                        {region.depth > 0 ? "+" : ""}
                        {region.depth}
                      </div>
                      <div className="text-[8px] text-gray-600">
                        Z {region.z}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-3 grid grid-cols-7 gap-0.5 rounded-xl bg-black/30 p-1">
                  {faceGeometry.current.grid.map((value, index) => (
                    <div
                      key={index}
                      className="aspect-square rounded-sm bg-cyan-400/20"
                      style={{
                        opacity:
                          value == null
                            ? 0.08
                            : 0.25 + Math.min(
                              0.75,
                              Math.abs(value) * 8
                            ),
                      }}
                    />
                  ))}
                </div>

                {faceGeometry.score != null && (
                  <div className="mt-3 text-center text-xs font-black text-cyan-300">
                    Shape Match {faceGeometry.score}%
                  </div>
                )}
              </>
            ) : (
              <div className="rounded-xl border border-dashed border-cyan-400/10 p-6 text-center text-xs text-gray-600">
                หันใบหน้าเข้ากล้องเพื่อสร้าง 3D profile
              </div>
            )}
          </section>
        </div>
      </main>

      {showFacePeopleModal && (
        <div className="fixed inset-0 z-[10050] flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm">
          <div className="flex max-h-[85dvh] w-full max-w-[430px] flex-col overflow-hidden rounded-2xl border border-cyan-400/20 bg-gray-950 shadow-2xl">
            <div className="flex items-center gap-3 border-b border-white/10 px-4 py-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-black text-white">เพิ่มรายชื่อ</div> 
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowFacePeopleModal(false);
                  setGroupSearch("");
                }}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/5 text-lg text-gray-300"
              >
                ×
              </button>
            </div>

            <div className="border-b border-white/10 p-3">
              <input
                autoFocus
                value={groupSearch}
                onChange={(e) => setGroupSearch(e.target.value)}
                placeholder="ค้นหาชื่อ หรือรหัสพนักงาน..."
                className="w-full rounded-xl border border-gray-700 bg-black/30 px-3 py-2.5 text-xs text-white outline-none placeholder:text-gray-600 focus:border-cyan-400"
              />
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
              {facePeopleLoading ? (
                <div className="flex min-h-40 items-center justify-center">
                  <div className="text-xs text-cyan-300">กำลังโหลดรายชื่อจาก API...</div>
                </div>
              ) : (
                renderFacePeopleModalContent()
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
