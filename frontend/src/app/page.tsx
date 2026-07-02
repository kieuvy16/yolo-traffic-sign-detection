"use client";

import Image from "next/image";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  Clock3,
  Film,
  Gauge,
  Image as ImageIcon,
  Loader2,
  PauseCircle,
  PlayCircle,
  RefreshCcw,
  Search,
  Settings2,
  Square,
  Upload,
  Volume2,
  VolumeX,
  Wifi,
  ListVideo,
} from "lucide-react";

type AppTab = "camera" | "upload" | "video";
type DetectMode = "camera" | "upload" | "video";

type Detection = {
  class: string;
  display_name?: string;
  confidence: number;
  bbox?: number[];
};

type DetectResponse = {
  success?: boolean;
  error?: string;
  image?: string;
  detections?: Detection[];
  count?: number;
  inference_ms?: number;
  imgsz?: number;
};

type HealthResponse = {
  ok?: boolean;
  model_path?: string;
  imgsz?: number;
  conf?: number;
  device?: string;
  request_gap_ms?: number;
};

type RecentEvent = {
  id: string;
  mode: DetectMode;
  label: string;
  confidence: number;
  time: string;
};

type VideoItem = {
  id: string;
  file: File;
  url: string;
  name: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:5000";

const SETTINGS_KEY = "traffic-sign-settings-v4";
const MAX_VIDEO_FILES = 5;

function safeJsonParse<T>(value: string | null | undefined, fallback: T): T {
  if (!value || value === "undefined" || value === "null") return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function formatLatency(value: number | null) {
  if (value == null || Number.isNaN(value)) return "--";
  return `${value.toFixed(1)} ms`;
}

function formatConfidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function normalizeImageSource(value?: string | null) {
  if (!value) return null;
  if (value.startsWith("data:image/")) return value;
  return `data:image/jpeg;base64,${value}`;
}

function uid() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function blobFromCanvas(
  canvas: HTMLCanvasElement,
  quality: number
): Promise<Blob | null> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
  });
}

function MetricCard({
  title,
  value,
  icon,
  tone = "slate",
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  tone?: "emerald" | "blue" | "amber" | "violet" | "rose" | "slate";
}) {
  const toneClass =
    tone === "emerald"
      ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
      : tone === "blue"
      ? "border-blue-400/20 bg-blue-400/10 text-blue-100"
      : tone === "amber"
      ? "border-amber-400/20 bg-amber-400/10 text-amber-100"
      : tone === "violet"
      ? "border-violet-400/20 bg-violet-400/10 text-violet-100"
      : tone === "rose"
      ? "border-rose-400/20 bg-rose-400/10 text-rose-100"
      : "border-white/10 bg-white/5 text-slate-100";

  return (
    <div className={cn("rounded-3xl border p-4 shadow-lg", toneClass)}>
      <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wide opacity-80">
        {icon}
        <span>{title}</span>
      </div>
      <div className="text-3xl font-semibold">{value}</div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-2xl border px-4 py-2.5 text-sm font-medium transition",
        active
          ? "border-cyan-400/40 bg-cyan-400/15 text-cyan-100 shadow-[0_0_0_1px_rgba(3,211,238,0.15)]"
          : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
      )}
    >
      {icon}
      {children}
    </button>
  );
}

function ActionButton({
  onClick,
  disabled,
  icon,
  tone = "primary",
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  icon?: React.ReactNode;
  tone?: "primary" | "secondary" | "danger";
  children: React.ReactNode;
}) {
  const toneClass =
    tone === "primary"
      ? "border-cyan-400/30 bg-cyan-500/15 text-cyan-100 hover:bg-cyan-500/20"
      : tone === "danger"
      ? "border-rose-400/30 bg-rose-500/15 text-rose-100 hover:bg-rose-500/20"
      : "border-white/10 bg-white/5 text-slate-100 hover:bg-white/10";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-semibold transition",
        toneClass,
        disabled && "cursor-not-allowed opacity-50"
      )}
    >
      {icon}
      {children}
    </button>
  );
}

export default function Page() {
  const mountedRef = useRef(false);

  const [activeTab, setActiveTab] = useState<AppTab>("camera");
  const [error, setError] = useState<string | null>(null);
  const [serverOnline, setServerOnline] = useState(false);

  // const [resultImage, setResultImage] = useState<string | null>(null);
  // const [detections, setDetections] = useState<Detection[]>([]);
  // const [recentEvents, setRecentEvents] = useState<RecentEvent[]>([]);
  // const [lastInferenceMs, setLastInferenceMs] = useState<number | null>(null);
  // const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  // const [requestCount, setRequestCount] = useState(0);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const [videoQueue, setVideoQueue] = useState<VideoItem[]>([]);
  const [activeVideoIndex, setActiveVideoIndex] = useState(0);

  const [isLoading, setIsLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraLoopOn, setCameraLoopOn] = useState(false);
  const [videoLoopOn, setVideoLoopOn] = useState(false);
  const [videoPlaying, setVideoPlaying] = useState(false);

  const [captureDelayMs, setCaptureDelayMs] = useState(700);
  const [jpegQuality, setJpegQuality] = useState(0.74);
  const [maxClientWidth, setMaxClientWidth] = useState(960);
  const [autoSpeakMinConfidence, setAutoSpeakMinConfidence] = useState(0.75);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [speechStatus, setSpeechStatus] = useState("Sẵn sàng đọc tiếng Việt");
  const [cameraFacingMode, setCameraFacingMode] = useState<
    "environment" | "user"
  >("environment");

  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speechEnabledRef = useRef(true);
  const lastSpokenRef = useRef<Record<string, number>>({});

  const cameraVideoRef = useRef<HTMLVideoElement>(null);
  const cameraCanvasRef = useRef<HTMLCanvasElement>(null);
  const uploadedVideoRef = useRef<HTMLVideoElement>(null);
  const uploadedVideoCanvasRef = useRef<HTMLCanvasElement>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const cameraLoopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const videoLoopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );

  const cameraLoopEnabledRef = useRef(false);
  const videoLoopEnabledRef = useRef(false);
  const isCameraRequestInFlightRef = useRef(false);
  const isVideoRequestInFlightRef = useRef(false);

  const averageConfidence = useMemo(() => {
    if (!detections.length) return 0;
    return (
      detections.reduce((sum, item) => sum + item.confidence, 0) /
      detections.length
    );
  }, [detections]);

  const topDetection = detections[0] || null;
  const activeVideo = videoQueue[activeVideoIndex] || null;

  useEffect(() => {
    mountedRef.current = true;

    const saved = safeJsonParse(
      typeof window !== "undefined" ? localStorage.getItem(SETTINGS_KEY) : null,
      {
        captureDelayMs: 700,
        jpegQuality: 0.74,
        maxClientWidth: 960,
        autoSpeakMinConfidence: 0.75,
        speechEnabled: true,
        cameraFacingMode: "environment",
      }
    );

    setCaptureDelayMs(
      typeof saved.captureDelayMs === "number" ? saved.captureDelayMs : 700
    );
    setJpegQuality(
      typeof saved.jpegQuality === "number" ? saved.jpegQuality : 0.74
    );
    setMaxClientWidth(
      typeof saved.maxClientWidth === "number" ? saved.maxClientWidth : 960
    );
    setAutoSpeakMinConfidence(
      typeof saved.autoSpeakMinConfidence === "number"
        ? saved.autoSpeakMinConfidence
        : 0.75
    );
    setSpeechEnabled(
      typeof saved.speechEnabled === "boolean" ? saved.speechEnabled : true
    );
    setCameraFacingMode(
      saved.cameraFacingMode === "user" ? "user" : "environment"
    );

    return () => {
      mountedRef.current = false;
      cleanupAllLoops();
      cleanupAudio();
      cleanupCameraStream();
      cleanupVideoUrls(videoQueue);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    speechEnabledRef.current = speechEnabled;
  }, [speechEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        captureDelayMs,
        jpegQuality,
        maxClientWidth,
        autoSpeakMinConfidence,
        speechEnabled,
        cameraFacingMode,
      })
    );
  }, [
    captureDelayMs,
    jpegQuality,
    maxClientWidth,
    autoSpeakMinConfidence,
    speechEnabled,
    cameraFacingMode,
  ]);

  const cleanupAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      if (audioRef.current.src) {
        URL.revokeObjectURL(audioRef.current.src);
      }
      audioRef.current.src = "";
      audioRef.current = null;
    }
  }, []);

  const cleanupCameraStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (cameraVideoRef.current) {
      cameraVideoRef.current.pause();
      cameraVideoRef.current.srcObject = null;
    }
  }, []);

  const cleanupVideoUrls = useCallback((items: VideoItem[]) => {
    items.forEach((item) => URL.revokeObjectURL(item.url));
  }, []);

  const cleanupAllLoops = useCallback(() => {
    cameraLoopEnabledRef.current = false;
    videoLoopEnabledRef.current = false;
    setCameraLoopOn(false);
    setVideoLoopOn(false);

    if (cameraLoopTimeoutRef.current) {
      clearTimeout(cameraLoopTimeoutRef.current);
      cameraLoopTimeoutRef.current = null;
    }
    if (videoLoopTimeoutRef.current) {
      clearTimeout(videoLoopTimeoutRef.current);
      videoLoopTimeoutRef.current = null;
    }
  }, []);

  const pollHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error("Server offline");
      const data: HealthResponse = await response.json();
      setServerOnline(Boolean(data.ok));
    } catch {
      setServerOnline(false);
    }
  }, []);

  useEffect(() => {
    void pollHealth();
    const timer = setInterval(() => void pollHealth(), 5000);
    return () => clearInterval(timer);
  }, [pollHealth]);

  const queueRecentEvent = useCallback(
    (mode: DetectMode, items: Detection[]) => {
      if (!items.length) return;
      const next = items.slice(0, 3).map((item) => ({
        id: uid(),
        mode,
        label: item.display_name || item.class,
        confidence: item.confidence,
        time: new Date().toLocaleTimeString("vi-VN"),
      }));
      setRecentEvents((prev) => [...next, ...prev].slice(0, 10));
    },
    []
  );

  const speakDetectionVietnamese = useCallback(
    async (items: Detection[]) => {
      if (!speechEnabledRef.current) return;
      if (!Array.isArray(items) || items.length === 0) return;

      const validItems = items.filter(
        (item) => (item.confidence ?? 0) >= autoSpeakMinConfidence
      );
      if (!validItems.length) return;

      const labels = Array.from(
        new Set(
          validItems
            .map((item) => item.display_name || item.class)
            .filter(Boolean)
        )
      );
      if (!labels.length) return;

      const now = Date.now();
      const key = labels.join(" | ");
      const lastTime = lastSpokenRef.current[key] || 0;
      if (now - lastTime < 5000) return;
      lastSpokenRef.current[key] = now;

      const speechText = labels.join(", ");

      try {
        setSpeechStatus(`Đang tạo giọng đọc: ${speechText}`);
        const response = await fetch(`${API_BASE_URL}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: speechText }),
        });

        if (!response.ok) {
          throw new Error("Không tạo được âm thanh tiếng Việt");
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        cleanupAudio();
        const audio = new Audio(audioUrl);
        audioRef.current = audio;
        audio.onended = () => setSpeechStatus(`Đã đọc xong: ${speechText}`);
        audio.onerror = () => setSpeechStatus("Lỗi phát âm thanh tiếng Việt");

        await audio.play();
        setSpeechStatus(`Đang đọc: ${speechText}`);
      } catch (err) {
        console.error("Vietnamese TTS error:", err);
        setSpeechStatus("Không đọc được tiếng Việt");
      }
    },
    [autoSpeakMinConfidence, cleanupAudio]
  );

  const applyDetectionResponse = useCallback(
    (mode: DetectMode, data: DetectResponse) => {
      setResultImage(normalizeImageSource(data.image));
      setDetections(Array.isArray(data.detections) ? data.detections : []);
      setLastInferenceMs(
        typeof data.inference_ms === "number" ? data.inference_ms : null
      );
      setRequestCount((prev) => prev + 1);
      setLastUpdatedAt(new Date().toLocaleTimeString("vi-VN"));
      queueRecentEvent(mode, data.detections || []);
      if (data.detections && data.detections.length > 0) {
        void speakDetectionVietnamese(data.detections);
      }
    },
    [queueRecentEvent, speakDetectionVietnamese]
  );

  const sendForDetection = useCallback(
    async (file: Blob | File, mode: DetectMode) => {
      const formData = new FormData();
      formData.append(
        "file",
        file,
        file instanceof File ? file.name : `${mode}.jpg`
      );

      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: "POST",
        body: formData,
        cache: "no-store",
      });

      const data: DetectResponse = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Nhận diện thất bại");
      }

      applyDetectionResponse(mode, data);
      return data;
    },
    [applyDetectionResponse]
  );

  const resizeFrameToCanvas = useCallback(
    (source: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      const width = source.videoWidth;
      const height = source.videoHeight;
      if (!width || !height) return false;

      const scale = width > maxClientWidth ? maxClientWidth / width : 1;
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));

      const ctx = canvas.getContext("2d", { alpha: false });
      if (!ctx) return false;
      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
      return true;
    },
    [maxClientWidth]
  );

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Vui lòng chọn file hình ảnh");
      return;
    }
    setSelectedFile(file);
    setSelectedImage(URL.createObjectURL(file));
    setError(null);
    setResultImage(null);
    setDetections([]);
  }, []);

  const handleVideoSelect = useCallback(
    (files: FileList | File[]) => {
      const videoFiles = Array.from(files).filter((file) =>
        file.type.startsWith("video/")
      );
      if (!videoFiles.length) {
        setError("Vui lòng chọn file video hợp lệ");
        return;
      }

      const limited = videoFiles.slice(0, MAX_VIDEO_FILES);
      cleanupAllLoops();
      cleanupAudio();
      cleanupVideoUrls(videoQueue);

      const nextQueue = limited.map((file) => ({
        id: uid(),
        file,
        url: URL.createObjectURL(file),
        name: file.name,
      }));

      setVideoQueue(nextQueue);
      setActiveVideoIndex(0);
      setVideoPlaying(false);
      setError(null);
      setResultImage(null);
      setDetections([]);
    },
    [cleanupAllLoops, cleanupAudio, cleanupVideoUrls, videoQueue]
  );

  const detectUploadedImage = useCallback(async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    try {
      await sendForDetection(selectedFile, "upload");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không thể nhận diện ảnh");
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, sendForDetection]);

  const captureCameraFrame = useCallback(async () => {
    if (isCameraRequestInFlightRef.current) return;
    const video = cameraVideoRef.current;
    const canvas = cameraCanvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const drawn = resizeFrameToCanvas(video, canvas);
    if (!drawn) return;

    isCameraRequestInFlightRef.current = true;
    setError(null);
    try {
      const blob = await blobFromCanvas(canvas, jpegQuality);
      if (!blob) throw new Error("Không tạo được frame từ camera");
      await sendForDetection(blob, "camera");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Không thể nhận diện camera"
      );
    } finally {
      isCameraRequestInFlightRef.current = false;
    }
  }, [jpegQuality, resizeFrameToCanvas, sendForDetection]);

  const detectVideoFrame = useCallback(async () => {
    if (isVideoRequestInFlightRef.current) return;
    const video = uploadedVideoRef.current;
    const canvas = uploadedVideoCanvasRef.current;
    if (
      !video ||
      !canvas ||
      video.readyState < 2 ||
      video.paused ||
      video.ended
    ) {
      return;
    }

    const drawn = resizeFrameToCanvas(video, canvas);
    if (!drawn) return;

    isVideoRequestInFlightRef.current = true;
    setError(null);
    try {
      const blob = await blobFromCanvas(canvas, jpegQuality);
      if (!blob) throw new Error("Không tạo được frame từ video");
      await sendForDetection(blob, "video");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Không thể nhận diện video"
      );
    } finally {
      isVideoRequestInFlightRef.current = false;
    }
  }, [jpegQuality, resizeFrameToCanvas, sendForDetection]);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Trình duyệt không hỗ trợ camera");
      }

      cleanupCameraStream();

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: cameraFacingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 60 },
          },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        });
      }

      streamRef.current = stream;
      const video = cameraVideoRef.current;
      if (!video) throw new Error("Không tìm thấy thẻ video");

      video.srcObject = stream;
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;

      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => resolve();
      });

      await video.play();
      setCameraOn(true);
    } catch (err) {
      console.error("Camera error:", err);
      setCameraOn(false);
      setError(
        err instanceof Error
          ? err.message
          : "Không thể truy cập camera. Hãy kiểm tra quyền camera và thiết bị đầu vào."
      );
    }
  }, [cameraFacingMode, cleanupCameraStream]);

  const stopCamera = useCallback(() => {
    cameraLoopEnabledRef.current = false;
    setCameraLoopOn(false);
    if (cameraLoopTimeoutRef.current) {
      clearTimeout(cameraLoopTimeoutRef.current);
      cameraLoopTimeoutRef.current = null;
    }
    cleanupAudio();
    cleanupCameraStream();
    setCameraOn(false);
  }, [cleanupAudio, cleanupCameraStream]);

  const runCameraLoop = useCallback(async () => {
    if (!cameraLoopEnabledRef.current || !mountedRef.current) return;
    await captureCameraFrame();
    if (!cameraLoopEnabledRef.current || !mountedRef.current) return;
    cameraLoopTimeoutRef.current = setTimeout(runCameraLoop, captureDelayMs);
  }, [captureCameraFrame, captureDelayMs]);

  const runVideoLoop = useCallback(async () => {
    if (!videoLoopEnabledRef.current || !mountedRef.current) return;
    await detectVideoFrame();
    if (!videoLoopEnabledRef.current || !mountedRef.current) return;
    videoLoopTimeoutRef.current = setTimeout(runVideoLoop, captureDelayMs);
  }, [detectVideoFrame, captureDelayMs]);

  const toggleCameraLoop = useCallback(async () => {
    if (!cameraOn) {
      await startCamera();
    }

    const next = !cameraLoopOn;
    cameraLoopEnabledRef.current = next;
    setCameraLoopOn(next);

    if (!next) {
      if (cameraLoopTimeoutRef.current) {
        clearTimeout(cameraLoopTimeoutRef.current);
        cameraLoopTimeoutRef.current = null;
      }
      return;
    }

    void runCameraLoop();
  }, [cameraLoopOn, cameraOn, runCameraLoop, startCamera]);

  const playCurrentVideo = useCallback(async () => {
    const video = uploadedVideoRef.current;
    if (!video || !activeVideo) return;

    video.src = activeVideo.url;
    video.load();

    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    await video.play();
    setVideoPlaying(true);
  }, [activeVideo]);

  const advanceToNextVideo = useCallback(async () => {
    const nextIndex = activeVideoIndex + 1;
    if (nextIndex >= videoQueue.length) {
      videoLoopEnabledRef.current = false;
      setVideoLoopOn(false);
      setVideoPlaying(false);
      return;
    }

    setActiveVideoIndex(nextIndex);
  }, [activeVideoIndex, videoQueue.length]);

  useEffect(() => {
    if (!activeVideo || !videoLoopOn) return;
    void playCurrentVideo();
  }, [activeVideo, videoLoopOn, playCurrentVideo]);

  const toggleVideoLoop = useCallback(async () => {
    if (!activeVideo) return;

    const next = !videoLoopOn;
    videoLoopEnabledRef.current = next;
    setVideoLoopOn(next);

    if (!next) {
      if (videoLoopTimeoutRef.current) {
        clearTimeout(videoLoopTimeoutRef.current);
        videoLoopTimeoutRef.current = null;
      }
      uploadedVideoRef.current?.pause();
      setVideoPlaying(false);
      cleanupAudio();
      return;
    }

    await playCurrentVideo();
    void runVideoLoop();
  }, [activeVideo, cleanupAudio, playCurrentVideo, runVideoLoop, videoLoopOn]);

  const changeTab = useCallback(
    (tab: AppTab) => {
      setActiveTab(tab);
      setError(null);
      cleanupAudio();
      cleanupAllLoops();
      if (tab !== "camera") {
        setCameraLoopOn(false);
      }
      if (tab !== "video") {
        setVideoLoopOn(false);
        setVideoPlaying(false);
        uploadedVideoRef.current?.pause();
      }
    },
    [cleanupAudio, cleanupAllLoops]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (!files.length) return;

      if (activeTab === "upload") {
        const file = Array.from(files).find((f) => f.type.startsWith("image/"));
        if (file) {
          handleFileSelect(file);
        } else {
          setError("Vui lòng chọn file hình ảnh");
        }
      } else if (activeTab === "video") {
        handleVideoSelect(files);
      }
    },
    [activeTab, handleFileSelect, handleVideoSelect]
  );

  const testVietnameseSpeech = useCallback(async () => {
    await speakDetectionVietnamese([
      {
        class: "test_vi",
        display_name: "Giới hạn tốc độ bốn mươi ki lô mét trên giờ",
        confidence: 0.99,
      },
    ]);
  }, [speakDetectionVietnamese]);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#10224f_0%,_#071227_42%,_#020816_100%)] text-white">
      <div className="mx-auto max-w-[1600px] px-4 py-8 md:px-6 xl:px-8">
        <header className="mb-6 rounded-[2rem] border border-white/10 bg-white/5 p-5 shadow-2xl backdrop-blur-xl">
          <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
            <div>
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1.5 text-sm text-cyan-100">
                <Camera className="h-4 w-4" />
                <span>Traffic Sign Detection Console</span>
              </div>
              <h1 className="text-3xl font-bold leading-tight md:text-5xl">
                Hệ thống nhận diện biển báo giao thông thời gian thực
              </h1>
              <p className="mt-3 max-w-3xl text-sm text-slate-300 md:text-lg"></p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <MetricCard
                title="Server"
                value={serverOnline ? "Online" : "Offline"}
                icon={<Wifi className="h-4 w-4" />}
                tone={serverOnline ? "emerald" : "rose"}
              />
              <MetricCard
                title="Độ trễ"
                value={formatLatency(lastInferenceMs)}
                icon={<Clock3 className="h-4 w-4" />}
                tone="blue"
              />
              <MetricCard
                title="Request"
                value={String(requestCount)}
                icon={<RefreshCcw className="h-4 w-4" />}
                tone="amber"
              />
              <MetricCard
                title="Độ tin cậy TB"
                value={
                  detections.length ? formatConfidence(averageConfidence) : "--"
                }
                icon={<Gauge className="h-4 w-4" />}
                tone="violet"
              />
            </div>
          </div>
        </header>

        <section className="grid gap-6 xl:grid-cols-[1.45fr_0.95fr]">
          <div className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-xl backdrop-blur-xl">
              <div className="mb-4 flex flex-wrap gap-2">
                <TabButton
                  active={activeTab === "camera"}
                  onClick={() => changeTab("camera")}
                  icon={<Camera className="h-4 w-4" />}
                >
                  Camera live
                </TabButton>
                <TabButton
                  active={activeTab === "upload"}
                  onClick={() => changeTab("upload")}
                  icon={<ImageIcon className="h-4 w-4" />}
                >
                  Ảnh tĩnh
                </TabButton>
                <TabButton
                  active={activeTab === "video"}
                  onClick={() => changeTab("video")}
                  icon={<Film className="h-4 w-4" />}
                >
                  Video file
                </TabButton>
              </div>

              {activeTab === "camera" && (
                <div className="space-y-4">
                  <div className="grid gap-4 lg:grid-cols-[1.4fr_0.6fr]">
                    <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/70">
                      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 text-sm text-slate-300">
                        <span className="font-medium">Nguồn camera</span>
                        <span
                          className={cn(
                            "rounded-full px-2.5 py-1 text-xs",
                            cameraOn
                              ? "bg-emerald-400/15 text-emerald-200"
                              : "bg-slate-700/60 text-slate-300"
                          )}
                        >
                          {cameraOn ? "Đang kết nối" : "Chưa bật"}
                        </span>
                      </div>
                      <div className="relative aspect-video bg-black">
                        <video
                          ref={cameraVideoRef}
                          autoPlay
                          playsInline
                          muted
                          className="h-full w-full object-cover"
                        />
                        <canvas ref={cameraCanvasRef} className="hidden" />
                        {!cameraOn && (
                          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-slate-400">
                            <Camera className="h-12 w-12 opacity-60" />
                            <div className="text-center">
                              <p className="text-base font-medium">
                                Camera đang tắt
                              </p>
                              <p className="text-sm text-slate-500">
                                Bấm “Bật camera” để kết nối webcam
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="space-y-4 rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                        <Settings2 className="h-4 w-4 text-cyan-300" />
                        Điều khiển realtime
                      </div>

                      <label className="block rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="text-slate-300">Chu kỳ quét</span>
                          <span className="font-semibold text-white">
                            {captureDelayMs} ms
                          </span>
                        </div>
                        <input
                          type="range"
                          min={250}
                          max={2000}
                          step={50}
                          value={captureDelayMs}
                          onChange={(e) =>
                            setCaptureDelayMs(Number(e.target.value))
                          }
                          className="w-full"
                        />
                      </label>

                      <label className="block rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="text-slate-300">
                            Chất lượng JPEG
                          </span>
                          <span className="font-semibold text-white">
                            {Math.round(jpegQuality * 100)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={40}
                          max={95}
                          step={1}
                          value={Math.round(jpegQuality * 100)}
                          onChange={(e) =>
                            setJpegQuality(Number(e.target.value) / 100)
                          }
                          className="w-full"
                        />
                      </label>

                      <label className="block rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="text-slate-300">
                            Độ rộng frame gửi
                          </span>
                          <span className="font-semibold text-white">
                            {maxClientWidth}px
                          </span>
                        </div>
                        <input
                          type="range"
                          min={480}
                          max={1280}
                          step={40}
                          value={maxClientWidth}
                          onChange={(e) =>
                            setMaxClientWidth(Number(e.target.value))
                          }
                          className="w-full"
                        />
                      </label>

                      <label className="block rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="text-slate-300">Ngưỡng tự đọc</span>
                          <span className="font-semibold text-white">
                            {Math.round(autoSpeakMinConfidence * 100)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={50}
                          max={95}
                          step={1}
                          value={Math.round(autoSpeakMinConfidence * 100)}
                          onChange={(e) =>
                            setAutoSpeakMinConfidence(
                              Number(e.target.value) / 100
                            )
                          }
                          className="w-full"
                        />
                      </label>

                      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="font-semibold text-slate-100">
                            Đọc biển báo tiếng Việt
                          </span>
                          <button
                            onClick={() => setSpeechEnabled((prev) => !prev)}
                            className={cn(
                              "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold transition",
                              speechEnabled
                                ? "bg-emerald-400/15 text-emerald-200"
                                : "bg-slate-700/60 text-slate-300"
                            )}
                          >
                            {speechEnabled ? (
                              <Volume2 className="h-3.5 w-3.5" />
                            ) : (
                              <VolumeX className="h-3.5 w-3.5" />
                            )}
                            {speechEnabled ? "Bật" : "Tắt"}
                          </button>
                        </div>
                        <p className="text-xs text-slate-400">{speechStatus}</p>
                        <button
                          onClick={() => void testVietnameseSpeech()}
                          className="mt-3 inline-flex items-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-500/15 px-3 py-2 text-sm font-semibold text-cyan-100"
                        >
                          <Volume2 className="h-4 w-4" />
                          Test đọc tiếng Việt
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-4">
                    <ActionButton
                      onClick={() => void startCamera()}
                      icon={<Camera className="h-4 w-4" />}
                      tone="secondary"
                    >
                      Bật camera
                    </ActionButton>
                    <ActionButton
                      onClick={toggleCameraLoop}
                      disabled={!serverOnline}
                      icon={
                        cameraLoopOn ? (
                          <PauseCircle className="h-4 w-4" />
                        ) : (
                          <Search className="h-4 w-4" />
                        )
                      }
                      tone="primary"
                    >
                      {cameraLoopOn
                        ? "Dừng quét realtime"
                        : "Bắt đầu quét realtime"}
                    </ActionButton>
                    <ActionButton
                      onClick={() => void captureCameraFrame()}
                      disabled={!cameraOn || cameraLoopOn}
                      icon={<Search className="h-4 w-4" />}
                      tone="secondary"
                    >
                      Chụp một frame
                    </ActionButton>
                    <ActionButton
                      onClick={() =>
                        setCameraFacingMode((prev) =>
                          prev === "environment" ? "user" : "environment"
                        )
                      }
                      icon={<RefreshCcw className="h-4 w-4" />}
                      tone="secondary"
                    >
                      Đổi camera
                    </ActionButton>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2">
                    <ActionButton
                      onClick={stopCamera}
                      disabled={!cameraOn}
                      icon={<Square className="h-4 w-4" />}
                      tone="danger"
                    >
                      Tắt camera
                    </ActionButton>
                    <div className="flex items-center rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
                      Camera đang dùng:
                      <span className="ml-2 font-semibold text-white">
                        {cameraFacingMode === "environment"
                          ? "Mặc định / camera sau"
                          : "Camera trước"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === "upload" && (
                <div className="space-y-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileSelect(file);
                    }}
                  />
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                    className="cursor-pointer rounded-2xl border border-dashed border-white/15 bg-slate-950/40 p-6 text-center transition hover:bg-white/5"
                  >
                    {selectedImage ? (
                      <div>
                        <Image
                          src={selectedImage}
                          alt="Selected"
                          width={900}
                          height={560}
                          className="mx-auto max-h-[420px] w-auto rounded-2xl object-contain"
                        />
                        <p className="mt-4 text-sm text-slate-400">
                          Nhấn để chọn ảnh khác
                        </p>
                      </div>
                    ) : (
                      <div className="py-16">
                        <Upload className="mx-auto mb-4 h-12 w-12 text-slate-500" />
                        <p className="text-lg font-semibold text-slate-200">
                          Kéo thả ảnh vào đây
                        </p>
                        <p className="mt-1 text-sm text-slate-400">
                          hoặc nhấn để chọn file ảnh
                        </p>
                      </div>
                    )}
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <ActionButton
                      onClick={() => void detectUploadedImage()}
                      disabled={!selectedFile || isLoading}
                      icon={
                        isLoading ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Search className="h-4 w-4" />
                        )
                      }
                      tone="primary"
                    >
                      Nhận diện ảnh
                    </ActionButton>
                    <ActionButton
                      onClick={() => fileInputRef.current?.click()}
                      icon={<ImageIcon className="h-4 w-4" />}
                      tone="secondary"
                    >
                      Chọn ảnh khác
                    </ActionButton>
                    <ActionButton
                      onClick={() => {
                        setSelectedFile(null);
                        setSelectedImage(null);
                        cleanupAudio();
                      }}
                      disabled={!selectedFile}
                      icon={<Square className="h-4 w-4" />}
                      tone="secondary"
                    >
                      Xóa ảnh
                    </ActionButton>
                  </div>
                </div>
              )}

              {activeTab === "video" && (
                <div className="space-y-4">
                  <input
                    ref={videoInputRef}
                    type="file"
                    accept="video/*"
                    multiple
                    className="hidden"
                    onChange={(e) => {
                      const files = e.target.files;
                      if (files?.length) handleVideoSelect(files);
                    }}
                  />

                  {videoQueue.length === 0 ? (
                    <div
                      onClick={() => videoInputRef.current?.click()}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={handleDrop}
                      className="cursor-pointer rounded-2xl border border-dashed border-white/15 bg-slate-950/40 p-6 text-center transition hover:bg-white/5"
                    >
                      <div className="py-16">
                        <Film className="mx-auto mb-4 h-12 w-12 text-slate-500" />
                        <p className="text-lg font-semibold text-slate-200">
                          Kéo thả tối đa 5 video vào đây
                        </p>
                        <p className="mt-1 text-sm text-slate-400">
                          hoặc nhấn để chọn nhiều file video
                        </p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="grid gap-4 lg:grid-cols-[1.4fr_0.6fr]">
                        <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/70">
                          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 text-sm text-slate-300">
                            <span className="font-medium">Nguồn video</span>
                            <span
                              className={cn(
                                "rounded-full px-2.5 py-1 text-xs",
                                videoLoopOn
                                  ? "bg-blue-400/15 text-blue-200"
                                  : "bg-slate-700/60 text-slate-300"
                              )}
                            >
                              {videoLoopOn ? "Đang quét" : "Chờ thao tác"}
                            </span>
                          </div>
                          <div className="aspect-video bg-black">
                            <video
                              ref={uploadedVideoRef}
                              className="h-full w-full object-contain"
                              playsInline
                              onEnded={() => {
                                setVideoPlaying(false);
                                if (videoLoopEnabledRef.current) {
                                  void advanceToNextVideo();
                                }
                              }}
                              onPlay={() => setVideoPlaying(true)}
                              onPause={() => setVideoPlaying(false)}
                            />
                            <canvas
                              ref={uploadedVideoCanvasRef}
                              className="hidden"
                            />
                          </div>
                        </div>

                        <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
                            <ListVideo className="h-4 w-4 text-cyan-300" />
                            Danh sách video ({videoQueue.length}/
                            {MAX_VIDEO_FILES})
                          </div>
                          <div className="space-y-2">
                            {videoQueue.map((item, index) => (
                              <button
                                key={item.id}
                                onClick={() => setActiveVideoIndex(index)}
                                className={cn(
                                  "w-full rounded-xl border px-3 py-2 text-left text-sm transition",
                                  index === activeVideoIndex
                                    ? "border-cyan-400/40 bg-cyan-400/15 text-cyan-100"
                                    : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                                )}
                              >
                                <div className="font-medium">
                                  {index + 1}. {item.name}
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="grid gap-3 md:grid-cols-5">
                        <ActionButton
                          onClick={() => videoInputRef.current?.click()}
                          icon={<Upload className="h-4 w-4" />}
                          tone="secondary"
                        >
                          Thêm / đổi video
                        </ActionButton>
                        <ActionButton
                          onClick={toggleVideoLoop}
                          icon={
                            videoLoopOn ? (
                              <PauseCircle className="h-4 w-4" />
                            ) : (
                              <Search className="h-4 w-4" />
                            )
                          }
                          tone="primary"
                        >
                          {videoLoopOn ? "Dừng nhận diện" : "Nhận diện toàn bộ"}
                        </ActionButton>
                        <ActionButton
                          onClick={() => {
                            const el = uploadedVideoRef.current;
                            if (!el) return;
                            if (el.paused) {
                              void el.play();
                            } else {
                              el.pause();
                            }
                          }}
                          icon={
                            videoPlaying ? (
                              <PauseCircle className="h-4 w-4" />
                            ) : (
                              <PlayCircle className="h-4 w-4" />
                            )
                          }
                          tone="secondary"
                        >
                          {videoPlaying ? "Tạm dừng" : "Phát video"}
                        </ActionButton>
                        <ActionButton
                          onClick={() => void detectVideoFrame()}
                          disabled={videoLoopOn}
                          icon={<Camera className="h-4 w-4" />}
                          tone="secondary"
                        >
                          Quét frame hiện tại
                        </ActionButton>
                        <ActionButton
                          onClick={() => {
                            cleanupAllLoops();
                            cleanupAudio();
                            uploadedVideoRef.current?.pause();
                            setVideoPlaying(false);
                            cleanupVideoUrls(videoQueue);
                            setVideoQueue([]);
                            setActiveVideoIndex(0);
                          }}
                          icon={<Square className="h-4 w-4" />}
                          tone="secondary"
                        >
                          Xóa danh sách
                        </ActionButton>
                      </div>
                    </>
                  )}
                </div>
              )}

              {error && (
                <div className="mt-4 flex items-start gap-3 rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                  <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-xl backdrop-blur-xl">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-200">
                    Khung kết quả
                  </p>
                  <p className="text-xs text-slate-400">
                    {lastUpdatedAt
                      ? `Cập nhật lần cuối lúc ${lastUpdatedAt}`
                      : "Chưa có kết quả mới"}
                  </p>
                </div>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-300">
                  {topDetection
                    ? topDetection.display_name || topDetection.class
                    : "No target"}
                </span>
              </div>

              <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/70">
                <div className="aspect-video">
                  {resultImage ? (
                    <Image
                      src={resultImage}
                      alt="Detection result"
                      width={1200}
                      height={720}
                      className="h-full w-full object-contain"
                      unoptimized
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center">
                      <div className="text-center text-slate-500">
                        <Search className="mx-auto mb-3 h-14 w-14 opacity-60" />
                        <p className="text-base font-medium">
                          Kết quả nhận diện sẽ hiển thị tại đây
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-xl backdrop-blur-xl">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-200">
                    Danh sách biển báo
                  </p>
                  <p className="text-xs text-slate-400">
                    Ưu tiên biển có độ tin cậy cao hơn
                  </p>
                </div>
                <span className="rounded-full bg-emerald-400/15 px-3 py-1 text-xs text-emerald-200">
                  {detections.length} mục tiêu
                </span>
              </div>

              <div className="space-y-3">
                {detections.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/40 px-4 py-8 text-center text-slate-400">
                    Chưa có biển báo nào được phát hiện
                  </div>
                ) : (
                  detections.map((item, index) => (
                    <div
                      key={`${item.class}-${index}`}
                      className="rounded-2xl border border-white/10 bg-gradient-to-r from-rose-500/15 to-violet-500/15 p-4"
                    >
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <div>
                          <p className="font-semibold text-white">
                            {item.display_name || item.class}
                          </p>
                          <p className="text-xs text-slate-300">
                            Mã lớp: {item.class}
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-sm text-slate-100">
                          <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                          {formatConfidence(item.confidence)}
                        </div>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-black/30">
                        <div
                          className="h-full rounded-full bg-white/90"
                          style={{
                            width: `${Math.min(100, item.confidence * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-xl backdrop-blur-xl">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-200">
                    Nhật ký phát hiện gần đây
                  </p>
                  <p className="text-xs text-slate-400">
                    Lưu các lần nhận diện thành công gần nhất
                  </p>
                </div>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-300">
                  {recentEvents.length} dòng
                </span>
              </div>

              <div className="space-y-2">
                {recentEvents.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/40 px-4 py-8 text-center text-slate-400">
                    Chưa có sự kiện nào để hiển thị
                  </div>
                ) : (
                  recentEvents.map((item) => (
                    <div
                      key={item.id}
                      className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3"
                    >
                      <div>
                        <p className="text-sm font-medium text-white">
                          {item.label}
                        </p>
                        <p className="text-xs text-slate-400">
                          {item.mode.toUpperCase()} • {item.time}
                        </p>
                      </div>
                      <div className="text-sm font-semibold text-cyan-100">
                        {formatConfidence(item.confidence)}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
