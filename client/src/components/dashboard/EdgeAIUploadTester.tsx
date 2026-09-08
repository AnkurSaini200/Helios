import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  UploadCloud,
  Cpu,
  Sparkles,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  FileImage,
  Layers,
  Zap,
  Droplets,
  Car,
  Gauge,
  Activity,
  Radio,
  ExternalLink,
  Video,
  Play,
  Loader2,
  Trophy,
  BarChart3,
  Clock,
} from "lucide-react";
import { Card } from "../ui/Card";
import { Badge } from "../ui/Badge";
import {
  uploadAndDetectImage,
  uploadAndDetectPothole,
  uploadAndDetectWaterlogging,
  uploadAndDetectTraffic,
  uploadVideo,
  streamVideoStatus,
  getVideoResult,
  DetectUploadResponse,
  PotholeUploadResponse,
  WaterloggingUploadResponse,
  TrafficUploadResponse,
  VideoJobStatus,
} from "../../services/incidents";
import { Incident, Bus } from "../../types";
import { fetchBuses } from "../../services/buses";
type ModelMode = "accident" | "pothole" | "waterlogging" | "traffic";

interface ModelMeta {
  id: ModelMode;
  name: string;
  badge: string;
  tag: string;
  desc: string;
  icon: React.ElementType;
  activeColor: string;
}

const MODELS: ModelMeta[] = [
  {
    id: "accident",
    name: "Accidents / SOS",
    badge: "best.pt",
    tag: "Crash & Collision",
    desc: "YOLOv8n-Accident-EdgeNet real-time collision detection with telemetry deceleration fusion",
    icon: ShieldAlert,
    activeColor: "bg-red-500/20 text-red-300 border-red-500/50 shadow-glow-emergency",
  },
  {
    id: "pothole",
    name: "Pothole Monitor",
    badge: "yolo26n_pothole.pt",
    tag: "Road Surface",
    desc: "YOLO26n road surface damage detection with bounding boxes & severity classification",
    icon: AlertTriangle,
    activeColor: "bg-amber-500/20 text-amber-300 border-amber-500/50",
  },
  {
    id: "waterlogging",
    name: "Waterlogging / Flood",
    badge: "best.pt (Seg)",
    tag: "Flood Mask",
    desc: "YOLOv8 Instance Segmentation for road water coverage % & flood hazard scoring",
    icon: Droplets,
    activeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50",
  },
  {
    id: "traffic",
    name: "Traffic & Vehicles",
    badge: "yolo26n.pt",
    tag: "Vehicle Flow",
    desc: "Multi-class vehicle counting (Car, Bus, Truck, Bike) & PCU congestion density engine",
    icon: Car,
    activeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/50",
  },
];

function formatBoxName(name: string): string {
  if (!name || name === "-" || name.startsWith("class_")) return "Traffic Collision";
  return name;
}

interface EdgeAIUploadTesterProps {
  onOpenDossier?: (incident: Incident) => void;
}

export const EdgeAIUploadTester: React.FC<EdgeAIUploadTesterProps> = ({ onOpenDossier }) => {
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState<ModelMode>("accident");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Per-model result states
  const [accidentResult, setAccidentResult] = useState<DetectUploadResponse | null>(null);
  const [potholeResult, setPotholeResult] = useState<PotholeUploadResponse | null>(null);
  const [waterlogResult, setWaterlogResult] = useState<WaterloggingUploadResponse | null>(null);
  const [trafficResult, setTrafficResult] = useState<TrafficUploadResponse | null>(null);

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [boostDecel, setBoostDecel] = useState(true);

  // ── Video Pipeline State ──
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewName, setVideoPreviewName] = useState<string | null>(null);
  const [isVideoUploading, setIsVideoUploading] = useState(false);
  const [videoJob, setVideoJob] = useState<VideoJobStatus | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Bus Selection State
  const [buses, setBuses] = useState<Bus[]>([]);
  const [selectedBusId, setSelectedBusId] = useState<string>("");

  // ── Live Dashcam Pipeline State ──
  const [dashcamUrl, setDashcamUrl] = useState("http://10.2.43.57:8080/video");
  const [isDashcamActive, setIsDashcamActive] = useState(false);
  const [dashcamJobId, setDashcamJobId] = useState<string | null>(null);

  useEffect(() => {
    fetchBuses().then(setBuses).catch(console.error);
  }, []);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeModelMeta = MODELS.find((m) => m.id === selectedModel) || MODELS[0];

  const clearResults = () => {
    setAccidentResult(null);
    setPotholeResult(null);
    setWaterlogResult(null);
    setTrafficResult(null);
    setErrorMsg(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
  };

  const processSelectedFile = (file: File) => {
    setSelectedFile(file);
    clearResults();

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      processSelectedFile(file);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;

    try {
      setIsUploading(true);
      setErrorMsg(null);

      if (selectedModel === "accident") {
        const res = await uploadAndDetectImage(
          selectedFile,
          undefined,
          boostDecel ? 0.15 : 0.0,
          false
        );
        setAccidentResult(res);
      } else if (selectedModel === "pothole") {
        const res = await uploadAndDetectPothole(selectedFile);
        setPotholeResult(res);
      } else if (selectedModel === "waterlogging") {
        const res = await uploadAndDetectWaterlogging(selectedFile);
        setWaterlogResult(res);
      } else if (selectedModel === "traffic") {
        const res = await uploadAndDetectTraffic(selectedFile);
        setTrafficResult(res);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Failed to analyze image with YOLO model");
    } finally {
      setIsUploading(false);
    }
  };

  // Get current active result
  const currentResult =
    selectedModel === "accident"
      ? accidentResult
      : selectedModel === "pothole"
      ? potholeResult
      : selectedModel === "waterlogging"
      ? waterlogResult
      : trafficResult;

  const currentImageUrl = currentResult?.image_url || previewUrl;

  // ── Video Pipeline Handlers ──
  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("video/")) {
      setVideoFile(file);
      setVideoPreviewName(file.name);
      setVideoJob(null);
      setVideoError(null);
    }
  };

  const handleVideoDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("video/")) {
      setVideoFile(file);
      setVideoPreviewName(file.name);
      setVideoJob(null);
      setVideoError(null);
    }
  };

  const handleRunVideoPipeline = async () => {
    if (!videoFile) return;
    try {
      setIsVideoUploading(true);
      setVideoError(null);
      setVideoJob(null);

      const { job_id } = await uploadVideo(videoFile, selectedBusId || undefined);

      // Start SSE stream
      if (eventSourceRef.current) eventSourceRef.current.close();

      const es = streamVideoStatus(
        job_id,
        (data) => {
          setVideoJob(data);
          if (data.status === "completed" || data.status === "failed") {
            setIsVideoUploading(false);
            if (data.status === "failed") {
              setVideoError(data.error || "Pipeline failed");
            }
          }
        },
        () => {
          // SSE error — fallback to polling result
          setTimeout(async () => {
            try {
              const result = await getVideoResult(job_id);
              setVideoJob(result);
            } catch {
              setVideoError("Lost connection to pipeline");
            } finally {
              setIsVideoUploading(false);
            }
          }, 2000);
        }
      );
      eventSourceRef.current = es;
    } catch (err: any) {
      setVideoError(err.message || "Failed to start video pipeline");
      setIsVideoUploading(false);
    }
  };

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  const getModelDisplayName = (model: string) => {
    const names: Record<string, string> = {
      waterlogging: "Waterlogging / Flood",
      accident: "Accident / SOS",
      pothole: "Pothole Detection",
      vehicle_detection: "Vehicle / Traffic",
    };
    return names[model] || model;
  };

  const getModelColor = (model: string) => {
    const colors: Record<string, string> = {
      waterlogging: "emerald",
      accident: "red",
      pothole: "amber",
      vehicle_detection: "cyan",
    };
    return colors[model] || "slate";
  };

  const videoProgress = videoJob
    ? videoJob.total_frames > 0
      ? Math.round((videoJob.processed_frames / videoJob.total_frames) * 100)
      : 0
    : 0;

  return (
    <Card
      title="Edge AI Live Model Verification & Custom Image Lab"
      subtitle={activeModelMeta.desc}
      action={
        <div className="flex items-center gap-2">
          <Badge variant="solar" size="sm">
            <Cpu className="w-3 h-3 mr-1" />
            YOLO Weights: {activeModelMeta.badge}
          </Badge>
        </div>
      }
    >
      {/* Model Selection Tabs */}
      <div className="mb-5 grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-1.5 rounded-2xl bg-helios-950/90 border border-slate-800">
        {MODELS.map((model) => {
          const Icon = model.icon;
          const isSelected = selectedModel === model.id;
          return (
            <button
              key={model.id}
              onClick={() => {
                setSelectedModel(model.id);
                setErrorMsg(null);
              }}
              className={`px-3 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center justify-center gap-2 border cursor-pointer ${
                isSelected
                  ? model.activeColor
                  : "bg-helios-850/60 text-slate-400 border-transparent hover:text-slate-200 hover:bg-helios-850"
              }`}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span className="truncate">{model.name}</span>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Dropzone & Model-Specific Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
              previewUrl
                ? "border-solar-500/50 bg-helios-900/60"
                : "border-slate-700/80 hover:border-solar-500/60 bg-helios-850/60 hover:bg-helios-850"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            {previewUrl ? (
              <div className="relative w-full aspect-video rounded-xl overflow-hidden border border-slate-700 bg-black/60 group">
                <img
                  src={previewUrl}
                  alt="Upload Preview"
                  className="w-full h-full object-contain"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center font-mono text-xs text-white">
                  Click to choose different image
                </div>
              </div>
            ) : (
              <div className="py-6 space-y-2.5">
                <div className="w-12 h-12 mx-auto rounded-full bg-solar-500/10 border border-solar-500/20 text-solar-400 flex items-center justify-center">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-xs font-bold font-mono text-slate-200">
                    Upload Frame to Test {activeModelMeta.tag}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1">
                    Supports JPG, PNG, WEBP, or BMP
                  </div>
                </div>
                <div className="inline-block px-3 py-1 rounded-lg bg-slate-800 text-[10px] font-mono text-slate-300 border border-slate-700">
                  Browse from Computer
                </div>
              </div>
            )}
          </div>

          {/* Model-Specific Context Controls */}
          {selectedModel === "accident" && (
            <div className="p-3 rounded-xl bg-helios-850/80 border border-slate-800 flex items-center justify-between font-mono text-xs">
              <div>
                <div className="font-bold text-slate-200 flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-solar-400" />
                  Telemetry Deceleration Signal
                </div>
                <div className="text-[10px] text-slate-400">
                  Fuse sudden brake G-force telemetry (+15% confidence boost)
                </div>
              </div>
              <button
                type="button"
                onClick={() => setBoostDecel(!boostDecel)}
                className={`px-2.5 py-1 rounded-md text-[10px] font-bold uppercase transition-colors cursor-pointer ${
                  boostDecel
                    ? "bg-solar-500 text-helios-950"
                    : "bg-slate-800 text-slate-400 border border-slate-700"
                }`}
              >
                {boostDecel ? "ACTIVE" : "OFF"}
              </button>
            </div>
          )}

          {selectedModel === "pothole" && (
            <div className="p-3 rounded-xl bg-helios-850/80 border border-slate-800 flex items-center justify-between font-mono text-xs">
              <div>
                <div className="font-bold text-amber-300 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  Road Surface Depth Profiling
                </div>
                <div className="text-[10px] text-slate-400">
                  YOLO26n bounding box + severity threshold (Confidence &gt; 25%)
                </div>
              </div>
              <span className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
                ACTIVE
              </span>
            </div>
          )}

          {selectedModel === "waterlogging" && (
            <div className="p-3 rounded-xl bg-helios-850/80 border border-slate-800 flex items-center justify-between font-mono text-xs">
              <div>
                <div className="font-bold text-emerald-300 flex items-center gap-1.5">
                  <Droplets className="w-3.5 h-3.5 text-emerald-400" />
                  Instance Segmentation ROI
                </div>
                <div className="text-[10px] text-slate-400">
                  Road surface polygon mask + green overlay + flood hazard score
                </div>
              </div>
              <span className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ACTIVE
              </span>
            </div>
          )}

          {selectedModel === "traffic" && (
            <div className="p-3 rounded-xl bg-helios-850/80 border border-slate-800 flex items-center justify-between font-mono text-xs">
              <div>
                <div className="font-bold text-emerald-300 flex items-center gap-1.5">
                  <Car className="w-3.5 h-3.5 text-emerald-400" />
                  PCU Density Weighting
                </div>
                <div className="text-[10px] text-slate-400">
                  Passenger Car Unit calculations: Bus (3.5), Truck (3.0), Car (1.0), Moto (0.5)
                </div>
              </div>
              <span className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ACTIVE
              </span>
            </div>
          )}

          {/* Inference Trigger Button */}
          <button
            onClick={handleRunInference}
            disabled={!selectedFile || isUploading}
            className={`w-full py-2.5 px-4 rounded-xl font-mono text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all cursor-pointer ${
              !selectedFile
                ? "bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed"
                : isUploading
                ? "bg-solar-500/50 text-helios-950 cursor-wait"
                : "bg-gradient-to-r from-solar-500 via-amber-400 to-solar-500 hover:from-solar-400 hover:to-solar-500 text-helios-950 shadow-glow-solar"
            }`}
          >
            {isUploading ? (
              <>
                <span className="w-4 h-4 border-2 border-helios-950 border-t-transparent rounded-full animate-spin" />
                Running {activeModelMeta.name} Inference...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Run Edge AI Detection ({activeModelMeta.tag})
              </>
            )}
          </button>

          {errorMsg && (
            <div className="p-3 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-mono flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Right Column: Visual Result with Bounding Boxes & Diagnostics (7 cols) */}
        <div className="lg:col-span-7 flex flex-col justify-between space-y-4">
          {currentResult ? (
            <div className="space-y-3.5">
              {/* ─── ACCIDENT RESULT ─── */}
              {selectedModel === "accident" && accidentResult && (
                <>
                  <div
                    className={`flex flex-wrap items-center justify-between gap-2 p-3.5 rounded-xl font-mono border transition-all ${
                      accidentResult.detected
                        ? "bg-red-950/40 border-red-500/50 shadow-glow-emergency"
                        : "bg-helios-850 border-slate-800"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          accidentResult.detected ? "bg-red-500 animate-ping" : "bg-emerald-400"
                        }`}
                      />
                      <span className={`text-xs font-bold ${accidentResult.detected ? "text-red-200" : "text-white"}`}>
                        {accidentResult.detected ? "ACCIDENT DETECTED!" : "NO ACCIDENT DETECTED"}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <Badge variant={accidentResult.detected ? "danger" : "neutral"} size="sm">
                        {accidentResult.severity.toUpperCase()} SEVERITY
                      </Badge>
                      <span className="text-[11px] text-solar-400 font-bold">
                        Confidence: {Math.round(accidentResult.confidence * 100)}%
                      </span>
                      <span className="text-[10px] text-slate-400">
                        ({accidentResult.latency_ms} ms)
                      </span>
                    </div>
                  </div>

                  <div className="relative rounded-xl overflow-hidden border-2 border-slate-700 bg-black aspect-video max-h-72 flex items-center justify-center">
                    <img
                      src={currentImageUrl || accidentResult.image_url}
                      alt="Accident Detection Result"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 left-2 bg-helios-950/80 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono text-solar-400 border border-solar-500/30 flex items-center gap-1.5">
                      <Layers className="w-3 h-3" />
                      YOLOv8 Accident Overlay Active
                    </div>
                    {accidentResult.detected && (
                      <div className="absolute top-2 right-2 bg-red-600/90 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono text-white font-bold border border-red-400/40 flex items-center gap-1.5 animate-pulse">
                        <ShieldAlert className="w-3 h-3" />
                        EMERGENCY DETECTED
                      </div>
                    )}
                  </div>

                  <div className="p-3.5 rounded-xl bg-helios-850/60 border border-slate-800 space-y-2.5 font-mono text-xs">
                    <div className="flex items-center justify-between text-[10px] uppercase text-slate-400 font-bold">
                      <span>Detected Classes ({accidentResult.boxes.length} found):</span>
                      {accidentResult.detected && (
                        <span className="text-red-400 font-bold flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping inline-block" />
                          Live Incident Created
                        </span>
                      )}
                    </div>
                    {accidentResult.boxes.length === 0 ? (
                      <div className="text-slate-500 text-[11px]">No bounding boxes found.</div>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {accidentResult.boxes.map((box, idx) => {
                          const cleanName = formatBoxName(box.class_name);
                          return (
                            <span
                              key={idx}
                              className={`px-2.5 py-1 rounded-lg text-[11px] border font-bold flex items-center gap-1.5 ${
                                box.is_crash || accidentResult.detected
                                  ? "bg-red-500/20 text-red-200 border-red-500/40"
                                  : "bg-slate-800 text-slate-300 border-slate-700"
                              }`}
                            >
                              <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
                              <span>{cleanName}:</span>
                              <span className="text-solar-400">{Math.round(box.confidence * 100)}%</span>
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {accidentResult.incident && (
                    <div className="flex flex-wrap items-center justify-between gap-2 pt-1 font-mono text-xs">
                      <span className="text-[11px] text-slate-400">
                        Incident ID: <strong className="text-white">{accidentResult.incident.id}</strong> (Bus: {accidentResult.bus_id})
                      </span>

                      <div className="flex items-center gap-2">
                        <a
                          href="/accidents"
                          className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold uppercase tracking-wider flex items-center gap-1 border border-slate-700 transition-colors cursor-pointer"
                        >
                          <ExternalLink className="w-3.5 h-3.5" />
                          View on Dashboard
                        </a>
                        {onOpenDossier && (
                          <button
                            onClick={() => onOpenDossier(accidentResult.incident!)}
                            className="px-3.5 py-1.5 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold text-xs uppercase tracking-wider flex items-center gap-1.5 shadow-glow-emergency transition-all cursor-pointer"
                          >
                            <ShieldAlert className="w-3.5 h-3.5" />
                            Open Emergency Dossier
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* ─── POTHOLE RESULT ─── */}
              {selectedModel === "pothole" && potholeResult && (
                <>
                  <div className="flex flex-wrap items-center justify-between gap-2 p-3.5 rounded-xl bg-helios-850 border border-slate-800 font-mono">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          potholeResult.detected ? "bg-amber-400 animate-pulse" : "bg-emerald-400"
                        }`}
                      />
                      <span className="text-xs font-bold text-white">
                        {potholeResult.detected
                          ? `${potholeResult.boxes.length} POTHOLE(S) DETECTED!`
                          : "NO POTHOLES DETECTED"}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <Badge variant={potholeResult.detected ? "warning" : "neutral"} size="sm">
                        {potholeResult.severity.toUpperCase()} SEVERITY
                      </Badge>
                      <span className="text-[11px] text-amber-400 font-bold">
                        Top Conf: {Math.round(potholeResult.confidence * 100)}%
                      </span>
                      <span className="text-[10px] text-slate-400">
                        ({potholeResult.latency_ms} ms)
                      </span>
                    </div>
                  </div>

                  <div className="relative rounded-xl overflow-hidden border-2 border-slate-700 bg-black aspect-video max-h-72 flex items-center justify-center">
                    <img
                      src={currentImageUrl || potholeResult.image_url}
                      alt="Pothole Detection Result"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 left-2 bg-helios-950/80 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono text-amber-400 border border-amber-500/30 flex items-center gap-1.5">
                      <Layers className="w-3 h-3" />
                      YOLO26n Pothole Bounding Boxes Active
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-helios-850/60 border border-slate-800 space-y-2 font-mono text-xs">
                    <div className="text-[10px] uppercase text-slate-400 font-bold">
                      Detected Pothole Instances ({potholeResult.boxes.length} found):
                    </div>
                    {potholeResult.boxes.length === 0 ? (
                      <div className="text-slate-500 text-[11px]">Road surface is clean. No potholes found.</div>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {potholeResult.boxes.map((box, idx) => (
                          <span
                            key={idx}
                            className="px-2.5 py-1 rounded-lg text-[11px] border font-bold flex items-center gap-1 bg-amber-500/20 text-amber-300 border-amber-500/40"
                          >
                            <AlertTriangle className="w-3 h-3 text-amber-400" />
                            Pothole #{idx + 1}: {Math.round(box.confidence * 100)}%
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {potholeResult.incident && (
                    <div className="flex items-center justify-between pt-1 font-mono text-xs text-slate-400">
                      <span>
                        Incident ID: <strong className="text-white">{potholeResult.incident.id}</strong> (Bus: {potholeResult.bus_id})
                      </span>
                      <span className="text-amber-400 font-bold">Road Hazard Logged to Database</span>
                    </div>
                  )}
                </>
              )}

              {/* ─── WATERLOGGING RESULT ─── */}
              {selectedModel === "waterlogging" && waterlogResult && (
                <>
                  <div className="flex flex-wrap items-center justify-between gap-2 p-3.5 rounded-xl bg-helios-850 border border-slate-800 font-mono">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          waterlogResult.detected ? "bg-emerald-400 animate-pulse" : "bg-emerald-400"
                        }`}
                      />
                      <span className="text-xs font-bold text-white">
                        {waterlogResult.detected
                          ? `WATERLOGGING: ${waterlogResult.severity_title.toUpperCase()}`
                          : "NO SIGNIFICANT WATERLOGGING"}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <Badge variant={waterlogResult.detected ? "success" : "neutral"} size="sm">
                        {waterlogResult.severity.toUpperCase()}
                      </Badge>
                      <span className="text-[11px] text-emerald-400 font-bold">
                        Coverage: {waterlogResult.road_coverage_pct}%
                      </span>
                      <span className="text-[10px] text-slate-400">
                        ({waterlogResult.latency_ms} ms)
                      </span>
                    </div>
                  </div>

                  <div className="relative rounded-xl overflow-hidden border-2 border-slate-700 bg-black aspect-video max-h-72 flex items-center justify-center">
                    <img
                      src={currentImageUrl || waterlogResult.image_url}
                      alt="Waterlogging Segmentation Result"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 left-2 bg-helios-950/80 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
                      <Droplets className="w-3 h-3" />
                      YOLOv8-Seg Waterlogging Mask Active
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-helios-850/60 border border-slate-800 space-y-3 font-mono text-xs">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-[10px] uppercase text-slate-400 font-bold mb-1">
                          Road Water Coverage
                        </div>
                        <div className="text-base font-bold text-emerald-400">
                          {waterlogResult.road_coverage_pct}%
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-1.5 mt-1.5 overflow-hidden">
                          <div
                            className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, waterlogResult.road_coverage_pct * 2)}%` }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="text-[10px] uppercase text-slate-400 font-bold mb-1">
                          Flood Hazard Index
                        </div>
                        <div className="text-base font-bold text-white">
                          {waterlogResult.water_hazard_score} <span className="text-xs text-slate-500">/ 100</span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-1.5 mt-1.5 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              waterlogResult.water_hazard_score > 60
                                ? "bg-red-500"
                                : waterlogResult.water_hazard_score > 30
                                ? "bg-amber-500"
                                : "bg-emerald-500"
                            }`}
                            style={{ width: `${Math.min(100, waterlogResult.water_hazard_score)}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="pt-1 text-[11px] text-slate-400 flex items-center justify-between">
                      <span>Status: <strong className="text-white">{waterlogResult.severity_title}</strong></span>
                      {waterlogResult.needs_alert && (
                        <span className="text-red-400 font-bold flex items-center gap-1">
                          <Radio className="w-3 h-3 animate-ping" /> Drainage Alert Required
                        </span>
                      )}
                    </div>
                  </div>

                  {waterlogResult.incident && (
                    <div className="flex items-center justify-between pt-1 font-mono text-xs text-slate-400">
                      <span>
                        Incident ID: <strong className="text-white">{waterlogResult.incident.id}</strong> (Bus: {waterlogResult.bus_id})
                      </span>
                      <span className="text-emerald-400 font-bold">Broadcast via WebSocket</span>
                    </div>
                  )}
                </>
              )}

              {/* ─── TRAFFIC RESULT ─── */}
              {selectedModel === "traffic" && trafficResult && (
                <>
                  <div className="flex flex-wrap items-center justify-between gap-2 p-3.5 rounded-xl bg-helios-850 border border-slate-800 font-mono">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
                      <span className="text-xs font-bold text-white">
                        TRAFFIC FLOW: {trafficResult.congestion_status.toUpperCase()}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <Badge variant="success" size="sm">
                        {trafficResult.severity.toUpperCase()}
                      </Badge>
                      <span className="text-[11px] text-emerald-400 font-bold">
                        Density: {Math.round(trafficResult.density_pct)}%
                      </span>
                      <span className="text-[10px] text-slate-400">
                        ({trafficResult.latency_ms} ms)
                      </span>
                    </div>
                  </div>

                  <div className="relative rounded-xl overflow-hidden border-2 border-slate-700 bg-black aspect-video max-h-72 flex items-center justify-center">
                    <img
                      src={currentImageUrl || trafficResult.image_url}
                      alt="Traffic Flow Analysis Result"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 left-2 bg-helios-950/80 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
                      <Car className="w-3 h-3" />
                      YOLO26n Vehicle Classification Active
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-helios-850/60 border border-slate-800 space-y-3 font-mono text-xs">
                    <div className="text-[10px] uppercase text-slate-400 font-bold flex items-center justify-between">
                      <span>Vehicle Breakdown ({trafficResult.vehicles_detected} total detected):</span>
                      <span className="text-emerald-400">{trafficResult.total_pcu} PCU</span>
                    </div>

                    <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                      {Object.entries(trafficResult.breakdown || {}).map(([vType, count]) => (
                        <div
                          key={vType}
                          className="p-2 rounded-lg bg-helios-900 border border-slate-800 text-center"
                        >
                          <div className="text-[10px] text-slate-400 capitalize">{vType}</div>
                          <div className="text-sm font-bold text-white mt-0.5">{count}</div>
                        </div>
                      ))}
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>Congestion Density Index</span>
                        <span className="text-white font-bold">{trafficResult.density_pct}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            trafficResult.density_pct > 75
                              ? "bg-red-500"
                              : trafficResult.density_pct > 45
                              ? "bg-amber-500"
                              : "bg-emerald-500"
                          }`}
                          style={{ width: `${Math.min(100, trafficResult.density_pct)}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {trafficResult.incident && (
                    <div className="flex items-center justify-between pt-1 font-mono text-xs text-slate-400">
                      <span>
                        Telemetry ID: <strong className="text-white">{trafficResult.incident.id}</strong> (Bus: {trafficResult.bus_id})
                      </span>
                      <span className="text-emerald-400 font-bold">Traffic Telemetry Synchronized</span>
                    </div>
                  )}
                </>
              )}
            </div>
          ) : (
            <div className="h-full min-h-[260px] rounded-2xl border border-slate-800/80 bg-helios-900/40 flex flex-col items-center justify-center p-8 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-slate-800/80 text-slate-400 flex items-center justify-center">
                <Cpu className="w-6 h-6" />
              </div>
              <div className="max-w-sm space-y-1">
                <h4 className="text-xs font-bold font-mono text-slate-300 uppercase">
                  Awaiting {activeModelMeta.tag} Frame Input
                </h4>
                <p className="text-[11px] text-slate-500 font-sans">
                  Select an image on the left and click "Run Edge AI Detection" to run real inference using{" "}
                  <code className="text-solar-400 font-mono">{activeModelMeta.badge}</code>.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  VIDEO PIPELINE SECTION                                    */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <div className="mt-8 pt-6 border-t border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/30 flex items-center justify-center">
              <Video className="w-4 h-4 text-violet-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold font-mono text-white">Video Analysis Pipeline</h3>
              <p className="text-[10px] text-slate-400 font-mono">Multi-model priority inference on video frames — Waterlogging → Accident → Pothole → Vehicle</p>
            </div>
          </div>
          <Badge variant="solar" size="sm">
            <BarChart3 className="w-3 h-3 mr-1" />
            4-Model Cascade
          </Badge>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* ── Left: Video Upload Zone ── */}
          <div className="lg:col-span-5 space-y-4">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleVideoDrop}
              onClick={() => videoInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
                videoPreviewName
                  ? "border-violet-500/50 bg-helios-900/60"
                  : "border-slate-700/80 hover:border-violet-500/60 bg-helios-850/60 hover:bg-helios-850"
              }`}
            >
              <input
                ref={videoInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleVideoFileChange}
              />

              {videoPreviewName ? (
                <div className="space-y-2 py-2">
                  <div className="w-14 h-14 mx-auto rounded-xl bg-violet-500/10 border border-violet-500/30 text-violet-400 flex items-center justify-center">
                    <Video className="w-7 h-7" />
                  </div>
                  <div className="text-xs font-bold font-mono text-violet-300 truncate max-w-[240px]">
                    {videoPreviewName}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    {videoFile && `${(videoFile.size / (1024 * 1024)).toFixed(1)} MB`} · Click to change
                  </div>
                </div>
              ) : (
                <div className="py-6 space-y-2.5">
                  <div className="w-12 h-12 mx-auto rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 flex items-center justify-center">
                    <Video className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="text-xs font-bold font-mono text-slate-200">
                      Upload Video for Multi-Model Analysis
                    </div>
                    <div className="text-[11px] text-slate-400 mt-1">
                      Supports MP4, AVI, MOV, WebM (max 100 MB)
                    </div>
                  </div>
                  <div className="inline-block px-3 py-1 rounded-lg bg-slate-800 text-[10px] font-mono text-slate-300 border border-slate-700">
                    Browse from Computer
                  </div>
                </div>
              )}
            </div>

            {/* Frame Division Info */}
            <div className="p-3 rounded-xl bg-helios-850/80 border border-slate-800 font-mono text-xs space-y-2">
              <div className="font-bold text-violet-300 flex items-center gap-1.5 text-[11px]">
                <Layers className="w-3.5 h-3.5 text-violet-400" />
                Frame Distribution Strategy
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { name: "Waterlogging", pct: "17%", color: "emerald" },
                  { name: "Accident", pct: "33%", color: "red" },
                  { name: "Pothole", pct: "33%", color: "amber" },
                  { name: "Vehicle Det.", pct: "17%", color: "cyan" },
                ].map((m) => (
                  <div key={m.name} className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-helios-900 border border-slate-800">
                    <span className="text-[10px] text-slate-300">{m.name}</span>
                    <span className={`text-[10px] font-bold text-${m.color}-400`}>{m.pct}</span>
                  </div>
                ))}
              </div>
              <div className="text-[10px] text-slate-500 pt-1">
                Priority: If ≥60% confidence found, lower-priority models skipped. Vehicle always runs.
              </div>
            </div>

            {/* Bus Selector */}
            <div className="flex flex-col gap-2 pt-2 pb-2">
              <label className="text-xs font-mono text-slate-400 uppercase">Simulate from Bus (For GPS Context)</label>
              <div className="relative group">
                <select
                  className="w-full appearance-none bg-helios-850 border border-slate-700 text-white text-sm rounded-xl pl-3 pr-8 py-2.5 focus:outline-none focus:border-indigo-500 hover:border-slate-500 cursor-pointer transition-colors"
                  value={selectedBusId}
                  onChange={(e) => setSelectedBusId(e.target.value)}
                  disabled={isVideoUploading}
                >
                  <option value="">Default (BUS-HYD-VID)</option>
                  {buses.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.id} ({b.route})
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-400 group-hover:text-white">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                </div>
              </div>
            </div>

            {/* Run Pipeline Button */}
            <button
              onClick={handleRunVideoPipeline}
              disabled={!videoFile || isVideoUploading}
              className={`w-full py-2.5 px-4 rounded-xl font-mono text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all cursor-pointer ${
                !videoFile
                  ? "bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed"
                  : isVideoUploading
                  ? "bg-violet-500/50 text-white cursor-wait"
                  : "bg-gradient-to-r from-violet-500 via-purple-400 to-violet-500 hover:from-violet-400 hover:to-violet-500 text-white shadow-lg shadow-violet-500/20"
              }`}
            >
              {isVideoUploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing Video Pipeline...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Video Pipeline (4-Model Cascade)
                </>
              )}
            </button>

            {videoError && (
              <div className="p-3 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-mono flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{videoError}</span>
              </div>
            )}
          </div>

          {/* ── Right: Pipeline Progress & Results ── */}
          <div className="lg:col-span-7 flex flex-col space-y-4">
            {videoJob ? (
              <div className="space-y-4">
                {/* Progress Bar */}
                {videoJob.status === "processing" && (
                  <div className="p-4 rounded-xl bg-helios-850/80 border border-violet-500/30 space-y-3">
                    <div className="flex items-center justify-between font-mono text-xs">
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                        <span className="text-violet-300 font-bold uppercase">
                          {videoJob.phase === "extracting_frames" && "Extracting Frames..."}
                          {videoJob.phase === "dividing_frames" && "Dividing Frames Among Models..."}
                          {videoJob.phase === "running_waterlogging" && "Running Waterlogging Model..."}
                          {videoJob.phase === "running_accident" && "Running Accident Model..."}
                          {videoJob.phase === "running_pothole" && "Running Pothole Model..."}
                          {videoJob.phase === "running_vehicle_detection" && "Running Vehicle Detection..."}
                          {videoJob.phase === "winner_found" && "Winner Found! Finishing..."}
                          {videoJob.phase === "initializing" && "Initializing Pipeline..."}
                        </span>
                      </div>
                      <span className="text-slate-400">
                        {videoJob.processed_frames}/{videoJob.total_frames} frames
                      </span>
                    </div>

                    <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-violet-500 to-purple-400 h-full rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${videoProgress}%` }}
                      />
                    </div>

                    {/* Frame Division Display */}
                    {Object.keys(videoJob.frame_division).length > 0 && (
                      <div className="flex flex-wrap gap-2 text-[10px] font-mono">
                        {Object.entries(videoJob.frame_division).map(([model, count]) => (
                          <span
                            key={model}
                            className={`px-2 py-0.5 rounded-md border ${
                              videoJob.current_model === model
                                ? "bg-violet-500/20 border-violet-500/50 text-violet-300"
                                : "bg-slate-800/60 border-slate-700 text-slate-500"
                            }`}
                          >
                            {model}: {count as number}f
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* ── WINNER CARD ── */}
                {videoJob.status === "completed" && (
                  <>
                    {videoJob.winner ? (
                      <div className={`p-4 rounded-xl border space-y-3 ${
                        videoJob.winner.model === "waterlogging"
                          ? "bg-emerald-950/40 border-emerald-500/50"
                          : videoJob.winner.model === "accident"
                          ? "bg-red-950/40 border-red-500/50 shadow-glow-emergency"
                          : "bg-amber-950/40 border-amber-500/50"
                      }`}>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Trophy className={`w-5 h-5 ${
                              videoJob.winner.model === "waterlogging" ? "text-emerald-400" :
                              videoJob.winner.model === "accident" ? "text-red-400" : "text-amber-400"
                            }`} />
                            <div>
                              <div className="text-xs font-bold font-mono text-white uppercase">
                                {getModelDisplayName(videoJob.winner.model)} — DETECTED
                              </div>
                              <div className="text-[10px] text-slate-400 font-mono">
                                Frame #{videoJob.winner.frame_index} · Priority Winner
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={videoJob.winner.severity === "critical" ? "danger" : "warning"} size="sm">
                              {videoJob.winner.severity.toUpperCase()}
                            </Badge>
                            <span className={`text-sm font-bold font-mono ${
                              videoJob.winner.model === "waterlogging" ? "text-emerald-400" :
                              videoJob.winner.model === "accident" ? "text-red-400" : "text-amber-400"
                            }`}>
                              {Math.round(videoJob.winner.confidence * 100)}%
                            </span>
                          </div>
                        </div>

                        {videoJob.winner.annotated_image_url && (
                          <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-black aspect-video max-h-64 flex items-center justify-center">
                            <img
                              src={videoJob.winner.annotated_image_url}
                              alt={`${videoJob.winner.model} Detection`}
                              className="w-full h-full object-contain"
                            />
                            <div className={`absolute top-2 left-2 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono border flex items-center gap-1.5 ${
                              videoJob.winner.model === "waterlogging"
                                ? "bg-emerald-950/80 text-emerald-400 border-emerald-500/30"
                                : videoJob.winner.model === "accident"
                                ? "bg-red-950/80 text-red-400 border-red-500/30"
                                : "bg-amber-950/80 text-amber-400 border-amber-500/30"
                            }`}>
                              <Trophy className="w-3 h-3" />
                              Priority Winner Frame
                            </div>
                          </div>
                        )}

                        {/* GPS Coordinates and Map Link */}
                        {videoJob.winner?.gps && (
                          <div className="grid grid-cols-2 gap-3 font-mono text-xs mt-3">
                            <div className="p-2 rounded-lg bg-helios-900 border border-slate-800 flex flex-col justify-center">
                              <div className="text-[10px] text-slate-400">GPS Coordinates</div>
                              <div className="text-sm font-bold text-white">
                                {videoJob.winner?.gps?.lat.toFixed(4)}, {videoJob.winner?.gps?.lng.toFixed(4)}
                              </div>
                            </div>
                            <button
                              onClick={() => navigate(`/map?lat=${videoJob.winner?.gps?.lat}&lng=${videoJob.winner?.gps?.lng}&highlight=${videoJob.winner?.incident_id || videoJob.job_id}`)}
                              className="p-2 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/40 border border-indigo-500/50 flex items-center justify-center gap-2 transition-colors cursor-pointer text-indigo-300 font-bold"
                            >
                              <ExternalLink className="w-4 h-4" />
                              View on Live Map
                            </button>
                          </div>
                        )}

                        {/* Extra metrics for waterlogging */}
                        {videoJob.winner.model === "waterlogging" && videoJob.winner.coverage_pct !== undefined && (
                          <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                            <div className="p-2 rounded-lg bg-helios-900 border border-slate-800">
                              <div className="text-[10px] text-slate-400">Road Coverage</div>
                              <div className="text-sm font-bold text-emerald-400">{videoJob.winner.coverage_pct}%</div>
                            </div>
                            <div className="p-2 rounded-lg bg-helios-900 border border-slate-800">
                              <div className="text-[10px] text-slate-400">Hazard Score</div>
                              <div className="text-sm font-bold text-white">{videoJob.winner.hazard_score}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="p-4 rounded-xl bg-helios-850 border border-emerald-500/30 flex items-center gap-3">
                        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                        <div>
                          <div className="text-xs font-bold font-mono text-emerald-300">ROAD CLEAR — No Hazard Detected</div>
                          <div className="text-[10px] text-slate-400 font-mono">
                            No model triggered ≥60% confidence across all analyzed frames
                          </div>
                        </div>
                      </div>
                    )}

                    {/* ── VEHICLE SUMMARY (Always shown) ── */}
                    {videoJob.vehicle_summary && (
                      <div className="p-4 rounded-xl bg-helios-850/80 border border-cyan-500/30 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Car className="w-4 h-4 text-cyan-400" />
                            <span className="text-xs font-bold font-mono text-white uppercase">Vehicle / Traffic Summary</span>
                          </div>
                          <span className="text-[11px] font-mono font-bold text-cyan-400">
                            {videoJob.vehicle_summary.congestion_status}
                          </span>
                        </div>

                        <div className="grid grid-cols-3 gap-3">
                          <div className="p-2 rounded-lg bg-helios-900 border border-slate-800 text-center">
                            <div className="text-[10px] text-slate-400 font-mono">Avg Density</div>
                            <div className="text-sm font-bold text-white font-mono">{videoJob.vehicle_summary.avg_density_pct}%</div>
                          </div>
                          <div className="p-2 rounded-lg bg-helios-900 border border-slate-800 text-center">
                            <div className="text-[10px] text-slate-400 font-mono">Total Vehicles</div>
                            <div className="text-sm font-bold text-white font-mono">{videoJob.vehicle_summary.total_vehicles_seen}</div>
                          </div>
                          <div className="p-2 rounded-lg bg-helios-900 border border-slate-800 text-center">
                            <div className="text-[10px] text-slate-400 font-mono">Avg PCU</div>
                            <div className="text-sm font-bold text-white font-mono">{videoJob.vehicle_summary.avg_pcu}</div>
                          </div>
                        </div>

                        {/* Vehicle Breakdown */}
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(videoJob.vehicle_summary.breakdown || {}).map(([vType, count]) => (
                            <div key={vType} className="px-2.5 py-1 rounded-lg bg-helios-900 border border-slate-800 text-[11px] font-mono">
                              <span className="text-slate-400 capitalize">{vType}:</span>{" "}
                              <span className="text-white font-bold">{count}</span>
                            </div>
                          ))}
                        </div>

                        {/* Density Bar */}
                        <div>
                          <div className="flex justify-between text-[10px] font-mono text-slate-400 mb-1">
                            <span>Congestion Index</span>
                            <span className="text-white font-bold">{videoJob.vehicle_summary.avg_density_pct}%</span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                videoJob.vehicle_summary.avg_density_pct > 75
                                  ? "bg-red-500"
                                  : videoJob.vehicle_summary.avg_density_pct > 45
                                  ? "bg-amber-500"
                                  : "bg-emerald-500"
                              }`}
                              style={{ width: `${Math.min(100, videoJob.vehicle_summary.avg_density_pct)}%` }}
                            />
                          </div>
                        </div>

                        {videoJob.vehicle_summary.annotated_image_url && (
                          <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-black aspect-video max-h-48 flex items-center justify-center">
                            <img
                              src={videoJob.vehicle_summary.annotated_image_url}
                              alt="Traffic Analysis"
                              className="w-full h-full object-contain"
                            />
                            <div className="absolute top-2 left-2 bg-helios-950/80 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono text-cyan-400 border border-cyan-500/30 flex items-center gap-1.5">
                              <Car className="w-3 h-3" />
                              Peak Traffic Frame
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Processing Stats */}
                    <div className="flex items-center justify-between font-mono text-[10px] text-slate-500 px-1">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Processed in {(videoJob.processing_time_ms / 1000).toFixed(1)}s
                      </span>
                      <span>{videoJob.total_frames} frames analyzed · {videoJob.filename}</span>
                    </div>
                  </>
                )}

                {/* Failed State */}
                {videoJob.status === "failed" && (
                  <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-400" />
                    <div>
                      <div className="text-xs font-bold font-mono text-red-300">PIPELINE FAILED</div>
                      <div className="text-[10px] text-red-400/80 font-mono">{videoJob.error}</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full min-h-[260px] rounded-2xl border border-slate-800/80 bg-helios-900/40 flex flex-col items-center justify-center p-8 text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 flex items-center justify-center">
                  <Video className="w-6 h-6" />
                </div>
                <div className="max-w-sm space-y-1">
                  <h4 className="text-xs font-bold font-mono text-slate-300 uppercase">
                    Awaiting Video Input
                  </h4>
                  <p className="text-[11px] text-slate-500 font-sans">
                    Upload a dashcam or road surveillance video. The pipeline will extract frames and run
                    4 AI models in priority order, returning the highest-confidence hazard detection along
                    with full vehicle traffic analysis.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  LIVE DASHCAM PIPELINE (IP CAMERA)                           */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <div className="mt-8 border-t border-slate-800/80 pt-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold font-mono text-white flex items-center gap-2">
                <Video className="w-4 h-4 text-solar-400" />
                Live Dashcam Pipeline (IP Stream)
              </h3>
              <p className="text-[11px] text-slate-500 font-sans mt-1">
                Connect directly to an IP Camera (e.g. Android IP Webcam) for continuous real-time analysis using the 4-Model Cascade. Detections will automatically pop up on the Live Map globally!
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded border border-solar-500/30 bg-solar-500/10 text-solar-400 text-[10px] font-bold font-mono uppercase">
                Continuous Stream
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-4">
              <div className="space-y-1">
                <label className="text-[10px] font-bold font-mono text-slate-400 uppercase">IP Camera Stream URL</label>
                <input
                  type="text"
                  value={dashcamUrl}
                  onChange={(e) => setDashcamUrl(e.target.value)}
                  disabled={isDashcamActive}
                  placeholder="http://192.168.1.100:8080/video"
                  className="w-full bg-helios-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 font-mono focus:border-solar-500 focus:outline-none transition-colors disabled:opacity-50"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-bold font-mono text-slate-400 uppercase">Simulate From Bus (GPS Context)</label>
                <select
                  value={selectedBusId}
                  onChange={(e) => setSelectedBusId(e.target.value)}
                  disabled={isDashcamActive}
                  className="w-full bg-helios-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 font-mono focus:border-solar-500 focus:outline-none transition-colors disabled:opacity-50"
                >
                  <option value="">Default (BUS-HYD-VID)</option>
                  {buses.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.id} - {b.route}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={() => setIsDashcamActive(!isDashcamActive)}
                className={`w-full py-3 rounded-xl font-bold font-mono text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg ${
                  isDashcamActive
                    ? "bg-red-500 hover:bg-red-600 text-white shadow-red-500/20"
                    : "bg-solar-500 hover:bg-solar-400 text-helios-950 shadow-solar-500/20"
                }`}
              >
                {isDashcamActive ? (
                  <>Stop Live Dashcam</>
                ) : (
                  <>
                    <Video className="w-4 h-4" /> Connect & Stream
                  </>
                )}
              </button>
            </div>

            <div className="lg:col-span-2">
              {isDashcamActive ? (
                <div className="relative rounded-2xl border-2 border-solar-500/50 bg-black overflow-hidden shadow-2xl shadow-solar-500/10 min-h-[300px] flex items-center justify-center">
                  <div className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-red-500/30">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-[10px] font-bold font-mono text-red-400 uppercase">Live Pipeline Active</span>
                  </div>
                  {/* MJPEG Stream directly from FastAPI */}
                  <img
                    src={`http://localhost:8000/api/v1/dashcam/stream?url=${encodeURIComponent(dashcamUrl)}&bus_id=${selectedBusId || "BUS-HYD-VID"}`}
                    alt="Live Dashcam Stream"
                    className="w-full h-full object-contain"
                    onError={(e) => {
                      // Fallback if stream fails
                      e.currentTarget.style.display = 'none';
                      setIsDashcamActive(false);
                      alert("Failed to connect to IP Camera stream. Is the URL correct and accessible?");
                    }}
                  />
                </div>
              ) : (
                <div className="h-full min-h-[300px] rounded-2xl border border-slate-800/80 bg-helios-900/40 flex flex-col items-center justify-center p-8 text-center space-y-3">
                  <div className="w-12 h-12 rounded-full bg-solar-500/10 border border-solar-500/20 text-solar-400 flex items-center justify-center">
                    <Video className="w-6 h-6" />
                  </div>
                  <div className="max-w-sm space-y-1">
                    <h4 className="text-xs font-bold font-mono text-slate-300 uppercase">
                      Dashcam Disconnected
                    </h4>
                    <p className="text-[11px] text-slate-500 font-sans">
                      Enter an IP Camera MJPEG stream URL to begin real-time analysis. The 4-Model Cascade will process frames live and push verified incidents directly to the global database.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};
