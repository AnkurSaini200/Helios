import { Incident, IncidentSeverity, IncidentStatus } from "../types";
import { apiRequest, getApiBaseUrl } from "./api";

export interface IncidentFilters {
  event_type?: string;
  severity?: IncidentSeverity | string;
  status?: IncidentStatus | string;
  bus_id?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export async function fetchIncidents(filters: IncidentFilters = {}): Promise<Incident[]> {
  const params = new URLSearchParams();
  if (filters.event_type && filters.event_type !== "all") params.append("event_type", filters.event_type);
  if (filters.severity && filters.severity !== "all") params.append("severity", filters.severity);
  if (filters.status && filters.status !== "all") params.append("status", filters.status);
  if (filters.bus_id) params.append("bus_id", filters.bus_id);
  if (filters.search) params.append("search", filters.search);
  if (filters.limit) params.append("limit", filters.limit.toString());
  if (filters.offset) params.append("offset", filters.offset.toString());

  const query = params.toString() ? `?${params.toString()}` : "";
  return apiRequest<Incident[]>(`/incidents${query}`);
}

export async function fetchIncidentById(incidentId: string): Promise<Incident> {
  return apiRequest<Incident>(`/incidents/${incidentId}`);
}

export async function updateIncident(
  incidentId: string,
  data: { status?: IncidentStatus | string; severity?: IncidentSeverity | string; notes?: string }
): Promise<Incident> {
  return apiRequest<Incident>(`/incidents/${incidentId}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export async function deleteIncident(incidentId: string): Promise<{ status: string; message: string; id: string }> {
  return apiRequest<{ status: string; message: string; id: string }>(`/incidents/${incidentId}`, {
    method: "DELETE",
  });
}


export interface DetectBox {
  class_id: number;
  class_name: string;
  is_crash: boolean;
  confidence: number;
  bbox: number[];
}

export interface DetectUploadResponse {
  success: boolean;
  detected: boolean;
  confidence: number;
  raw_confidence: number;
  severity: string;
  latency_ms: number;
  boxes: DetectBox[];
  image_url: string;
  bus_id: string;
  incident?: Incident;
  message: string;
}

export async function uploadAndDetectImage(
  file: File,
  busId?: string,
  confidenceBoost: number = 0.15,
  forceAlert: boolean = false
): Promise<DetectUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (busId) formData.append("bus_id", busId);
  formData.append("confidence_boost", confidenceBoost.toString());
  formData.append("force_alert", forceAlert.toString());

  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/detect/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Upload detection failed");
  }

  return response.json();
}


// ─── Pothole AI Upload ───────────────────────────────────

export interface PotholeUploadResponse {
  success: boolean;
  detected: boolean;
  confidence: number;
  severity: string;
  latency_ms: number;
  boxes: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bbox: number[];
  }>;
  image_url: string;
  bus_id: string;
  incident?: Incident;
  message: string;
}

export async function uploadAndDetectPothole(
  file: File,
  busId?: string
): Promise<PotholeUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (busId) formData.append("bus_id", busId);

  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/detect/pothole/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Pothole detection failed");
  }

  return response.json();
}


// ─── Waterlogging AI Upload ──────────────────────────────

export interface WaterloggingUploadResponse {
  success: boolean;
  detected: boolean;
  confidence: number;
  severity: string;
  severity_title: string;
  road_coverage_pct: number;
  water_hazard_score: number;
  needs_alert: boolean;
  latency_ms: number;
  image_url: string;
  bus_id: string;
  incident?: Incident;
  message: string;
}

export async function uploadAndDetectWaterlogging(
  file: File,
  busId?: string
): Promise<WaterloggingUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (busId) formData.append("bus_id", busId);

  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/detect/waterlogging/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Waterlogging detection failed");
  }

  return response.json();
}


// ─── Traffic / Vehicle AI Upload ─────────────────────────

export interface TrafficUploadResponse {
  success: boolean;
  vehicles_detected: number;
  density_pct: number;
  total_pcu: number;
  congestion_status: string;
  severity: string;
  breakdown: Record<string, number>;
  latency_ms: number;
  boxes: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bbox: number[];
  }>;
  image_url: string;
  bus_id: string;
  incident?: Incident;
  message: string;
}

export async function uploadAndDetectTraffic(
  file: File,
  busId?: string
): Promise<TrafficUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (busId) formData.append("bus_id", busId);

  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/detect/traffic/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Traffic detection failed");
  }

  return response.json();
}


// ═══════════════════════════════════════════════════════════
//  VIDEO PIPELINE — Multi-Model Priority Inference
// ═══════════════════════════════════════════════════════════

export interface VideoWinner {
  model: string;
  frame_index: number;
  confidence: number;
  severity: string;
  detected: boolean;
  annotated_image_url?: string;
  // Waterlogging-specific
  coverage_pct?: number;
  hazard_score?: number;
  // Accident/Pothole-specific
  detections?: number;
  gps?: { lat: number; lng: number };
  incident_id?: string;
}

export interface VehicleSummary {
  avg_density_pct: number;
  avg_pcu: number;
  total_vehicles_seen: number;
  congestion_status: string;
  breakdown: Record<string, number>;
  annotated_image_url?: string;
}

export interface VideoJobStatus {
  job_id: string;
  status: "processing" | "completed" | "failed";
  phase: string;
  current_model: string | null;
  total_frames: number;
  processed_frames: number;
  frame_division: Record<string, number>;
  winner: VideoWinner | null;
  vehicle_summary: VehicleSummary | null;
  processing_time_ms: number;
  error: string | null;
  bus_id: string;
  filename: string;
}

export async function uploadVideo(file: File, busId?: string): Promise<{ job_id: string; status: string; message: string }> {
  const formData = new FormData();
  formData.append("file", file);
  if (busId) {
    formData.append("bus_id", busId);
  }

  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/detect/video/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Video upload failed");
  }

  return response.json();
}

export function streamVideoStatus(
  jobId: string,
  onUpdate: (data: VideoJobStatus) => void,
  onError?: (err: Event) => void
): EventSource {
  const baseUrl = getApiBaseUrl();
  const es = new EventSource(`${baseUrl}/detect/video/status/${jobId}`);

  es.onmessage = (event) => {
    try {
      const data: VideoJobStatus = JSON.parse(event.data);
      onUpdate(data);
      if (data.status === "completed" || data.status === "failed") {
        es.close();
      }
    } catch {
      // Ignore malformed SSE
    }
  };

  es.onerror = (err) => {
    if (onError) onError(err);
    es.close();
  };

  return es;
}

export async function getVideoResult(jobId: string): Promise<VideoJobStatus> {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/detect/video/result/${jobId}`);

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Failed to get video result");
  }

  return response.json();
}
