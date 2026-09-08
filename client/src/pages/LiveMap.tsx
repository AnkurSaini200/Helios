import React, { useEffect, useState, useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
MapContainer,
Marker,
Popup,
useMap,
} from "react-leaflet";
import L from "leaflet";
import { setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import "@maplibre/maplibre-gl-leaflet";
import {
Bus as BusIcon,
Siren,
Activity,
Droplets,
Milestone,
BarChart3,
Layers,
Filter,
Maximize2,
ExternalLink,
Navigation,
} from "lucide-react";
import { useHeliosWebSocket } from "../context/WebSocketContext";
import { fetchBuses } from "../services/buses";
import { fetchIncidents } from "../services/incidents";
import { Bus, Incident } from "../types";

setWorkerUrl(workerUrl);

// Helper component to recenter or pan map
const MapController: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
const map = useMap();
useEffect(() => {
map.setView(center, zoom);
}, [center, zoom, map]);
return null;
};

// OpenFreeMap + OpenStreetMap vector map layer
const OpenFreeMapLayer: React.FC = () => {
const map = useMap();

useEffect(() => {
const maplibreLayer = L.maplibreGL({
style: "https://tiles.openfreemap.org/styles/liberty",
});


maplibreLayer.addTo(map);

return () => {
  map.removeLayer(maplibreLayer);
};


}, [map]);

return null;
};

// Create custom colored Leaflet HTML markers
const createCustomIcon = (
colorBg: string,
borderColor: string,
label: string,
isPulsing = false
) => {
return L.divIcon({
className: "custom-leaflet-marker",
html: `      <div style="
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: ${colorBg};
        border: 2px solid ${borderColor};
        box-shadow: 0 0 12px ${colorBg};
        cursor: pointer;
      ">
        ${
          isPulsing
            ?`<div style="
             position: absolute;
             width: 100%;
             height: 100%;
             border-radius: 50%;
             background-color: ${colorBg};
             opacity: 0.7;
             animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;
           "></div>`             : ""
        }         <span style="font-size: 14px; font-weight: bold; color: #fff;">${label}</span>       </div>
    `,
iconSize: [32, 32],
iconAnchor: [16, 16],
popupAnchor: [0, -18],
});
};

export const LiveMap: React.FC = () => {
const navigate = useNavigate();
const [searchParams] = useSearchParams();
const { subscribe } = useHeliosWebSocket();

const [buses, setBuses] = useState<Bus[]>([]);
const [incidents, setIncidents] = useState<Incident[]>([]);
const [selectedFilter, setSelectedFilter] = useState<string>("all");

const queryLat = searchParams.get("lat");
const queryLng = searchParams.get("lng");
const queryHighlight = searchParams.get("highlight");

const initialLat = queryLat ? parseFloat(queryLat) : 17.4412;
const initialLng = queryLng ? parseFloat(queryLng) : 78.3921;

const [mapCenter, setMapCenter] = useState<[number, number]>([initialLat, initialLng]);
const [mapZoom, setMapZoom] = useState<number>(queryHighlight ? 16 : 13);
const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);

useEffect(() => {
const loadInitialData = async () => {
try {
const [busData, incData] = await Promise.all([
fetchBuses(),
fetchIncidents({ limit: 500 }),
]);
setBuses(busData);
setIncidents(incData);
} catch (err) {
console.error("Error loading map data", err);
}
};


loadInitialData();

// Listen for live GPS telemetry
const unsubGPS = subscribe("gps_updated", (updatedPositions: any[]) => {
  setBuses((prev) =>
    prev.map((bus) => {
      const update = updatedPositions.find((u) => u.id === bus.id);
      return update
        ? { ...bus, lat: update.lat, lng: update.lng, speed: update.speed, status: update.status }
        : bus;
    })
  );
});

// Listen for live incident additions
const unsubIncident = subscribe("incident_created", (newInc: Incident) => {
  setIncidents((prev) => [newInc, ...prev]);
});

return () => {
  unsubGPS();
  unsubIncident();
};


}, [subscribe]);

// Filtered markers
const filteredBuses = useMemo(() => {
if (selectedFilter === "all" || selectedFilter === "buses") return buses;
return [];
}, [buses, selectedFilter]);

const filteredIncidents = useMemo(() => {
if (selectedFilter === "all") return incidents;
if (selectedFilter === "buses") return [];
return incidents.filter((inc) => {
if (selectedFilter === "accidents") return inc.event_type === "accident";
if (selectedFilter === "potholes") return inc.event_type === "pothole";
if (selectedFilter === "waterlogging") return inc.event_type === "waterlogging";
if (selectedFilter === "traffic") return inc.event_type === "traffic";
if (selectedFilter === "road_issues") return inc.event_type === "road_sign";
return true;
});
}, [incidents, selectedFilter]);

const getIncidentIcon = (inc: Incident) => {
switch (inc.event_type) {
case "accident":
return createCustomIcon("#ef4444", "#f87171", "!", true);
case "pothole":
return createCustomIcon("#f59e0b", "#fde68a", "P", false);
case "waterlogging":
return createCustomIcon("#06b6d4", "#a5f3fc", "W", false);
case "traffic":
return createCustomIcon("#a855f7", "#e9d5ff", "T", false);
case "road_sign":
default:
return createCustomIcon("#eab308", "#fef08a", "S", false);
}
};

const getBusIcon = (bus: Bus) => {
const isOnline = bus.status === "online";
const isWarning = bus.status === "warning";
const color = isWarning ? "#ef4444" : isOnline ? "#10b981" : "#64748b";
return createCustomIcon(color, "#fff", "B", isWarning);
};

const filterButtons = [
{ id: "all", label: "All Telemetry", count: buses.length + incidents.length },
{ id: "buses", label: "Active Buses", count: buses.length, color: "text-emerald-400" },
{ id: "accidents", label: "Accidents", count: incidents.filter((i) => i.event_type === "accident").length, color: "text-red-400" },
{ id: "potholes", label: "Potholes", count: incidents.filter((i) => i.event_type === "pothole").length, color: "text-amber-400" },
{ id: "waterlogging", label: "Waterlogging", count: incidents.filter((i) => i.event_type === "waterlogging").length, color: "text-cyan-400" },
{ id: "traffic", label: "Traffic", count: incidents.filter((i) => i.event_type === "traffic").length, color: "text-purple-400" },
{ id: "road_issues", label: "Road Issues", count: incidents.filter((i) => i.event_type === "road_sign").length, color: "text-yellow-400" },
];

return ( <div className="space-y-4">
{/* Header and Filter Pills */} <div className="flex flex-wrap items-center justify-between gap-3 bg-helios-900/90 p-4 rounded-xl border border-slate-800 backdrop-blur-md"> <div className="flex items-center gap-2"> <Filter className="w-4 h-4 text-solar-400" /> <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
Map Layers & Filters: </span> </div>
    <div className="flex flex-wrap items-center gap-1.5">
      {filterButtons.map((btn) => (
        <button
          key={btn.id}
          onClick={() => setSelectedFilter(btn.id)}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
            selectedFilter === btn.id
              ? "bg-solar-500 text-helios-950 shadow-glow-solar"
              : "bg-helios-850 text-slate-300 hover:text-white border border-slate-800"
          }`}
        >
          <span className={btn.color}>{btn.label}</span>
          <span className="px-1 py-0.2 text-[10px] rounded bg-black/30 font-bold">
            {btn.count}
          </span>
        </button>
      ))}
    </div>

    <div className="flex flex-wrap items-center gap-1.5">
      <button
        onClick={() => {
          setMapCenter([17.4412, 78.3921]);
          setMapZoom(13);
        }}
        className="px-3 py-1.5 rounded-lg bg-helios-850 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-mono flex items-center gap-1.5 cursor-pointer"
      >
        <Navigation className="w-3.5 h-3.5 text-solar-400" />
        Center Cyberabad
      </button>

      <button
        onClick={() => {
          setMapCenter([16.5062, 80.6480]);
          setMapZoom(13);
        }}
        className="px-3 py-1.5 rounded-lg bg-helios-850 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-mono flex items-center gap-1.5 cursor-pointer"
      >
        <Navigation className="w-3.5 h-3.5 text-solar-400" />
        Vijayawada
      </button>

      <button
        onClick={() => {
          setMapCenter([16.4308, 80.5684]);
          setMapZoom(13);
        }}
        className="px-3 py-1.5 rounded-lg bg-helios-850 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-mono flex items-center gap-1.5 cursor-pointer"
      >
        <Navigation className="w-3.5 h-3.5 text-solar-400" />
        Mangalagiri
      </button>
    </div>
  </div>

  {/* Main Map Container */}
  <div className="relative w-full h-[640px] rounded-2xl overflow-hidden border border-slate-800 bg-helios-950 shadow-2xl">
    <MapContainer
      center={mapCenter}
      zoom={mapZoom}
      scrollWheelZoom={true}
      className="w-full h-full"
      minZoom={1}
    >
      <MapController center={mapCenter} zoom={mapZoom} />

      <OpenFreeMapLayer />

      {/* Render Active Buses */}
      {filteredBuses.map((bus) => (
        <Marker
          key={`bus-${bus.id}`}
          position={[bus.lat, bus.lng]}
          icon={getBusIcon(bus)}
        >
          <Popup>
            <div className="p-1 min-w-[200px] text-xs font-sans">
              <div className="flex items-center justify-between pb-1 mb-1 border-b border-slate-700">
                <span className="font-bold font-mono text-white text-sm">
                  {bus.id}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase font-bold ${
                    bus.status === "online"
                      ? "bg-emerald-500/20 text-emerald-400"
                      : "bg-red-500/20 text-red-400"
                  }`}
                >
                  {bus.status}
                </span>
              </div>
              <div className="text-slate-300 font-medium mb-2">{bus.route}</div>
              <div className="grid grid-cols-2 gap-1 text-[11px] font-mono text-slate-400">
                <div>Speed: <span className="text-white font-bold">{bus.speed} km/h</span></div>
                <div>Battery: <span className="text-emerald-400 font-bold">{bus.battery}%</span></div>
                <div>Jetson: <span className="text-solar-400 font-bold">{bus.jetson_temp}°C</span></div>
                <div>Driver: <span className="text-slate-200">{bus.driver_name.split(" ")[0]}</span></div>
              </div>
              <button
                onClick={() => navigate(`/buses/${bus.id}`)}
                className="w-full mt-3 py-1.5 rounded bg-solar-500 hover:bg-solar-600 text-helios-950 font-mono font-bold text-[11px] uppercase tracking-wider flex items-center justify-center gap-1 cursor-pointer"
              >
                View Bus Telemetry <ExternalLink className="w-3 h-3" />
              </button>
            </div>
          </Popup>
        </Marker>
      ))}

      {/* Render Incidents */}
      {filteredIncidents.map((inc) => (
        <Marker
          key={`inc-${inc.id}`}
          position={[inc.gps.lat, inc.gps.lng]}
          icon={getIncidentIcon(inc)}
        >
          <Popup>
            <div className="p-1 min-w-[220px] text-xs font-sans">
              <div className="flex items-center justify-between pb-1 mb-1 border-b border-slate-700">
                <span className="font-bold font-mono text-white text-sm">
                  {inc.id}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase font-bold ${
                    inc.event_type === "accident"
                      ? "bg-red-500/20 text-red-400"
                      : inc.event_type === "pothole"
                      ? "bg-amber-500/20 text-amber-400"
                      : "bg-cyan-500/20 text-cyan-400"
                  }`}
                >
                  {inc.event_type}
                </span>
              </div>

              {inc.image_url && (
                <div className="mb-2 rounded overflow-hidden bg-black max-h-48 flex items-center justify-center">
                  <img
                    src={inc.image_url}
                    alt="Detection"
                    className="w-full max-h-48 object-contain"
                  />
                </div>
              )}

              <div className="space-y-1 text-[11px] font-mono text-slate-300">
                <div>Bus Source: <span className="font-bold text-white">{inc.bus_id}</span></div>
                <div>Confidence: <span className="text-solar-400 font-bold">{Math.round(inc.confidence * 100)}%</span></div>
                <div>Severity: <span className="text-red-400 uppercase font-bold">{inc.severity}</span></div>
                <div>Time: <span className="text-slate-400">{new Date(inc.timestamp).toLocaleTimeString()}</span></div>
              </div>

              <button
                onClick={() => {
                  if (inc.event_type === "accident") {
                    navigate("/accidents");
                  } else {
                    navigate(`/incidents?search=${inc.id}`);
                  }
                }}
                className="w-full mt-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-100 font-mono font-bold text-[11px] uppercase tracking-wider flex items-center justify-center gap-1 cursor-pointer border border-slate-700"
              >
                View Incident Details <ExternalLink className="w-3 h-3 text-solar-400" />
              </button>
            </div>
          </Popup>
        </Marker>
      ))}

      {/* Render Highlight Marker from Video Pipeline */}
      {queryHighlight && queryLat && queryLng && (
        <Marker
          position={[parseFloat(queryLat), parseFloat(queryLng)]}
          icon={L.divIcon({
            className: "bg-transparent",
            html: `
              <div class="relative group cursor-pointer w-12 h-12 flex items-center justify-center">
                <div class="absolute inset-0 bg-indigo-500 rounded-full animate-ping opacity-50"></div>
                <div class="relative w-8 h-8 rounded-full shadow-lg border-2 bg-indigo-500 border-white text-white flex items-center justify-center shadow-indigo-500/50">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>
                </div>
              </div>
            `,
            iconSize: [48, 48],
            iconAnchor: [24, 24],
            popupAnchor: [0, -24],
          })}
          zIndexOffset={1000}
        >
          <Popup className="helios-popup">
            <div className="font-mono text-sm min-w-[220px]">
              <div className="flex items-center gap-2 mb-2 pb-2 border-b border-indigo-500/20">
                <div className="p-1.5 rounded-md bg-indigo-500/20">
                  <span className="text-indigo-400 font-bold">LIVE ANALYSIS</span>
                </div>
                <div className="flex flex-col">
                  <span className="font-bold text-white uppercase truncate">Video Pipeline Result</span>
                </div>
              </div>
              
              {(() => {
                const highlightedIncident = incidents.find(inc => inc.id === queryHighlight);
                if (highlightedIncident) {
                  return (
                    <div className="text-xs font-sans">
                      {highlightedIncident.image_url && (
                        <div className="mb-2 rounded overflow-hidden bg-black max-h-48 flex items-center justify-center">
                          <img
                            src={highlightedIncident.image_url}
                            alt="Detection"
                            className="w-full max-h-48 object-contain"
                          />
                        </div>
                      )}
                      <div className="space-y-1 font-mono text-[11px] text-slate-300">
                        <div>Bus Source: <span className="font-bold text-white">{highlightedIncident.bus_id}</span></div>
                        <div>Confidence: <span className="text-solar-400 font-bold">{Math.round(highlightedIncident.confidence * 100)}%</span></div>
                        <div>Severity: <span className={`font-bold uppercase ${highlightedIncident.severity === 'critical' ? 'text-red-400' : highlightedIncident.severity === 'high' ? 'text-orange-400' : 'text-amber-400'}`}>{highlightedIncident.severity}</span></div>
                        <div>Time: <span className="text-slate-400">{new Date(highlightedIncident.timestamp).toLocaleTimeString()}</span></div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (highlightedIncident.event_type === "accident") {
                            navigate("/accidents");
                          } else {
                            navigate(`/incidents?search=${highlightedIncident.id}`);
                          }
                        }}
                        className="w-full mt-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-100 font-mono font-bold text-[11px] uppercase tracking-wider flex items-center justify-center gap-1 cursor-pointer border border-slate-700"
                      >
                        View Incident Details <ExternalLink className="w-3 h-3 text-solar-400" />
                      </button>
                    </div>
                  );
                }
                
                return (
                  <>
                    <div className="text-xs text-slate-400">Job ID: {queryHighlight}</div>
                    <div className="mt-2 pt-2 border-t border-slate-700 text-[10px] text-slate-500 flex justify-between">
                      <span>{parseFloat(queryLat).toFixed(4)}, {parseFloat(queryLng).toFixed(4)}</span>
                      <span>Edge AI Test</span>
                    </div>
                  </>
                );
              })()}
            </div>
          </Popup>
        </Marker>
      )}
    </MapContainer>

    {/* Map Legend Overlay in Corner */}
    <div className="absolute bottom-5 right-5 bg-helios-900/90 border border-slate-700 backdrop-blur-md p-3.5 rounded-xl shadow-2xl z-[1000] text-xs font-mono max-w-xs pointer-events-auto">
      <div className="font-bold text-slate-200 uppercase tracking-wider mb-2 text-[11px] flex items-center gap-1.5">
        <Layers className="w-3.5 h-3.5 text-solar-400" />
        GIS Marker Legend
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-emerald-500 shadow-sm" />
          <span className="text-slate-300">Active Bus</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-red-500 shadow-sm animate-pulse" />
          <span className="text-slate-300">Accident</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-amber-500 shadow-sm" />
          <span className="text-slate-300">Pothole</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-cyan-500 shadow-sm" />
          <span className="text-slate-300">Waterlogging</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-purple-500 shadow-sm" />
          <span className="text-slate-300">Traffic Jam</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-yellow-500 shadow-sm" />
          <span className="text-slate-300">Road Sign Issue</span>
        </div>
      </div>
    </div>
  </div>
</div>

);
};
export default LiveMap;