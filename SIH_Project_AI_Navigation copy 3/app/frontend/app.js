/**
 * Antarctic AI Navigation — Full Feature app.js
 * Esri Ocean Basemap · MC Dropout Ensemble · Risk Heatmap · Wind Arrows
 * Shipping Lanes · AIS Vessels · Drift Playback · Multi-Stop · GPX Export
 * India Mission Planner · 7-Day Risk Timeline · Carbon Comparator
 */

// ─── Global State ─────────────────────────────────────────────────────────
let map = null;
let icebergsCatalog = [];
let stationsCatalog = {};
let bergSizes = {};
let selectedIcebergId = null;
let currentForecastHorizon = 48;
let lastRouteWaypoints = [];
let lastIndiaWaypoints = [];
let playbackTimer = null;
let playbackTrackPoints = [];
let riskTimelineChart = null;

// Map layers
let icebergsLayer, trackLayer, predictionLayer, riskLayer, routeLayer,
    stationsLayer, heatmapLayer, windLayer, shippingLanesLayer, vesselsLayer;
let isHeatmapVisible = true, isWindVisible = true, isLanesVisible = true,
    isVesselsVisible = true, isStationsVisible = true;

// FLAG_EMOJI mapping
const FLAG_EMOJI = {IN:"🇮🇳",US:"🇺🇸",GB:"🇬🇧",AU:"🇦🇺",DE:"🇩🇪",DK:"🇩🇰",RU:"🇷🇺"};

// ─── Shipping Lane Polygons ───────────────────────────────────────────────
const SHIPPING_LANES = [
  {name:"Drake Passage",  color:"#0ea5e9",
   coords:[[-55,-70],[-55,-50],[-62,-50],[-62,-70],[-55,-70]]},
  {name:"Cape of Good Hope Route", color:"#0284c7",
   coords:[[-38,12],[-38,26],[-52,26],[-52,12],[-38,12]]},
  {name:"Kerguelen Route", color:"#0c3460",
   coords:[[-42,65],[-42,80],[-52,80],[-52,65],[-42,65]]},
  {name:"Tasmania Route", color:"#1e3a5f",
   coords:[[-42,140],[-42,156],[-52,156],[-52,140],[-42,140]]},
];

// ─── Init ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  lucide.createIcons();
  initMap();
  await Promise.all([loadStations(), loadBergSizes()]);
  await loadIcebergs();
  loadVessels();
  loadShippingLanes();
  loadWindArrows();
  loadRiskHeatmap();
  populateTimelineBergSelect();
  refreshAlertFeed();
});

// ─── Map Init ─────────────────────────────────────────────────────────────
function initMap() {
  map = L.map("polar-map", {center:[-60,0], zoom:3, minZoom:2, maxZoom:12, zoomControl:false});
  L.control.zoom({position:"bottomleft"}).addTo(map);

  // Esri World Ocean Base (free, no API key)
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    {attribution:"Tiles &copy; Esri &mdash; GEBCO, NOAA | BYU/NIC Iceberg Database", maxZoom:13}
  ).addTo(map);
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
    {maxZoom:13, opacity:0.55}
  ).addTo(map);

  riskLayer        = L.layerGroup().addTo(map);
  trackLayer       = L.layerGroup().addTo(map);
  predictionLayer  = L.layerGroup().addTo(map);
  routeLayer       = L.layerGroup().addTo(map);
  stationsLayer    = L.layerGroup().addTo(map);
  icebergsLayer    = L.layerGroup().addTo(map);
  windLayer        = L.layerGroup().addTo(map);
  shippingLanesLayer = L.layerGroup().addTo(map);
  vesselsLayer     = L.layerGroup().addTo(map);
  heatmapLayer     = L.layerGroup().addTo(map);
}

// ─── Berg Sizes ──────────────────────────────────────────────────────────
async function loadBergSizes() {
  try {
    const res = await fetch("/api/icebergs/sizes");
    const data = await res.json();
    bergSizes = data.sizes_sq_km || {};
  } catch(e) { console.warn("Berg sizes unavailable:", e); }
}

// ─── Stations ─────────────────────────────────────────────────────────────
async function loadStations() {
  try {
    const res = await fetch("/api/stations");
    const data = await res.json();
    stationsCatalog = data.stations || {};
    stationsLayer.clearLayers();
    for (const [name, info] of Object.entries(stationsCatalog)) {
      const isIndia = info.country === "India";
      const isPort  = info.type === "port";
      const col  = isIndia ? "#0a1628" : (isPort ? "#1e3a5f" : "#0c3460");
      const sz   = isIndia ? 14 : 9;
      const icon = L.divIcon({className:"custom-station-pin",
        html:`<div style="width:${sz}px;height:${sz}px;border-radius:${isPort?3:50}%;background:${col};border:2px solid #3b82f6;box-shadow:0 0 ${isIndia?10:5}px ${col}cc"></div>`,
        iconSize:[sz,sz],iconAnchor:[sz/2,sz/2]});
      const m = L.marker([info.lat, info.lon],{icon});
      m.bindPopup(`<b style="color:#0284c7">${name}</b><br><small>${info.type==='station'?'🔬 Research Station':'⚓ Gateway Port'}</small><br><span style="font-family:monospace;font-size:0.8rem">${info.lat.toFixed(3)}°, ${info.lon.toFixed(3)}°</span>`);
      stationsLayer.addLayer(m);
    }
  } catch(e) { console.error("Stations load failed:", e); }
}

// ─── Icebergs ──────────────────────────────────────────────────────────────
async function loadIcebergs() {
  try {
    const res = await fetch("/api/icebergs");
    const data = await res.json();
    icebergsCatalog = data.icebergs || [];
    const sel = document.getElementById("iceberg-select");
    sel.innerHTML = "";
    document.getElementById("stat-bergs").innerText = data.total || 75;
    document.getElementById("sa-bergs").innerText   = data.total || 75;
    icebergsLayer.clearLayers();
    riskLayer.clearLayers();

    icebergsCatalog.forEach(berg => {
      const area = bergSizes[berg.iceberg_id] || 200;
      // Size-scale: radius 3px (small) → 10px (massive)
      const r = Math.max(3, Math.min(10, 3 + Math.log10(area) * 2.0));
      const col = "#0a2540";

      const opt = document.createElement("option");
      opt.value = berg.iceberg_id;
      opt.textContent = `${berg.iceberg_id}  (${berg.observations} obs · ${berg.speed_knots} kts)`;
      sel.appendChild(opt);

      const icon = L.divIcon({className:"berg-map-pin",
        html:`<div style="width:${r*2}px;height:${r*2}px;border-radius:50%;background:${col};border:1.5px solid #3b82f6;box-shadow:0 0 5px #0c346099;"></div>`,
        iconSize:[r*2,r*2],iconAnchor:[r,r]});
      const marker = L.marker([berg.latest_lat, berg.latest_lon],{icon});
      marker.bindTooltip(`<b>${berg.iceberg_id}</b> | ~${area.toLocaleString()} km² | ${berg.speed_knots} kts`,{direction:"top"});
      marker.on("click",()=>{sel.value=berg.iceberg_id; onIcebergSelected(berg.iceberg_id);});
      icebergsLayer.addLayer(marker);

      L.circle([berg.latest_lat,berg.latest_lon],{radius:22000,
        color:"#0c3460",weight:1,opacity:0.35,fillColor:"#0c3460",fillOpacity:0.05})
        .addTo(riskLayer);
    });

    const top = icebergsCatalog.find(b=>b.iceberg_id==="B09B") || icebergsCatalog[0];
    if (top) { sel.value = top.iceberg_id; onIcebergSelected(top.iceberg_id); }
  } catch(e) { console.error("Icebergs load failed:", e); }
}

// ─── Risk Heatmap ─────────────────────────────────────────────────────────
async function loadRiskHeatmap() {
  try {
    const res = await fetch("/api/risk-grid");
    const data = await res.json();
    heatmapLayer.clearLayers();
    const pts = data.heatmap_points.map(p => [p.lat, p.lon, p.intensity]);
    if (pts.length > 0) {
      const heat = L.heatLayer(pts, {radius:35, blur:30, maxZoom:8,
        gradient:{0.2:"#0a2540",0.5:"#0c3460",0.8:"#1e3a5f",1.0:"#0f172a"}});
      heatmapLayer.addLayer(heat);
    }
  } catch(e) { console.warn("Risk heatmap unavailable:", e); }
}

// ─── Wind Arrows (Open-Meteo free API, CORS-enabled, no key) ─────────────
async function loadWindArrows() {
  windLayer.clearLayers();
  const gridPoints = [];
  for (let lat = -55; lat >= -70; lat -= 5) {
    for (let lon = -150; lon <= 150; lon += 30) {
      gridPoints.push({lat,lon});
    }
  }
  // Batch the first 12 to keep it fast
  const sample = gridPoints.slice(0, 12);
  await Promise.allSettled(sample.map(async ({lat,lon}) => {
    try {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=wind_speed_10m,wind_direction_10m`;
      const r = await fetch(url);
      if (!r.ok) return;
      const d = await r.json();
      const spd = d.current?.wind_speed_10m ?? 0;
      const dir = d.current?.wind_direction_10m ?? 0;
      drawWindArrow(lat, lon, spd, dir);
    } catch {}
  }));
}

function drawWindArrow(lat, lon, speedKmh, dirDeg) {
  const arrowHtml = `
    <div style="transform:rotate(${dirDeg}deg);width:28px;height:28px;display:flex;align-items:center;justify-content:center;opacity:0.75">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <line x1="14" y1="24" x2="14" y2="4" stroke="#0c3460" stroke-width="2.5" stroke-linecap="round"/>
        <polyline points="8,11 14,4 20,11" fill="none" stroke="#0c3460" stroke-width="2.5" stroke-linejoin="round"/>
      </svg>
    </div>`;
  const icon = L.divIcon({className:"wind-arrow",html:arrowHtml,iconSize:[28,28],iconAnchor:[14,14]});
  const m = L.marker([lat,lon],{icon,interactive:false});
  m.bindTooltip(`💨 ${speedKmh} km/h @ ${dirDeg}°`,{permanent:false});
  windLayer.addLayer(m);
}

// ─── Shipping Lanes ───────────────────────────────────────────────────────
function loadShippingLanes() {
  shippingLanesLayer.clearLayers();
  SHIPPING_LANES.forEach(lane => {
    L.polygon(lane.coords, {
      color:lane.color, weight:1.5, opacity:0.7,
      fillColor:lane.color, fillOpacity:0.08, dashArray:"4,4"
    }).bindTooltip(`⚓ ${lane.name}`, {permanent:false})
      .addTo(shippingLanesLayer);
  });
}

// ─── AIS Vessels ──────────────────────────────────────────────────────────
async function loadVessels() {
  try {
    const res = await fetch("/api/vessels");
    const data = await res.json();
    vesselsLayer.clearLayers();
    document.getElementById("vessel-list").innerHTML = "";
    document.getElementById("stat-vessels").innerText = data.total;
    data.vessels.forEach(v => {
      const flag = FLAG_EMOJI[v.flag] || "🚢";
      // Ship marker as arrow rotated to heading
      const shipHtml = `<div style="transform:rotate(${v.heading}deg);font-size:18px;line-height:1;">▲</div>`;
      const icon = L.divIcon({className:"vessel-pin",
        html:`<div style="color:#1e3a5f;font-size:16px;filter:drop-shadow(0 1px 3px #0a254080)">${shipHtml}</div>`,
        iconSize:[20,20],iconAnchor:[10,10]});
      const marker = L.marker([v.lat,v.lon],{icon});
      marker.bindPopup(`
        <b>${flag} ${v.name}</b><br>
        <small>Type: ${v.type} | ${v.speed_knots} kts | HDG: ${v.heading}°</small><br>
        <b style="color:#0284c7">→ ${v.destination}</b> (ETA: ${v.eta_days}d)
      `);
      vesselsLayer.addLayer(marker);

      // Sidebar list item
      const el = document.createElement("div");
      el.className = "vessel-item";
      el.innerHTML = `
        <div class="vessel-flag">${flag}</div>
        <div class="vessel-info">
          <div class="vessel-name">${v.name}</div>
          <div class="vessel-detail">${v.speed_knots} kts · HDG ${v.heading}° · ETA ${v.eta_days}d</div>
          <div class="vessel-dest">→ ${v.destination}</div>
        </div>
        <span class="vessel-type-badge ${v.type}">${v.type}</span>`;
      el.onclick = () => { map.flyTo([v.lat, v.lon], 6); marker.openPopup(); };
      document.getElementById("vessel-list").appendChild(el);
    });
  } catch(e) { console.error("Vessels load failed:", e); }
}

// ─── Iceberg Selection ─────────────────────────────────────────────────────
async function onIcebergSelected(icebergId) {
  if (!icebergId) return;
  selectedIcebergId = icebergId;
  const berg = icebergsCatalog.find(b => b.iceberg_id === icebergId);
  if (!berg) return;
  const area = bergSizes[icebergId] || 200;

  document.getElementById("info-berg-id").innerText  = berg.iceberg_id;
  document.getElementById("info-berg-date").innerText = berg.last_observed_date || "";
  document.getElementById("info-coords").innerText   = `${berg.latest_lat.toFixed(3)}°S, ${berg.latest_lon.toFixed(3)}°`;
  document.getElementById("info-size").innerText     = `~${area.toLocaleString()} km²`;
  document.getElementById("info-speed").innerText    = `${berg.speed_knots} kts`;
  document.getElementById("info-heading").innerText  = `${(berg.heading_deg||0).toFixed(1)}°`;
  document.getElementById("prediction-results").classList.add("hidden");
  predictionLayer.clearLayers();
  trackLayer.clearLayers();

  try {
    const res  = await fetch(`/api/icebergs/${icebergId}/track?limit=200`);
    const data = await res.json();
    playbackTrackPoints = data.track || [];
    renderTrack(playbackTrackPoints, berg);
    // Update playback slider
    const sl = document.getElementById("playback-slider");
    sl.max = playbackTrackPoints.length - 1;
    sl.value = playbackTrackPoints.length - 1;
    updatePlaybackLabel(playbackTrackPoints.length - 1);
  } catch(e) { console.error("Track load failed:", e); }
}

function renderTrack(pts, berg, frameIdx = null) {
  trackLayer.clearLayers();
  if (!pts.length) return;
  const slice = frameIdx !== null ? pts.slice(0, frameIdx + 1) : pts;
  if (slice.length < 2) return;
  const lls  = slice.map(p => [p.lat, p.lon]);
  const mid  = Math.floor(lls.length / 2);
  L.polyline(lls.slice(0, mid), {color:"#1e3a5f", weight:1.5, opacity:0.3}).addTo(trackLayer);
  L.polyline(lls.slice(mid),   {color:"#0a2540", weight:3.0, opacity:0.9}).addTo(trackLayer);
  const cur = lls[lls.length-1];
  L.circleMarker(cur, {radius:8, fillColor:"#0a2540", color:"#1e3a5f", weight:2, fillOpacity:1})
    .bindPopup(`<b>${berg?.iceberg_id || ""}</b><br>${cur[0].toFixed(3)}°S, ${cur[1].toFixed(3)}°`)
    .addTo(trackLayer);
  if (frameIdx === null) map.flyTo([berg.latest_lat, berg.latest_lon], 5, {duration:1});
}

// ─── Drift Playback ────────────────────────────────────────────────────────
function scrubPlayback(val) {
  const idx = parseInt(val);
  updatePlaybackLabel(idx);
  const berg = icebergsCatalog.find(b => b.iceberg_id === selectedIcebergId);
  renderTrack(playbackTrackPoints, berg, idx);
}

function updatePlaybackLabel(idx) {
  const pt = playbackTrackPoints[idx];
  document.getElementById("playback-date-lbl").innerText = pt ? pt.date : "---";
}

function togglePlayback() {
  const btn = document.getElementById("btn-play");
  if (playbackTimer) { stopPlayback(); return; }
  const sl  = document.getElementById("playback-slider");
  let idx   = parseInt(sl.value);
  if (idx >= playbackTrackPoints.length - 1) idx = 0;
  btn.innerHTML = `<i data-lucide="pause"></i> Pause`;
  lucide.createIcons();
  playbackTimer = setInterval(() => {
    idx++;
    if (idx >= playbackTrackPoints.length) { stopPlayback(); return; }
    sl.value = idx;
    scrubPlayback(idx);
  }, 120);
}

function stopPlayback() {
  if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
  const btn = document.getElementById("btn-play");
  btn.innerHTML = `<i data-lucide="play"></i> Play`;
  lucide.createIcons();
}

// ─── Horizon Toggle ────────────────────────────────────────────────────────
function setHorizon(hours) {
  currentForecastHorizon = hours;
  document.getElementById("btn-hz-24").classList.toggle("active", hours === 24);
  document.getElementById("btn-hz-48").classList.toggle("active", hours === 48);
}

// ─── Trajectory Prediction (MC Dropout Ensemble) ──────────────────────────
async function runPrediction() {
  if (!selectedIcebergId) return;
  const btn = document.getElementById("btn-run-predict");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i> Running 15 MC passes…`;
  lucide.createIcons();

  try {
    const res  = await fetch("/api/predict", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({iceberg_id: selectedIcebergId, horizon_hours: currentForecastHorizon})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction error");

    predictionLayer.clearLayers();
    const curr = data.current_position;
    const hKey = `${currentForecastHorizon}h`;
    const fc   = data.forecasts[hKey];
    const gru  = fc.gru_prediction;
    const base = fc.baseline_prediction;

    document.getElementById("prediction-results").classList.remove("hidden");
    document.getElementById("pred-gru-coord").innerText  = `${gru.lat.toFixed(3)}°S, ${gru.lon.toFixed(3)}°`;
    document.getElementById("pred-gru-drift").innerText  = `Drift: ${fc.expected_drift_km} km`;
    document.getElementById("pred-gru-conf").innerText   = `±${gru.uncertainty_radius_km} km`;
    document.getElementById("pred-base-coord").innerText = `${base.lat.toFixed(3)}°S, ${base.lon.toFixed(3)}°`;

    // Ensemble fan lines (MC Dropout individual passes)
    if (gru.ensemble_fan && gru.ensemble_fan.length > 0) {
      gru.ensemble_fan.forEach(pt => {
        L.polyline([[curr.lat, curr.lon],[pt.lat, pt.lon]],
          {color:"#0c3460", weight:1, opacity:0.25}).addTo(predictionLayer);
        L.circleMarker([pt.lat, pt.lon],
          {radius:3, fillColor:"#1e3a5f", color:"none", fillOpacity:0.35})
          .addTo(predictionLayer);
      });
    }

    // Mean GRU prediction (bold)
    L.polyline([[curr.lat, curr.lon],[gru.lat, gru.lon]],
      {color:"#0a2540", weight:3.5, dashArray:"5,6"}).addTo(predictionLayer);
    L.circleMarker([gru.lat, gru.lon],
      {radius:9, fillColor:"#0a2540", color:"#1e3a5f", weight:2, fillOpacity:1})
      .bindPopup(`<b>GRU Mean +${currentForecastHorizon}h</b><br>${gru.lat.toFixed(3)}°S, ${gru.lon.toFixed(3)}°<br>Drift: ${fc.expected_drift_km} km<br>Uncertainty: ±${gru.uncertainty_radius_km} km`)
      .addTo(predictionLayer);
    // Uncertainty cone
    L.circle([gru.lat, gru.lon],
      {radius:gru.uncertainty_radius_km*1000, color:"#0c3460", weight:1.5,
       dashArray:"4,6", fillColor:"#0c3460", fillOpacity:0.09})
      .addTo(predictionLayer);

    // Baseline
    L.polyline([[curr.lat, curr.lon],[base.lat, base.lon]],
      {color:"#1e3a5f", weight:2, dashArray:"3,5"}).addTo(predictionLayer);
    L.circleMarker([base.lat, base.lon],
      {radius:6, fillColor:"#1e3a5f", color:"#0a2540", weight:1.5, fillOpacity:1})
      .bindPopup(`<b>CV Baseline +${currentForecastHorizon}h</b><br>${base.lat.toFixed(3)}°S, ${base.lon.toFixed(3)}°`)
      .addTo(predictionLayer);

    map.fitBounds(L.latLngBounds([[curr.lat,curr.lon],[gru.lat,gru.lon],[base.lat,base.lon]]),
      {padding:[50,50], maxZoom:7});
  } catch(e) {
    console.error("Prediction error:", e);
    alert("Prediction failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="sparkles"></i> Predict (GRU Ensemble vs Baseline)`;
    lucide.createIcons();
  }
}

// ─── A→B Route Optimization ───────────────────────────────────────────────
async function optimizeMaritimeRoute() {
  const startName = document.getElementById("route-start-select").value;
  const goalName  = document.getElementById("route-goal-select").value;
  const startInfo = stationsCatalog[startName];
  const goalInfo  = stationsCatalog[goalName];
  if (!startInfo || !goalInfo) { alert("Select valid ports."); return; }
  const riskW  = parseFloat(document.getElementById("risk-weight-slider").value);
  const incBergs = document.getElementById("check-iceberg-hazards").checked;
  const btn = document.getElementById("btn-optimize-route") || document.querySelector("[onclick='optimizeMaritimeRoute()']");
  if (btn) { btn.disabled = true; btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i> Solving…`; lucide.createIcons(); }

  try {
    const res  = await fetch("/api/route/optimize", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({start_lat:startInfo.lat, start_lon:startInfo.lon,
        goal_lat:goalInfo.lat, goal_lon:goalInfo.lon,
        start_name:startName, goal_name:goalName,
        risk_tolerance:riskW, vessel_speed_knots:14.0, include_all_icebergs:incBergs})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Route failed");
    lastRouteWaypoints = data.waypoints;
    showRouteTelemetry(data, startName, goalName);
    drawRoute(data.waypoints, routeLayer);
    map.fitBounds(L.latLngBounds(data.waypoints.map(w=>[w.lat,w.lon])),{padding:[60,60]});
  } catch(e) { alert("Route error: " + e.message); }
  finally { if(btn){btn.disabled=false; btn.innerHTML=`<i data-lucide="navigation-2"></i> Compute A* Route`; lucide.createIcons();} }
}

function drawRoute(waypoints, layer) {
  layer.clearLayers();
  const lls = waypoints.map(w => [w.lat, w.lon]);
  lls.forEach((ll, i) => {
    if (i > 0) {
      const risk = waypoints[i].risk;
      const col  = risk < 0.1 ? "#0a2540" : risk < 0.4 ? "#0c3460" : "#7f1d1d";
      L.polyline([lls[i-1], ll], {color:col, weight:5, opacity:0.92, lineJoin:"round"}).addTo(layer);
    }
  });
  L.circleMarker(lls[0], {radius:9, fillColor:"#0a2540", color:"#1e3a5f", weight:2, fillOpacity:1}).addTo(layer);
  L.circleMarker(lls[lls.length-1], {radius:9, fillColor:"#0a2540", color:"#1e3a5f", weight:2, fillOpacity:1}).addTo(layer);
}

function showRouteTelemetry(data, startName, goalName) {
  document.getElementById("route-results-card").classList.remove("hidden");
  document.getElementById("replan-alert-box").classList.add("hidden");
  document.getElementById("res-distance").innerText = `${data.total_distance_km} km (${data.total_distance_nm} NM)`;
  document.getElementById("res-transit").innerText  = `${data.estimated_time_hours} hrs @ 14 kts`;
  document.getElementById("res-waypoints").innerText = `${data.waypoints.length}`;
  document.getElementById("res-safety").innerText    = data.safety_rating;
}

// ─── Multi-Stop Routing ────────────────────────────────────────────────────
async function runMultiStopRoute() {
  const selects = document.querySelectorAll(".multistop-sel");
  const rawStops = [...selects].map(s => s.value).filter(v => v);
  if (rawStops.length < 2) { alert("Need at least 2 stops."); return; }
  const stops = rawStops.map(name => {
    const info = stationsCatalog[name];
    return info ? {name, lat:info.lat, lon:info.lon} : null;
  }).filter(Boolean);
  if (stops.length < 2) { alert("Could not resolve station coordinates."); return; }

  try {
    const res = await fetch("/api/route/multistop", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({stops, risk_tolerance:5.0, vessel_speed_knots:14.0, include_all_icebergs:true})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    lastRouteWaypoints = data.waypoints;
    document.getElementById("route-results-card").classList.remove("hidden");
    document.getElementById("res-distance").innerText = `${data.total_distance_km} km (${data.total_distance_nm} NM)`;
    document.getElementById("res-transit").innerText  = `${data.estimated_time_hours} hrs`;
    document.getElementById("res-waypoints").innerText = `${data.waypoints.length}`;
    document.getElementById("res-safety").innerText    = "Multi-Stop";
    drawRoute(data.waypoints, routeLayer);
    map.fitBounds(L.latLngBounds(data.waypoints.map(w=>[w.lat,w.lon])),{padding:[60,60]});
  } catch(e) { alert("Multi-stop route error: " + e.message); }
}

// ─── Dynamic Replan ────────────────────────────────────────────────────────
async function simulateDynamicReplan() {
  const startName = document.getElementById("route-start-select").value;
  const goalName  = document.getElementById("route-goal-select").value;
  const startInfo = stationsCatalog[startName];
  const goalInfo  = stationsCatalog[goalName];
  if (!startInfo || !goalInfo) return;
  const midLat = (startInfo.lat + goalInfo.lat) / 2;
  const midLon = (startInfo.lon + goalInfo.lon) / 2;
  try {
    const res = await fetch("/api/route/replan", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({current_lat:midLat, current_lon:midLon,
        destination_lat:goalInfo.lat, destination_lon:goalInfo.lon,
        encroaching_iceberg_id:"ALERT_BERG", drift_offset_km:30.0})
    });
    const data = await res.json();
    L.circle([midLat-0.4, midLon+0.5], {radius:35000, color:"#7f1d1d", weight:2,
      fillColor:"#7f1d1d", fillOpacity:0.22}).bindPopup("<b>⚠️ Encroaching Iceberg</b>").addTo(routeLayer);
    if (data.replanning_needed && data.new_route?.waypoints) {
      document.getElementById("replan-alert-box").classList.remove("hidden");
      document.getElementById("replan-alert-msg").innerText = "Danger! Berg within 25 km — AI computed collision-free detour.";
      L.polyline(data.new_route.waypoints.map(w=>[w.lat,w.lon]),
        {color:"#0a2540", weight:4, dashArray:"8,6"}).addTo(routeLayer);
    }
    map.flyTo([midLat,midLon],6);
  } catch(e) { console.error("Replan error:",e); }
}

// ─── GPX Export ───────────────────────────────────────────────────────────
function exportGPX(mode) {
  const wps = mode === "india" ? lastIndiaWaypoints : lastRouteWaypoints;
  if (!wps || !wps.length) { alert("Run a route first, then export."); return; }
  const now = new Date().toISOString();
  let gpx = `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Antarctic AI Navigator - SIH" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata><name>Antarctic Voyage Plan</name><time>${now}</time></metadata>
  <trk><name>Optimal A* Maritime Route</name><trkseg>
`;
  wps.forEach(w => {
    gpx += `    <trkpt lat="${w.lat}" lon="${w.lon}"><time>${now}</time></trkpt>\n`;
  });
  gpx += `  </trkseg></trk></gpx>`;
  const blob = new Blob([gpx], {type:"application/gpx+xml"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `antarctic_voyage_${Date.now()}.gpx`;
  a.click();
}

// ─── Fuel Comparison ──────────────────────────────────────────────────────
async function runFuelComparison() {
  const startName = document.getElementById("fuel-start-select").value;
  const goalName  = document.getElementById("fuel-goal-select").value;
  const startInfo = stationsCatalog[startName];
  const goalInfo  = stationsCatalog[goalName];
  if (!startInfo || !goalInfo) { alert("Select valid stations."); return; }
  const btn = document.getElementById("btn-compare-fuel");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i> Computing…`;
  lucide.createIcons();

  try {
    const res = await fetch("/api/route/compare", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({start_lat:startInfo.lat,start_lon:startInfo.lon,
        goal_lat:goalInfo.lat,goal_lon:goalInfo.lon,
        start_name:startName,goal_name:goalName,include_all_icebergs:true})
    });
    const data = await res.json();
    const panel = document.getElementById("fuel-panel-cards");
    panel.innerHTML = "";
    document.getElementById("fuel-compare-panel").classList.remove("hidden");
    const modes = ["basic","balanced","advanced"];
    const maxFuel = Math.max(...modes.map(m => data.modes[m]?.fuel_consumption_tonnes || 0));

    // Straight-line CO2 comparator (using Haversine formula client-side approx)
    const dx = Math.abs(goalInfo.lat - startInfo.lat);
    const dy = Math.abs(goalInfo.lon - startInfo.lon);
    const straightDistKm = Math.sqrt(dx*dx + dy*dy) * 111.0;
    const straightNm = straightDistKm / 1.852;
    const naiveFuel = round1(straightNm * 1.65);
    const naiveCO2  = round1(naiveFuel * 3.1);
    const aiMode = data.modes["balanced"];

    modes.forEach((modeKey, idx) => {
      const m = data.modes[modeKey];
      if (!m || m.status === "failed") return;
      const fuelPct = Math.round((m.fuel_consumption_tonnes / maxFuel) * 100);
      const riskLabel = m.average_risk_score < 0.05 ? "✅ Safe" : m.average_risk_score < 0.2 ? "⚠️ Caution" : "🔴 High";
      const valClass  = idx===0?"green":idx===1?"blue":"amber";

      panel.innerHTML += `<div class="fuel-mode-card">
        <div class="fuel-mode-header"><div class="fuel-mode-dot" style="background:${m.color}"></div>
        <div><div class="fuel-mode-title">${modeKey.charAt(0).toUpperCase()+modeKey.slice(1)}</div>
        <div class="fuel-mode-sub">${m.speed_knots} kts</div></div></div>
        <div class="fuel-divider"></div>
        <div class="fuel-stat"><span class="fuel-stat-lbl">Distance</span><span class="fuel-stat-val blue">${m.total_distance_km} km</span></div>
        <div class="fuel-stat"><span class="fuel-stat-lbl">Transit</span><span class="fuel-stat-val">${m.estimated_time_hours} hrs</span></div>
        <div class="fuel-stat"><span class="fuel-stat-lbl">Fuel (HFO)</span><span class="fuel-stat-val ${valClass}">${m.fuel_consumption_tonnes} t</span></div>
        <div class="fuel-bar-wrap"><div class="fuel-bar" style="width:${fuelPct}%;background:${m.color}"></div></div>
        <div class="fuel-bar-label">${fuelPct}% of max fuel</div>
        <div class="fuel-divider"></div>
        <div class="fuel-stat"><span class="fuel-stat-lbl">Est. Cost</span><span class="fuel-stat-val">$${m.fuel_cost_usd.toLocaleString()}</span></div>
        <div class="fuel-stat"><span class="fuel-stat-lbl">CO₂</span><span class="fuel-stat-val">${m.co2_emissions_tonnes} t</span></div>
        <div class="fuel-stat"><span class="fuel-stat-lbl">Safety</span><span class="fuel-stat-val green">${riskLabel}</span></div>
      </div>`;
    });

    // CO2 Savings Banner
    if (aiMode?.co2_emissions_tonnes) {
      const savings = round1(naiveCO2 - aiMode.co2_emissions_tonnes);
      const savePct = Math.round((savings / naiveCO2) * 100);
      const banner = document.getElementById("co2-saving-banner");
      banner.classList.remove("hidden");
      banner.textContent = `🌱 AI route saves ${savings} t CO₂ vs straight-line naive routing (−${savePct}%)`;
    }

    // Draw all 3 routes on map
    routeLayer.clearLayers();
    modes.forEach(mk => {
      const m = data.modes[mk];
      if (m?.status==="success" && m.waypoints) {
        L.polyline(m.waypoints.map(w=>[w.lat,w.lon]),
          {color:m.color, weight:4, opacity:0.85,
           dashArray:mk==="basic"?"8,6":mk==="advanced"?"2,4":null}).addTo(routeLayer);
      }
    });
    map.flyTo([(startInfo.lat+goalInfo.lat)/2,(startInfo.lon+goalInfo.lon)/2],4);
  } catch(e) { console.error("Fuel comparison error:",e); alert("Comparison failed: "+e.message); }
  finally { btn.disabled=false; btn.innerHTML=`<i data-lucide="zap"></i> Compare All 3 Fuel Modes`; lucide.createIcons(); }
}

function round1(v) { return Math.round(v * 10) / 10; }

// ─── India Mission Planner ────────────────────────────────────────────────
function flyToStation(name) {
  const info = stationsCatalog[name];
  if (info) map.flyTo([info.lat, info.lon], 7, {duration:1.5});
}

async function runIndiaMissionPlan() {
  const port    = document.getElementById("india-port-select").value;
  const station = document.getElementById("india-station-select").value;
  const viaBoth = document.getElementById("india-via-both").checked;
  const portInfo = stationsCatalog[port];
  const s1Info   = stationsCatalog[station];
  if (!portInfo || !s1Info) { alert("Station not found."); return; }

  const stops = [{name:port,lat:portInfo.lat,lon:portInfo.lon}];
  // If via both, add the other station too
  if (viaBoth) {
    const other = station === "Maitri (India)" ? "Bharati (India)" : "Maitri (India)";
    const otherInfo = stationsCatalog[other];
    stops.push({name:station,lat:s1Info.lat,lon:s1Info.lon});
    if (otherInfo) stops.push({name:other,lat:otherInfo.lat,lon:otherInfo.lon});
    stops.push({name:port,lat:portInfo.lat,lon:portInfo.lon});
  } else {
    stops.push({name:station,lat:s1Info.lat,lon:s1Info.lon});
  }

  const btn = document.getElementById("btn-india-plan");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i> Planning Mission…`;
  lucide.createIcons();

  try {
    const res = await fetch("/api/route/multistop", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({stops, risk_tolerance:5.0, vessel_speed_knots:13.0, include_all_icebergs:true})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    lastIndiaWaypoints = data.waypoints;

    const fuelT = round1(data.total_distance_nm / 1.852 * 1.65);
    const co2   = round1(fuelT * 3.1);
    document.getElementById("india-results-card").classList.remove("hidden");
    document.getElementById("india-res-dist").innerText = `${data.total_distance_km} km`;
    document.getElementById("india-res-time").innerText = `${data.estimated_time_hours} hrs`;
    document.getElementById("india-res-fuel").innerText = `${fuelT} t HFO`;
    document.getElementById("india-res-co2").innerText  = `${co2} t CO₂`;

    // Legs list
    const legsList = document.getElementById("india-legs-list");
    legsList.innerHTML = "";
    data.legs.forEach(leg => {
      const el = document.createElement("div");
      el.className = "leg-item";
      el.innerHTML = `<div class="leg-from-to">${leg.from} → ${leg.to}</div>
        <div class="leg-detail">${leg.distance_km} km / ${leg.distance_nm} NM</div>`;
      legsList.appendChild(el);
    });

    // Carbon comparator
    const straightKm = data.legs.reduce((s,l) => {
      const f = stationsCatalog[l.from], t = stationsCatalog[l.to];
      if (f && t) {
        const d = Math.sqrt(Math.pow((t.lat-f.lat)*111,2)+Math.pow((t.lon-f.lon)*111*Math.cos(f.lat*Math.PI/180),2));
        return s + d;
      }
      return s + 500;
    }, 0);
    const naiveNm = straightKm / 1.852;
    const naiveFuelT = round1(naiveNm * 1.65);
    const naiveCO2t  = round1(naiveFuelT * 3.1);
    const saving     = round1(naiveCO2t - co2);
    const savePct    = Math.round((saving/naiveCO2t)*100);

    document.getElementById("carbon-ai").innerText    = `${co2} t CO₂`;
    document.getElementById("carbon-naive").innerText = `${naiveCO2t} t CO₂`;
    document.getElementById("carbon-saving-pct").classList.remove("hidden");
    document.getElementById("carbon-saving-pct").innerText =
      `🌱 AI saves ${saving} t CO₂ vs straight-line routing (−${savePct}%) per mission`;

    drawRoute(data.waypoints, routeLayer);
    map.fitBounds(L.latLngBounds(data.waypoints.map(w=>[w.lat,w.lon])),{padding:[60,60]});
  } catch(e) { alert("Mission plan failed: "+e.message); }
  finally { btn.disabled=false; btn.innerHTML=`<i data-lucide="flag"></i> Plan Indian Supply Mission`; lucide.createIcons(); }
}

// ─── Analytics: Alert Feed ────────────────────────────────────────────────
async function refreshAlertFeed() {
  const feed = document.getElementById("alert-feed");
  feed.innerHTML = `<div class="alert-feed-item loading">Scanning ${icebergsCatalog.length} icebergs across 4 shipping lanes…</div>`;

  const LANES = [
    {name:"Drake Passage",clat:-58.5,clon:-60.0},
    {name:"Cape of Good Hope",clat:-45.0,clon:18.5},
    {name:"Kerguelen Route",clat:-47.0,clon:72.0},
    {name:"Tasmania Route",clat:-47.0,clon:147.0},
  ];

  const alerts = [];
  icebergsCatalog.forEach(berg => {
    LANES.forEach(lane => {
      const dLat = (berg.latest_lat - lane.clat) * 111;
      const dLon = (berg.latest_lon - lane.clon) * 111 * Math.cos(berg.latest_lat * Math.PI/180);
      const dist = Math.sqrt(dLat*dLat + dLon*dLon);
      if (dist < 500) {
        alerts.push({berg:berg.iceberg_id, lane:lane.name, dist:Math.round(dist),
          lat:berg.latest_lat, lon:berg.latest_lon,
          level: dist < 100 ? "danger" : "warn"});
      }
    });
  });

  alerts.sort((a,b)=>a.dist-b.dist);
  const topAlerts = alerts.slice(0, 15);
  document.getElementById("sa-alerts").innerText = topAlerts.filter(a=>a.level==="danger").length;
  document.getElementById("sa-max-threat").innerText = topAlerts[0]?.berg || "--";

  feed.innerHTML = "";
  if (!topAlerts.length) {
    feed.innerHTML = `<div class="alert-feed-item">✅ No active collision threats detected.</div>`;
    return;
  }
  topAlerts.forEach(a => {
    const el = document.createElement("div");
    el.className = `alert-feed-item ${a.level}`;
    el.innerHTML = `${a.level==="danger"?"🔴":"🟡"} <span class="alert-berg">${a.berg}</span> within <span class="alert-dist">${a.dist} km</span> of <span class="alert-lane">${a.lane}</span>`;
    el.onclick = () => map.flyTo([a.lat,a.lon],6);
    feed.appendChild(el);
  });
}

// ─── Analytics: 7-Day Risk Timeline Chart ─────────────────────────────────
function populateTimelineBergSelect() {
  const sel = document.getElementById("timeline-berg-select");
  icebergsCatalog.slice(0, 20).forEach(b => {
    const opt = document.createElement("option");
    opt.value = b.iceberg_id;
    opt.textContent = b.iceberg_id;
    sel.appendChild(opt);
  });
}

async function loadRiskTimeline(bergId) {
  if (!bergId) return;
  try {
    const res  = await fetch(`/api/icebergs/${bergId}/risk-timeline`);
    const data = await res.json();
    const tl   = data.timeline;
    const ctx  = document.getElementById("risk-timeline-chart").getContext("2d");

    if (riskTimelineChart) riskTimelineChart.destroy();

    const laneNames = tl[0]?.lane_risks.map(l=>l.lane) || [];
    const colors = ["#0a2540","#0c3460","#1e3a5f","#2d4a7a"];
    const datasets = laneNames.map((ln, i) => ({
      label: ln,
      data: tl.map(d => d.lane_risks.find(l=>l.lane===ln)?.risk_score ?? 0),
      borderColor: colors[i], backgroundColor: colors[i]+"22",
      borderWidth: 2, fill: true, tension: 0.4, pointRadius: 3
    }));

    riskTimelineChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: tl.map(d => `Day ${d.day}`),
        datasets
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend:{labels:{color:"#334155",font:{size:9}}}},
        scales: {
          x:{ticks:{color:"#64748b",font:{size:9}},grid:{color:"rgba(14,165,233,0.1)"}},
          y:{ticks:{color:"#64748b",font:{size:9}},grid:{color:"rgba(14,165,233,0.1)"},
             min:0,max:1,title:{display:true,text:"Risk Score",color:"#64748b",font:{size:9}}}
        }
      }
    });
  } catch(e) { console.error("Risk timeline error:", e); }
}

// ─── Tab Switching ─────────────────────────────────────────────────────────
function switchTab(tabId) {
  ["tracking","routing","fuel","india","analytics","benchmarks"].forEach(t => {
    document.getElementById(`tab-btn-${t}`)?.classList.remove("active");
    document.getElementById(`tab-${t}`)?.classList.remove("active");
  });
  document.getElementById(`tab-btn-${tabId}`)?.classList.add("active");
  document.getElementById(`tab-${tabId}`)?.classList.add("active");
  if (tabId === "benchmarks") {
    const img = document.getElementById("eval-plot-img");
    if (img) img.src = `/artifacts/trajectory_evaluation.png?t=${Date.now()}`;
  }
  if (tabId === "analytics" && icebergsCatalog.length > 0) {
    refreshAlertFeed();
    const sel = document.getElementById("timeline-berg-select");
    if (sel && sel.options.length <= 1) populateTimelineBergSelect();
  }
}

// ─── Layer Toggles ─────────────────────────────────────────────────────────
function resetPolarView() { map.flyTo([-60,0],3,{duration:1}); }

function toggleHeatmap() {
  isHeatmapVisible = !isHeatmapVisible;
  isHeatmapVisible ? map.addLayer(heatmapLayer) : map.removeLayer(heatmapLayer);
  document.getElementById("btn-toggle-heat").classList.toggle("active", isHeatmapVisible);
}

function toggleWindLayer() {
  isWindVisible = !isWindVisible;
  isWindVisible ? map.addLayer(windLayer) : map.removeLayer(windLayer);
  document.getElementById("btn-toggle-wind").classList.toggle("active", isWindVisible);
}

function toggleShippingLanes() {
  isLanesVisible = !isLanesVisible;
  isLanesVisible ? map.addLayer(shippingLanesLayer) : map.removeLayer(shippingLanesLayer);
  document.getElementById("btn-toggle-lanes").classList.toggle("active", isLanesVisible);
}

function toggleVesselsLayer() {
  isVesselsVisible = !isVesselsVisible;
  isVesselsVisible ? map.addLayer(vesselsLayer) : map.removeLayer(vesselsLayer);
  document.getElementById("btn-toggle-vessels").classList.toggle("active", isVesselsVisible);
}

function toggleStationsLayer() {
  isStationsVisible = !isStationsVisible;
  isStationsVisible ? map.addLayer(stationsLayer) : map.removeLayer(stationsLayer);
  document.getElementById("btn-toggle-stations").classList.toggle("active", isStationsVisible);
}
