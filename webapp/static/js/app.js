const esp32IpInput = document.getElementById("esp32Ip");
const deviceSelect = document.getElementById("deviceSelect");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const videoFeed = document.getElementById("videoFeed");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const signalBadge = document.getElementById("signalBadge");
const lastAnnouncement = document.getElementById("lastAnnouncement");
const streamSource = document.getElementById("streamSource");
const connectionDot = document.getElementById("connectionDot");

const IP_STORAGE_KEY = "esp32cam_ip";
const savedIp = localStorage.getItem(IP_STORAGE_KEY);
if (savedIp) {
  esp32IpInput.value = savedIp;
}

let statusTimer = null;

const BADGE_STYLES = {
  "Red Light": ["RED - Cross", "px-4 py-2 rounded-full font-semibold bg-red-500/20 text-red-400 border border-red-500"],
  "Green Light": ["GREEN - Wait", "px-4 py-2 rounded-full font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500"],
};
const BADGE_DEFAULT = ["No signal detected", "px-4 py-2 rounded-full font-semibold bg-slate-800 text-slate-400"];

async function loadAudioDevices() {
  try {
    const res = await fetch("/api/audio_devices");
    const devices = await res.json();
    deviceSelect.innerHTML = "";
    if (devices.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No output devices found (system default)";
      opt.value = "";
      deviceSelect.appendChild(opt);
    } else {
      devices.forEach((d) => {
        const opt = document.createElement("option");
        opt.value = d.index;
        opt.textContent = d.name;
        deviceSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Failed to load audio devices", err);
    deviceSelect.innerHTML = '<option value="">Could not load devices</option>';
  }
}

async function startSystem() {
  startBtn.disabled = true;
  const deviceIndex = deviceSelect.value === "" ? null : parseInt(deviceSelect.value, 10);
  const esp32Ip = esp32IpInput.value.trim();

  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_index: deviceIndex, esp32_ip: esp32Ip }),
  });
  const result = await res.json();

  if (!result.ok) {
    alert(result.error || "Failed to start");
    startBtn.disabled = false;
    return;
  }

  if (esp32Ip) {
    localStorage.setItem(IP_STORAGE_KEY, esp32Ip);
  }
  streamSource.textContent = result.stream_url || "-";

  videoFeed.src = "/video_feed?t=" + Date.now();
  videoFeed.classList.remove("hidden");
  videoPlaceholder.classList.add("hidden");
  stopBtn.disabled = false;
  deviceSelect.disabled = true;
  esp32IpInput.disabled = true;

  statusTimer = setInterval(pollStatus, 1000);
}

async function stopSystem() {
  stopBtn.disabled = true;
  await fetch("/api/stop", { method: "POST" });

  videoFeed.src = "";
  videoFeed.classList.add("hidden");
  videoPlaceholder.classList.remove("hidden");
  startBtn.disabled = false;
  deviceSelect.disabled = false;
  esp32IpInput.disabled = false;
  connectionDot.className = "h-2.5 w-2.5 rounded-full bg-slate-600";

  clearInterval(statusTimer);
  setBadge(null);
  lastAnnouncement.textContent = "-";
  streamSource.textContent = "-";
}

function setBadge(currentClass) {
  const [text, className] = BADGE_STYLES[currentClass] || BADGE_DEFAULT;
  signalBadge.textContent = text;
  signalBadge.className = className;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    connectionDot.className = "h-2.5 w-2.5 rounded-full " + (s.connected ? "bg-emerald-400" : "bg-red-500");
    setBadge(s.current_class);
    lastAnnouncement.textContent = s.last_spoken_text || "-";
    streamSource.textContent = s.stream_url || "-";
  } catch (err) {
    console.error("status poll failed", err);
  }
}

startBtn.addEventListener("click", startSystem);
stopBtn.addEventListener("click", stopSystem);
loadAudioDevices();
