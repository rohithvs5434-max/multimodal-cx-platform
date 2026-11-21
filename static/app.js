const state={lastId:"0-0",events:[],selectedFile:null,busy:false};
const $=id=>document.getElementById(id);
const input=$("audio-input"),drop=$("dropzone"),choose=$("choose-btn"),selected=$("selected-file"),fileName=$("file-name"),fileSize=$("file-size"),remove=$("remove-file"),run=$("analyze-btn"),runLabel=$("analyze-label"),progress=$("progress-wrap"),progressLabel=$("progress-label"),error=$("error-box"),empty=$("empty-result"),content=$("result-content"),transcript=$("result-transcript"),emotion=$("result-emotion"),sentiment=$("result-sentiment"),emotionBar=$("emotion-bar"),sentimentBar=$("sentiment-bar"),emotionConf=$("emotion-confidence"),sentimentConf=$("sentiment-confidence"),attention=$("result-alert"),eventsEl=$("events"),template=$("event-template");
const title=s=>String(s||"").replace(/\b\w/g,c=>c.toUpperCase()),pct=v=>`${(Number(v||0)*100).toFixed(1)}%`,size=n=>n<1048576?`${Math.round(n/1024)} KB`:`${(n/1048576).toFixed(2)} MB`;
function showErr(s){error.textContent=s;error.classList.remove("hidden")}function clearErr(){error.classList.add("hidden")}
function selectFile(f){if(!f){state.selectedFile=null;selected.classList.add("hidden");run.disabled=true;return}if(!f.type.startsWith("audio/")&&!/\.(wav|mp3|m4a|ogg|flac)$/i.test(f.name))return showErr("Please select a supported audio file.");if(f.size>25*1048576)return showErr("Audio exceeds the 25 MB upload limit.");clearErr();state.selectedFile=f;fileName.textContent=f.name;fileSize.textContent=size(f.size);selected.classList.remove("hidden");run.disabled=false}
choose.onclick=()=>input.click();drop.onclick=e=>{if(!e.target.closest("button"))input.click()};input.onchange=()=>selectFile(input.files[0]);remove.onclick=()=>{input.value="";selectFile(null);clearErr()};
["dragenter","dragover"].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.add("dragging")}));["dragleave","drop"].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.remove("dragging")}));drop.ondrop=e=>selectFile(e.dataTransfer.files[0]);
function setBusy(v){state.busy=v;run.disabled=v||!state.selectedFile;runLabel.textContent=v?"Running analysis…":"Run multimodal analysis";progress.classList.toggle("hidden",!v);if(v){progress.querySelector("i").style.width="72%";progressLabel.textContent="Whisper + emotion + sentiment analysis in progress…"}}
function showResult(e){state.shown=e;empty.classList.add("hidden");content.classList.remove("hidden");$("result-source").textContent="AUDIO ANALYSIS";transcript.textContent=e.transcript||"No transcript returned.";const em=e.emotion||{},se=e.sentiment||{},fu=e.fusion||{},ec=Number(em.confidence||0),sc=Number(se.confidence||0);emotion.textContent=title(em.label||"Unknown");sentiment.textContent=title(se.label||"Unknown");emotionConf.textContent=pct(ec);sentimentConf.textContent=pct(sc);emotionBar.style.width=`${ec*100}%`;sentimentBar.style.width=`${sc*100}%`;$("result-fusion").textContent=title(fu.state||"Unknown");$("result-fusion-score").textContent=Number(fu.score||0).toFixed(3);showAttention(e);showFollowup(e);if(!live.on)$("result-card").scrollIntoView({behavior:"smooth",block:"nearest"})}
function add(e){if(!e?.event_id||state.events.some(x=>x.event_id===e.event_id))return;state.events.push(e);state.lastId=e.event_id;localStorage.setItem("cx:last-event-id",state.lastId);render()}
function render(){const latest=state.events.at(-1);$("event-count").textContent=state.events.length;$("emotion").textContent=title(latest?.emotion?.label||"—");$("sentiment").textContent=title(latest?.sentiment?.label||"—");$("alert-count").textContent=state.events.filter(x=>x.alert&&!x.handled).length;eventsEl.innerHTML="";if(!state.events.length){eventsEl.innerHTML='<div style="padding:28px;text-align:center;color:#60718e;font-size:9px">No events yet — analyze an audio file to begin.</div>';return}[...state.events].reverse().forEach(e=>{const n=template.content.cloneNode(true),r=n.querySelector(".event");r.classList.toggle("real",e.source==="audio");r.classList.toggle("has-alert",!!e.alert&&!e.handled);r.classList.toggle("resolved",!!e.handled);n.querySelector(".source").textContent=e.source==="audio"?"AUDIO ANALYSIS":"SIMULATED";n.querySelector("time").textContent=new Date(e.timestamp).toLocaleTimeString([], {hour:"numeric",minute:"2-digit",second:"2-digit"});n.querySelector(".transcript").textContent=e.transcript||"—";n.querySelector(".emotion").textContent=`Emotion · ${title(e.emotion?.label)} (${pct(e.emotion?.confidence)})`;n.querySelector(".sentiment").textContent=`Sentiment · ${title(e.sentiment?.label)} (${pct(e.sentiment?.confidence)})`;const f=n.querySelector(".fusion");if(e.fusion)f.textContent=`Fusion · ${title(e.fusion.state)} (${Number(e.fusion.score||0).toFixed(3)})`;else f.remove();const a=n.querySelector(".alert");if(e.handled){a.textContent="Handled";a.dataset.severity="handled"}else if(e.attention){a.textContent=`${title(e.attention.severity)} attention`;a.dataset.severity=e.attention.severity}else if(e.alert)a.textContent="Customer Needs Attention";else a.remove();const cx=extras(e);if(cx)n.querySelector(".tags").parentNode.appendChild(cx);eventsEl.appendChild(n)})}
run.onclick=async()=>{if(!state.selectedFile||state.busy)return;clearErr();setBusy(true);try{const f=new FormData();f.append("audio",state.selectedFile);{const c=$("customer-id").value.trim();if(c)f.append("customer_id",c)}const r=await fetch("/api/analyze",{method:"POST",body:f}),d=await r.json();if(!r.ok)throw Error(d.detail||"Audio analysis failed.");showResult(d);add(d);input.value="";selectFile(null)}catch(e){showErr(e.message||"Unable to analyze audio.")}finally{setBusy(false)}};
$("demo").onclick=async()=>{const b=$("demo");try{b.disabled=true;b.textContent="Loading…";const r=await fetch("/api/demo/seed",{method:"POST"}),d=await r.json();if(!r.ok)throw Error(d.detail||"Unable to load demo events.");(d.events||[]).forEach(add)}catch(e){showErr(e.message)}finally{b.disabled=false;b.textContent="＋ Load demo events"}};
async function catchUp(){try{const r=await fetch("/api/events?last_event_id=0-0");const d=await r.json();state.events=d;render();if(d.length)state.lastId=d.at(-1).event_id}catch{showErr("Event history unavailable.")}}
function connect(){const p=location.protocol==="https:"?"wss":"ws",s=new WebSocket(`${p}://${location.host}/ws/events?last_event_id=${encodeURIComponent(state.lastId)}`);s.onopen=()=>{$("status").textContent="Live";document.body.classList.add("connected")};s.onmessage=m=>{try{const e=JSON.parse(m.data);add(e);if(e.source==="audio")showResult(e)}catch{}};s.onclose=()=>{$("status").textContent="Reconnecting";document.body.classList.remove("connected");setTimeout(connect,1500)};s.onerror=()=>s.close()}
render();catchUp().finally(connect);

/* ---------- Live microphone analysis ---------- */
const live={on:false,stream:null,rec:null,ctx:null,raf:0,pending:0,clips:0};
const SEG_MS=6000,MIN_BYTES=4000,MAX_PENDING=2;
const liveBtn=$("live-btn"),liveLabel=$("live-label"),liveStatus=$("live-status"),liveCount=$("live-count"),liveViz=$("live-viz");
const bars=[...Array(24)].map(()=>{const i=document.createElement("i");liveViz.appendChild(i);return i});
const setLive=t=>{liveStatus.textContent=t};
function startMeter(stream){
  const AC=window.AudioContext||window.webkitAudioContext;if(!AC)return;
  const ctx=new AC(),an=ctx.createAnalyser();an.fftSize=64;ctx.createMediaStreamSource(stream).connect(an);
  const data=new Uint8Array(an.frequencyBinCount);live.ctx=ctx;
  const tick=()=>{an.getByteFrequencyData(data);bars.forEach((b,i)=>{b.style.height=`${Math.max(8,data[i]/255*100)}%`});live.raf=requestAnimationFrame(tick)};tick()}
function stopMeter(){cancelAnimationFrame(live.raf);bars.forEach(b=>b.style.height="8%");if(live.ctx){live.ctx.close();live.ctx=null}}
function recordSegment(){return new Promise(res=>{const rec=new MediaRecorder(live.stream),chunks=[];live.rec=rec;rec.ondataavailable=e=>e.data.size&&chunks.push(e.data);rec.onstop=()=>res(new Blob(chunks,{type:rec.mimeType}));rec.start();setTimeout(()=>rec.state!=="inactive"&&rec.stop(),SEG_MS)})}
async function sendClip(blob){
  if(blob.size<MIN_BYTES)return;
  if(live.pending>=MAX_PENDING){setLive("Analysis is behind — skipped a clip");return}
  live.pending++;setLive("Analysing…");
  try{
    const ext=blob.type.includes("mp4")?"mp4":blob.type.includes("ogg")?"ogg":"webm",f=new FormData();
    f.append("audio",blob,`live.${ext}`);{const c=$("customer-id").value.trim();if(c)f.append("customer_id",c)}
    const r=await fetch("/api/analyze",{method:"POST",body:f}),d=await r.json();
    if(!r.ok)throw Error(d.detail||"Live analysis failed.");
    live.clips++;liveCount.textContent=`${live.clips} clip${live.clips===1?"":"s"}`;
    showResult(d);add(d);
  }catch(e){showErr(e.message||"Live analysis failed.")}
  finally{live.pending--;setLive(live.on?"Listening…":"Idle")}
}
async function startLive(){
  clearErr();
  if(!navigator.mediaDevices?.getUserMedia||typeof MediaRecorder==="undefined")return showErr("Live recording is not supported in this browser.");
  try{live.stream=await navigator.mediaDevices.getUserMedia({audio:true})}
  catch{return showErr("Microphone access was denied. Allow it in the browser address bar and try again.")}
  live.on=true;liveBtn.classList.add("recording");liveLabel.textContent="Stop live analysis";setLive("Listening…");startMeter(live.stream);
  while(live.on){const blob=await recordSegment();sendClip(blob)}
  live.stream.getTracks().forEach(t=>t.stop());stopMeter();
}
function stopLive(){live.on=false;liveBtn.classList.remove("recording");liveLabel.textContent="Start live analysis";if(live.rec&&live.rec.state!=="inactive")live.rec.stop()}
liveBtn.onclick=()=>live.on?stopLive():startLive();
window.addEventListener("beforeunload",()=>{if(live.on)stopLive()});

/* ---------- Case handling: reason, suggested actions, follow-up ---------- */
function followText(f){
  const a=Number(f.previous_score).toFixed(2),b=Number(f.new_score).toFixed(2);
  if(f.outcome==="improved")return `Follow-up: sentiment improved (${a} → ${b}). The earlier issue looks resolved.`;
  if(f.outcome==="worse")return `Follow-up: sentiment got worse (${a} → ${b}). Escalate this case.`;
  return `Follow-up: no clear change in sentiment (${a} → ${b}). Check whether the issue is really fixed.`;
}
function showAttention(e){
  const a=e.attention,show=!!(a||e.alert)&&!e.handled;
  attention.classList.toggle("hidden",!show);if(!show)return;
  attention.dataset.severity=a?a.severity:"medium";
  $("alert-title").textContent=a?a.summary:"Customer needs attention";
  $("alert-detail").textContent=a?"Why: "+a.reason_label+". Suggested next steps:":"Negative sentiment detected in this interaction.";
  const ul=$("alert-actions");ul.innerHTML="";
  (a?a.actions:[]).forEach(t=>{const li=document.createElement("li");li.textContent=t;ul.appendChild(li)});
}
function showFollowup(e){
  const box=$("result-followup");
  box.classList.toggle("hidden",!e.followup);if(!e.followup)return;
  box.className="followup "+e.followup.outcome;box.textContent=followText(e.followup);
}
async function handleAlert(e,btn){
  btn.disabled=true;
  try{
    const r=await fetch(`/api/alerts/${encodeURIComponent(e.event_id)}/handle`,{method:"POST"}),d=await r.json();
    if(!r.ok)throw Error(d.detail||"Could not mark the alert as handled.");
    e.handled=true;if(state.shown===e)showAttention(e);render();
  }catch(x){btn.disabled=false;showErr(x.message)}
}
function extras(e){
  const wrap=document.createElement("div");wrap.className="case";let any=false;
  const a=e.attention;
  if(a){
    any=true;wrap.dataset.severity=e.handled?"handled":a.severity;
    const h=document.createElement("b");h.textContent=e.handled?`Handled: ${a.reason_label}`:a.summary;wrap.appendChild(h);
    if(!e.handled){const ul=document.createElement("ul");
    a.actions.forEach(t=>{const li=document.createElement("li");li.textContent=t;ul.appendChild(li)});
    wrap.appendChild(ul)}
    const row=document.createElement("div");row.className="case-row";
    if(e.handled){
      const s=document.createElement("span");s.className="handled";
      s.textContent=e.customer_id?`Handled · waiting for ${e.customer_id}'s next interaction`:"Handled";row.appendChild(s);
    }else{
      const b=document.createElement("button");b.type="button";b.className="handle-btn";b.textContent="Mark as handled";
      b.onclick=()=>handleAlert(e,b);row.appendChild(b);
      if(!e.customer_id){const n=document.createElement("small");n.textContent="Analyse with a Customer ID to track the follow-up.";row.appendChild(n)}
    }
    wrap.appendChild(row);
  }
  if(e.followup){any=true;const f=document.createElement("div");f.className="followup "+e.followup.outcome;f.textContent=followText(e.followup);wrap.appendChild(f)}
  return any?wrap:null;
}
