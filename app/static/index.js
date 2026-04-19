// === CONSTANTS ===

var ROUTE_STOPS = {
  '9X':   ['Miyapur','Kukatpally','SR Nagar','Ameerpet','Mehdipatnam'],
  '47L':  ['Secunderabad Station','Ameerpet','Jubilee Hills Checkpost','Banjara Hills','Gachibowli'],
  '127K': ['MGBS','Koti','Mehdipatnam','SR Nagar','Hitech City'],
  '218':  ['ECIL','Uppal','LB Nagar','Dilsukhnagar','MGBS'],
  '10H':  ['Secunderabad Station','ECIL','Uppal','LB Nagar'],
  '5K':   ['Madhapur','Hitech City','Gachibowli','Miyapur'],
  '216':  ['Charminar','Koti','MGBS','Mehdipatnam','Ameerpet'],
  '400':  ['Dilsukhnagar','LB Nagar','Uppal','ECIL','Secunderabad Station']
};

var PANEL_META = {
  predict:  { title: 'Predict Crowd',       sub: 'Enter trip details to forecast bus crowd level' },
  forecast: { title: 'Route Overview',      sub: 'See crowd levels for every stop on a route right now' },
  besttime: { title: 'Best Time to Travel', sub: 'Find the least crowded window for your route and stop' },
  mapview:  { title: 'Stop Map',            sub: 'View all TSRTC stops on the map' },
  feedback: { title: 'Give Feedback',       sub: 'Help improve predictions by reporting actual crowd' },
  about:    { title: 'About',               sub: 'Project details, model info, and features' },
  profile:  { title: 'My Profile',          sub: 'Your account info and recent predictions' }
};

// === STATE ===

var map = null;
var selectedFeedback = null;
var predictionHistory = JSON.parse(localStorage.getItem('predHistory') || '[]');
var profileData = null;

// === PANEL NAVIGATION ===

function showPanel(name) {
  document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.nav-item').forEach(function(n) { n.classList.remove('active'); });
  document.getElementById('panel-' + name).classList.add('active');
  var navEl = document.querySelector('[data-panel="' + name + '"]');
  if (navEl) { navEl.classList.add('active'); }
  document.getElementById('panel-title').textContent    = PANEL_META[name].title;
  document.getElementById('panel-subtitle').textContent = PANEL_META[name].sub;
  if (name === 'mapview') { initMap(); }
  if (name === 'profile') {
    loadProfileData();
    document.getElementById('profile-pred-count').textContent = predictionHistory.length;
    renderProfileHistory();
  }
}

// === STOP DROPDOWN FILTER ===

function filterStops(routeSelectId, stopSelectId) {
  var route   = document.getElementById(routeSelectId).value;
  var stopSel = document.getElementById(stopSelectId);
  var stops   = ROUTE_STOPS[route] || [];
  stopSel.innerHTML = stops.map(function(s) {
    return '<option value="' + s + '">' + s + '</option>';
  }).join('');
}

function updateFcStopInfo() {
  var route = document.getElementById('fc-route').value;
  var stops = ROUTE_STOPS[route] || [];
  document.getElementById('fc-stop-info').textContent =
    'Will show crowd at ' + stops.length + ' stops: ' + stops.join(' > ');
}

// === FILL NOW ===

function fillNow() {
  var now  = new Date();
  document.getElementById('f-hour').value = now.getHours();
  var days  = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  var today = days[now.getDay()];
  var sel   = document.getElementById('f-day');
  for (var i = 0; i < sel.options.length; i++) {
    if (sel.options[i].value === today) { sel.options[i].selected = true; break; }
  }
}

// === HELPERS ===

function crowdIcon(l)  { return l === 'High' ? '[HIGH]' : l === 'Medium' ? '[MED]' : '[LOW]'; }
function crowdClass(l) { return l.toLowerCase(); }

// === RENDER RESULT ===

function renderResult(data) {
  var prediction  = data.prediction;
  var confidence  = data.confidence;
  var proba       = data.proba;
  var forecast    = data.forecast;
  var alternates  = data.alternates;
  var event_alert = data.event_alert;

  var cls  = crowdClass(prediction);
  var icon = prediction === 'High' ? '(!!)' : prediction === 'Medium' ? '(~)' : '(ok)';
  var msg  = prediction === 'High'
    ? 'Heavy congestion expected. Consider alternate stops or waiting.'
    : prediction === 'Medium'
    ? 'Moderate crowd. Limited seating -- you can still board comfortably.'
    : 'Smooth travel. Good seat availability expected.';

  var html = '';

  if (event_alert) {
    html += '<div class="event-alert"><i class="fas fa-exclamation-triangle"></i>' + event_alert + '</div>';
  }

  html += '<div class="result-banner ' + cls + '">'
        + '<div class="result-icon">' + icon + '</div>'
        + '<div class="result-text"><h3>' + prediction + ' Crowd</h3><p>' + msg + '</p></div>'
        + '</div>';

  var cert = confidence >= 75 ? 'High certainty' : confidence >= 55 ? 'Moderate certainty' : 'Low -- check again closer to departure';
  html += '<div class="card-box" style="margin-bottom:16px">'
        + '<div class="card-title">Model Confidence</div>'
        + '<div class="conf-label"><span>Confidence: <strong>' + confidence + '%</strong></span>'
        + '<span style="font-size:11px;color:#94a3b8">' + cert + '</span></div>'
        + '<div class="conf-bar"><div class="conf-fill ' + cls + '" style="width:' + confidence + '%"></div></div>'
        + '<div style="margin-top:14px">';

  var probaKeys = Object.keys(proba);
  for (var pi = 0; pi < probaKeys.length; pi++) {
    var lbl = probaKeys[pi];
    var pct = proba[lbl];
    var bc  = lbl === 'High' ? 'var(--red)' : lbl === 'Medium' ? 'var(--yellow)' : 'var(--green)';
    html += '<div class="proba-row"><div class="proba-label">' + lbl + '</div>'
          + '<div class="proba-bar-bg"><div class="proba-bar-fill" style="width:' + pct + '%;background:' + bc + '"></div></div>'
          + '<div class="proba-pct">' + pct + '%</div></div>';
  }
  html += '</div></div>';

  html += '<div class="card-box" style="margin-bottom:16px">'
        + '<div class="card-title"><i class="fas fa-clock me-2"></i>Next 7 Hours</div>'
        + '<div class="forecast-strip">';
  for (var fi = 0; fi < forecast.length; fi++) {
    var fc    = forecast[fi];
    var fcCls = crowdClass(fc.level);
    var hh    = fc.hour < 10 ? '0' + fc.hour : '' + fc.hour;
    html += '<div class="forecast-cell ' + (fi === 0 ? 'current' : '') + '">'
          + '<div class="fc-hour">' + hh + ':00</div>'
          + '<div class="fc-dot ' + fcCls + '">' + crowdIcon(fc.level) + '</div>'
          + '<div class="fc-lbl ' + fcCls + '">' + fc.level + '</div>'
          + '<div class="fc-tag">' + fc.label + '</div></div>';
  }
  html += '</div></div>';

  if (alternates && alternates.length > 0) {
    html += '<div class="card-box"><div class="card-title"><i class="fas fa-route me-2"></i>Alternate Stops on Same Route</div>'
          + '<p style="font-size:13px;color:#64748b;margin-bottom:12px">These stops have lower predicted crowd right now:</p>';
    for (var ai = 0; ai < alternates.length; ai++) {
      var a = alternates[ai];
      html += '<div class="alt-card"><div>'
            + '<div class="alt-stop-name"><i class="fas fa-map-pin me-2" style="color:#94a3b8;font-size:12px"></i>' + a.stop + '</div>'
            + (a.dist ? '<div class="alt-stop-dist">' + a.dist + '</div>' : '')
            + '</div><span class="alt-badge ' + crowdClass(a.level) + '">' + crowdIcon(a.level) + ' ' + a.level + '</span></div>';
    }
    html += '</div>';
  }

  var histEntry = {
    route: document.getElementById('f-route').value,
    stop:  document.getElementById('f-stop').value,
    hour:  document.getElementById('f-hour').value,
    day:   document.getElementById('f-day').value,
    prediction: prediction,
    time:  new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
  };
  predictionHistory.unshift(histEntry);
  predictionHistory = predictionHistory.slice(0, 5);
  localStorage.setItem('predHistory', JSON.stringify(predictionHistory));

  document.getElementById('fb-last-pred').innerHTML =
    'Last prediction: <strong>' + prediction + ' crowd</strong> at <strong>'
    + document.getElementById('f-stop').value + '</strong> -- '
    + document.getElementById('f-hour').value + ':00, ' + document.getElementById('f-day').value;

  document.getElementById('result-area').innerHTML = html;
}

// === RUN PREDICTION ===

function runPredict() {
  var btn = document.getElementById('predict-btn');
  btn.innerHTML = '<span class="spinner"></span> Predicting...';
  btn.disabled  = true;
  var payload = {
    route:   document.getElementById('f-route').value,
    stop:    document.getElementById('f-stop').value,
    hour:    document.getElementById('f-hour').value,
    day:     document.getElementById('f-day').value,
    weather: document.getElementById('f-weather').value,
    buses:   6,
    month:   window.CURRENT_MONTH
  };
  fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    .then(function(res)  { return res.json(); })
    .then(function(data) { renderResult(data); })
    .catch(function()    {
      document.getElementById('result-area').innerHTML = '<div class="card-box" style="color:var(--red)">Error connecting to server.</div>';
    })
    .finally(function() {
      btn.innerHTML = '<i class="fas fa-search me-2"></i>Predict Crowd Level';
      btn.disabled  = false;
    });
}

// === ROUTE OVERVIEW ===

function runRouteOverview() {
  var route        = document.getElementById('fc-route').value;
  var weather      = document.getElementById('fc-weather').value;
  var hour         = parseInt(document.getElementById('fc-hour').value);
  var dayNames     = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  var day          = dayNames[new Date().getDay()];
  var stopsOnRoute = ROUTE_STOPS[route] || [];

  if (stopsOnRoute.length === 0) {
    document.getElementById('fc-result').innerHTML = '<div class="card-box" style="color:var(--red)">No stops found for this route.</div>';
    return;
  }

  document.getElementById('fc-result').innerHTML =
    '<div class="card-box text-center" style="padding:30px">'
    + '<span class="spinner" style="width:28px;height:28px;border-color:rgba(0,0,0,.15);border-top-color:var(--navy)"></span>'
    + '<p style="color:#94a3b8;margin-top:12px">Checking all ' + stopsOnRoute.length + ' stops...</p></div>';

  var results  = [];
  var fetchOrder = stopsOnRoute.slice();
  var promises = fetchOrder.map(function(stop) {
    var payload = { route: route, stop: stop, hour: hour, day: day, weather: weather, buses: 6, month: window.CURRENT_MONTH };
    return fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(function(res)  { return res.json(); })
      .then(function(data) { results.push({ stop: stop, level: data.prediction, confidence: data.confidence }); });
  });

  Promise.all(promises).then(function() {
    var highCount = 0; var medCount = 0; var lowCount = 0;
    for (var k = 0; k < results.length; k++) {
      if (results[k].level === 'High')   { highCount++; }
      if (results[k].level === 'Medium') { medCount++; }
      if (results[k].level === 'Low')    { lowCount++; }
    }
    var hh = hour < 10 ? '0' + hour : '' + hour;
    var html = '<div class="card-box" style="margin-bottom:16px">'
             + '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:4px">'
             + '<span style="font-size:13px;font-weight:600;color:var(--green)"> ' + lowCount + ' Low</span>'
             + '<span style="font-size:13px;font-weight:600;color:var(--yellow)"> ' + medCount + ' Medium</span>'
             + '<span style="font-size:13px;font-weight:600;color:var(--red)"> ' + highCount + ' High</span>'
             + '</div><p style="font-size:12px;color:#94a3b8;margin:0">Route ' + route + ' | ' + hh + ':00 | ' + day + '</p></div>';

    html += '<div class="card-box"><div class="card-title"><i class="fas fa-bus me-2"></i>All Stops | Route ' + route + '</div>';
    for (var i = 0; i < results.length; i++) {
      var r = results[i]; var cls = crowdClass(r.level);
      html += '<div class="alt-card"><div>'
            + '<div class="alt-stop-name"><span style="font-size:11px;font-weight:500;color:#94a3b8;margin-right:8px">Stop ' + (i + 1) + '</span>'
            + '<i class="fas fa-map-pin me-1" style="color:#cbd5e1;font-size:11px"></i>' + r.stop + '</div>'
            + '<div class="alt-stop-dist">' + r.confidence + '% model confidence</div>'
            + '</div><span class="alt-badge ' + cls + '">' + crowdIcon(r.level) + ' ' + r.level + '</span></div>';
    }
    html += '</div>';
    document.getElementById('fc-result').innerHTML = html;
  });
}

// === BEST TIME ===

function runBestTime() {
  document.getElementById('bt-result').innerHTML =
    '<div class="card-box text-center" style="padding:30px">'
    + '<span class="spinner" style="width:28px;height:28px;border-color:rgba(0,0,0,.15);border-top-color:var(--navy)"></span>'
    + '<p style="color:#94a3b8;margin-top:12px">Analyzing all 24 hours...</p></div>';

  var stop    = document.getElementById('bt-stop').value;
  var route   = document.getElementById('bt-route').value;
  var day     = document.getElementById('bt-day').value;
  var weather = document.getElementById('bt-weather').value;
  var results = [];
  var hours   = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23];

  var promises = hours.map(function(h) {
    var payload = { route: route, stop: stop, hour: h, day: day, weather: weather, buses: 6, month: window.CURRENT_MONTH };
    return fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(function(res)  { return res.json(); })
      .then(function(data) { results.push({ hour: h, level: data.prediction, confidence: data.confidence }); });
  });

  Promise.all(promises).then(function() {
    results.sort(function(a, b) { return a.hour - b.hour; });
    var lowWindows = results.filter(function(r) { return r.level === 'Low'; });
    var html = '';

    if (lowWindows.length > 0) {
      html += '<div class="card-box" style="margin-bottom:16px"><div class="card-title"><i class="fas fa-star me-2" style="color:var(--green)"></i>Best Travel Windows</div><div class="best-windows">';
      var shown = lowWindows.slice(0, 5);
      for (var wi = 0; wi < shown.length; wi++) {
        var w  = shown[wi];
        var wh = w.hour < 10 ? '0' + w.hour : '' + w.hour;
        var wn = (w.hour + 1) % 24;
        var wnS = wn < 10 ? '0' + wn : '' + wn;
        html += '<div class="best-pill"><i class="fas fa-check-circle"></i><span>' + wh + ':00 - ' + wnS + ':00</span></div>';
      }
      html += '</div></div>';
    }

    html += '<div class="card-box"><div class="card-title">All-Day Crowd Heatmap - ' + stop + '</div><div class="hour-grid">';
    for (var ri = 0; ri < results.length; ri++) {
      var r = results[ri]; var cls = crowdClass(r.level);
      var clr = r.level === 'High' ? 'var(--red)' : r.level === 'Medium' ? 'var(--yellow)' : 'var(--green)';
      var rh  = r.hour < 10 ? '0' + r.hour : '' + r.hour;
      html += '<div class="hour-cell ' + cls + '">'
            + '<div style="font-weight:700;font-size:13px;color:#1e293b">' + rh + ':00</div>'
            + '<div style="font-size:16px;margin:4px 0">' + crowdIcon(r.level) + '</div>'
            + '<div style="font-size:11px;font-weight:600;color:' + clr + '">' + r.level + '</div></div>';
    }
    html += '</div></div>';
    document.getElementById('bt-result').innerHTML = html;
  });
}

// === MAP ===

function initMap() {
  if (map) { return; }
  setTimeout(function() {
    map = L.map('map').setView([17.415, 78.44], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '(c) OpenStreetMap' }).addTo(map);
    var coords = window.STOP_COORDS;
    Object.keys(coords).forEach(function(name) {
      L.marker(coords[name]).addTo(map).bindPopup('<strong>' + name + '</strong><br><small>TSRTC Bus Stop</small>');
    });
  }, 100);
}

// === FEEDBACK ===

function selectFb(btn, val) {
  selectedFeedback = val;
  document.querySelectorAll('.fb-btn').forEach(function(b) { b.classList.remove('selected'); });
  btn.classList.add('selected');
}

function submitFeedback() {
  if (!selectedFeedback) { alert('Please select the actual crowd level.'); return; }
  document.getElementById('fb-thanks').style.display = 'block';
  document.querySelectorAll('.fb-btn').forEach(function(b) { b.classList.remove('selected'); });
  selectedFeedback = null;
  setTimeout(function() { document.getElementById('fb-thanks').style.display = 'none'; }, 3000);
}

// === PROFILE ===

function renderProfileHistory() {
  var container = document.getElementById('profile-history');
  if (!predictionHistory.length) { return; }
  var crowdColors = { High: '#fee2e2', Medium: '#fffbeb', Low: '#f0fdf4' };
  var crowdText   = { High: 'var(--red)', Medium: 'var(--yellow)', Low: 'var(--green)' };
  var html = '<div style="display:flex;flex-direction:column;gap:10px">';
  for (var i = 0; i < predictionHistory.length; i++) {
    var p  = predictionHistory[i];
    var hh = parseInt(p.hour) < 10 ? '0' + p.hour : '' + p.hour;
    html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:13px 16px;background:' + crowdColors[p.prediction] + ';border-radius:10px;border:1px solid #e2e8f0">'
          + '<div>'
          + '<div style="font-weight:600;font-size:14px;color:#1e293b"><span style="color:#94a3b8;font-size:11px;margin-right:6px">#' + (i + 1) + '</span>Route ' + p.route + ' - ' + p.stop + '</div>'
          + '<div style="font-size:12px;color:#64748b;margin-top:3px">' + p.day + ' - ' + hh + ':00 - ' + p.time + '</div>'
          + '</div>'
          + '<span style="font-size:13px;font-weight:700;color:' + crowdText[p.prediction] + ';padding:4px 12px;border-radius:20px;background:#fff;border:1px solid currentColor">' + crowdIcon(p.prediction) + ' ' + p.prediction + '</span>'
          + '</div>';
  }
  html += '</div>';
  container.innerHTML = html;
}

function loadProfileData() {
  if (profileData) { return; }
  fetch('/profile-data')
    .then(function(res)  { return res.json(); })
    .then(function(data) {
      profileData = data;
      var initials = data.username ? data.username[0].toUpperCase() : '?';
      document.getElementById('sidebar-avatar').textContent       = initials;
      document.getElementById('sidebar-username').textContent     = data.username;
      document.getElementById('profile-big-avatar').textContent   = initials;
      document.getElementById('profile-display-name').textContent = data.username;
      document.getElementById('profile-username').textContent     = data.username;
      document.getElementById('profile-email').textContent        = data.email;
    })
    .catch(function(e) { console.error('Profile load failed', e); });
}

// === INIT ===

window.addEventListener('DOMContentLoaded', function() {
  filterStops('f-route',  'f-stop');
  filterStops('bt-route', 'bt-stop');
  updateFcStopInfo();
  loadProfileData();
});