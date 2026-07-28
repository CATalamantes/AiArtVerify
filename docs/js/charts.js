/* Hand-rolled SVG charts -- no Plotly/Chart.js. Four chart types, all
   simple enough that a charting library would be the heaviest thing on the
   page for what they do. Colors reused from the validated dataviz palette
   (see css/base.css tokens): status colors for the gauge bands, the
   sequential blue ramp for the heatmap, violet as the single-series accent
   for the growth curve and category bars. */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function clearSvg(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function humanNumber(n) {
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (abs >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return Math.round(n).toLocaleString();
}

// Sequential blue ramp (light -> dark), matching the validated palette.
const SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

function seqColor(t) {
  // t in [0,1] -> interpolate across SEQ_RAMP stops
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (SEQ_RAMP.length - 1);
  const i = Math.floor(scaled);
  const frac = scaled - i;
  if (i >= SEQ_RAMP.length - 1) return SEQ_RAMP[SEQ_RAMP.length - 1];
  return lerpColor(SEQ_RAMP[i], SEQ_RAMP[i + 1], frac);
}

function lerpColor(hexA, hexB, t) {
  const a = hexToRgb(hexA), b = hexToRgb(hexB);
  const r = Math.round(a.r + (b.r - a.r) * t);
  const g = Math.round(a.g + (b.g - a.g) * t);
  const bl = Math.round(a.b + (b.b - a.b) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.substring(0, 2), 16),
    g: parseInt(h.substring(2, 4), 16),
    b: parseInt(h.substring(4, 6), 16),
  };
}

/* ------------------------------------------------------------------------ */
/* Gauge: semicircle 0-100 with band colors + a baseline tick                */
/* ------------------------------------------------------------------------ */
function renderGauge(svg, { score, bandColor, baseRate }) {
  clearSvg(svg);
  const cx = 150, cy = 150, r = 110, thickness = 22;

  const bands = [
    { from: 0, to: 10, color: "rgba(208,59,59,.35)" },
    { from: 10, to: 25, color: "rgba(236,131,90,.35)" },
    { from: 25, to: 50, color: "rgba(250,178,25,.35)" },
    { from: 50, to: 100, color: "rgba(12,163,12,.35)" },
  ];

  const angleFor = (pct) => Math.PI - (pct / 100) * Math.PI; // 180deg (left) -> 0deg (right)

  const arcPath = (fromPct, toPct, radius) => {
    const a0 = angleFor(fromPct), a1 = angleFor(toPct);
    const x0 = cx + radius * Math.cos(a0), y0 = cy - radius * Math.sin(a0);
    const x1 = cx + radius * Math.cos(a1), y1 = cy - radius * Math.sin(a1);
    const largeArc = Math.abs(a0 - a1) > Math.PI ? 1 : 0;
    return `M ${x0} ${y0} A ${radius} ${radius} 0 ${largeArc} 1 ${x1} ${y1}`;
  };

  for (const band of bands) {
    const path = svgEl("path", {
      d: arcPath(band.from, band.to, r),
      fill: "none",
      stroke: band.color,
      "stroke-width": thickness,
      "stroke-linecap": "butt",
    });
    svg.appendChild(path);
  }

  // filled progress arc up to the score, on top of the band track
  const progress = svgEl("path", {
    d: arcPath(0, score, r),
    fill: "none",
    stroke: bandColor,
    "stroke-width": thickness,
    "stroke-linecap": "round",
  });
  svg.appendChild(progress);

  // baseline tick (average trending video)
  const tickAngle = angleFor(baseRate);
  const tx0 = cx + (r - thickness / 2 - 4) * Math.cos(tickAngle);
  const ty0 = cy - (r - thickness / 2 - 4) * Math.sin(tickAngle);
  const tx1 = cx + (r + thickness / 2 + 4) * Math.cos(tickAngle);
  const ty1 = cy - (r + thickness / 2 + 4) * Math.sin(tickAngle);
  svg.appendChild(svgEl("line", {
    x1: tx0, y1: ty0, x2: tx1, y2: ty1,
    stroke: "#ffffff", "stroke-width": 3, "stroke-linecap": "round",
  }));

  // axis labels
  [0, 25, 50, 100].forEach((v) => {
    const a = angleFor(v);
    const lx = cx + (r + thickness / 2 + 16) * Math.cos(a);
    const ly = cy - (r + thickness / 2 + 16) * Math.sin(a);
    const t = svgEl("text", {
      x: lx, y: ly + 4, "text-anchor": v === 0 ? "start" : v === 100 ? "end" : "middle",
      fill: "#898781", "font-size": "11",
    });
    t.textContent = v;
    svg.appendChild(t);
  });
}

/* ------------------------------------------------------------------------ */
/* "Where you land": histogram of all 6,668 trending videos' view counts,   */
/* with a marker line at this prediction. Log-scale x-axis since raw view   */
/* counts span 4+ orders of magnitude.                                     */
/* ------------------------------------------------------------------------ */
function renderViewsHistogram(svg, { distLogViews, myViews, nBins = 24 }) {
  clearSvg(svg);
  const W = 480, H = 220, padL = 40, padR = 12, padT = 20, padB = 28;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const xMin = Math.min(...distLogViews), xMax = Math.max(...distLogViews);
  const binWidth = (xMax - xMin) / nBins;
  const counts = new Array(nBins).fill(0);
  distLogViews.forEach((v) => {
    const i = Math.min(nBins - 1, Math.max(0, Math.floor((v - xMin) / binWidth)));
    counts[i]++;
  });
  const maxCount = Math.max(...counts);

  const xScale = (v) => padL + ((v - xMin) / (xMax - xMin)) * plotW;
  const yScale = (c) => padT + plotH - (c / maxCount) * plotH;

  const barW = plotW / nBins;
  counts.forEach((c, i) => {
    const x = padL + i * barW;
    const y = yScale(c);
    svg.appendChild(svgEl("rect", {
      x: x + 0.5, y, width: Math.max(barW - 1, 1), height: padT + plotH - y,
      fill: "#2c2c2a",
    }));
  });

  // marker at this prediction (log1p(views), matching how the backend built
  // the distribution -- log1p/expm1 are exact inverses so this lines up).
  const myLogViews = Math.log1p(myViews);
  const mx = xScale(Math.min(Math.max(myLogViews, xMin), xMax));
  svg.appendChild(svgEl("line", {
    x1: mx, x2: mx, y1: padT - 6, y2: padT + plotH,
    stroke: "#9085e9", "stroke-width": 3, "stroke-linecap": "round",
  }));
  const label = svgEl("text", {
    x: mx, y: padT - 10, "text-anchor": mx > W - 70 ? "end" : mx < 70 ? "start" : "middle",
    fill: "#9085e9", "font-size": "10", "font-weight": "700",
  });
  label.textContent = "YOUR VIDEO";
  svg.appendChild(label);

  // x-axis labels in real view-count units
  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
    const logV = xMin + (xMax - xMin) * f;
    const t = svgEl("text", {
      x: xScale(logV), y: H - 8, "text-anchor": "middle", fill: "#898781", "font-size": "10",
    });
    t.textContent = humanNumber(Math.expm1(logV));
    svg.appendChild(t);
  });
}

/* ------------------------------------------------------------------------ */
/* Growth curve: area + line, hours (x) vs predicted views (y)              */
/* ------------------------------------------------------------------------ */
function renderGrowthCurve(svg, { hours, views, horizonHours }) {
  clearSvg(svg);
  const W = 480, H = 220, padL = 46, padR = 12, padT = 12, padB = 28;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const xMin = Math.min(...hours), xMax = Math.max(...hours);
  const yMax = Math.max(...views) * 1.08;

  const xScale = (h) => padL + ((h - xMin) / (xMax - xMin)) * plotW;
  const yScale = (v) => padT + plotH - (v / yMax) * plotH;

  // gridlines
  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
    const y = padT + plotH * (1 - f);
    svg.appendChild(svgEl("line", {
      x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#2c2c2a", "stroke-width": 1,
    }));
    const label = svgEl("text", {
      x: padL - 8, y: y + 4, "text-anchor": "end", fill: "#898781", "font-size": "10",
    });
    label.textContent = humanNumber(yMax * f);
    svg.appendChild(label);
  });

  const points = hours.map((h, i) => [xScale(h), yScale(views[i])]);
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  const areaPath = `${linePath} L ${points[points.length - 1][0]} ${padT + plotH} L ${points[0][0]} ${padT + plotH} Z`;

  svg.appendChild(svgEl("path", { d: areaPath, fill: "rgba(144,133,233,.16)", stroke: "none" }));
  svg.appendChild(svgEl("path", { d: linePath, fill: "none", stroke: "#9085e9", "stroke-width": 2.5 }));

  // horizon marker
  const hx = xScale(horizonHours);
  svg.appendChild(svgEl("line", {
    x1: hx, x2: hx, y1: padT, y2: padT + plotH,
    stroke: "#898781", "stroke-width": 1, "stroke-dasharray": "3,3",
  }));
  const closestIdx = hours.reduce((best, h, i) => Math.abs(h - horizonHours) < Math.abs(hours[best] - horizonHours) ? i : best, 0);
  svg.appendChild(svgEl("circle", {
    cx: hx, cy: yScale(views[closestIdx]), r: 5, fill: "#9085e9", stroke: "#1a1a19", "stroke-width": 2,
  }));

  // x-axis labels
  [xMin, (xMin + xMax) / 2, xMax].forEach((h) => {
    const t = svgEl("text", {
      x: xScale(h), y: H - 8, "text-anchor": "middle", fill: "#898781", "font-size": "10",
    });
    t.textContent = `${Math.round(h)}h`;
    svg.appendChild(t);
  });
}

/* ------------------------------------------------------------------------ */
/* Heatmap: 7 (day) x 24 (hour) grid, colored by P(hit)                     */
/* ------------------------------------------------------------------------ */
function renderHeatmap(svg, { grid, days, currentDay, currentHour }) {
  clearSvg(svg);
  const W = 480, H = 220, padL = 34, padR = 8, padT = 6, padB = 20;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const cols = 24, rows = 7;
  const cellW = plotW / cols, cellH = plotH / rows;

  const flat = grid.flat();
  const min = Math.min(...flat), max = Math.max(...flat);
  const norm = (v) => (max === min ? 0.5 : (v - min) / (max - min));

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = padL + c * cellW, y = padT + r * cellH;
      const isCurrent = r === currentDay && c === currentHour;
      svg.appendChild(svgEl("rect", {
        x, y, width: cellW - 1, height: cellH - 1,
        fill: seqColor(norm(grid[r][c])),
        stroke: isCurrent ? "#ffffff" : "none",
        "stroke-width": isCurrent ? 2 : 0,
      }));
    }
  }

  days.forEach((d, r) => {
    const t = svgEl("text", {
      x: padL - 6, y: padT + r * cellH + cellH / 2 + 3, "text-anchor": "end",
      fill: "#898781", "font-size": "9",
    });
    t.textContent = d.slice(0, 3);
    svg.appendChild(t);
  });

  [0, 6, 12, 18, 23].forEach((h) => {
    const t = svgEl("text", {
      x: padL + h * cellW + cellW / 2, y: H - 6, "text-anchor": "middle",
      fill: "#898781", "font-size": "9",
    });
    t.textContent = h;
    svg.appendChild(t);
  });
}

/* ------------------------------------------------------------------------ */
/* Category fit: horizontal bar list                                        */
/* ------------------------------------------------------------------------ */
function renderCategoryBars(svg, { categories, currentCategory }) {
  clearSvg(svg);
  const W = 300, padL = 8, padR = 44, rowH = 26, gap = 6;
  const H = categories.length * (rowH + gap);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const maxProb = Math.max(...categories.map((c) => c.probability), 0.01);
  const plotW = W - padL - padR;

  categories.forEach((cat, i) => {
    const y = i * (rowH + gap);
    const w = (cat.probability / maxProb) * plotW;
    const isCurrent = cat.category === currentCategory;

    svg.appendChild(svgEl("rect", {
      x: padL, y, width: plotW, height: rowH, fill: "#232322", rx: 4,
    }));
    svg.appendChild(svgEl("rect", {
      x: padL, y, width: Math.max(w, 2), height: rowH, rx: 4,
      fill: isCurrent ? "#0ca30c" : "#9085e9",
      opacity: isCurrent ? 1 : 0.75,
    }));
    const label = svgEl("text", {
      x: padL + 8, y: y + rowH / 2 + 4, fill: "#ffffff", "font-size": "11", "font-weight": isCurrent ? "700" : "400",
    });
    label.textContent = cat.category;
    svg.appendChild(label);

    const score = svgEl("text", {
      x: W - padR + 8, y: y + rowH / 2 + 4, fill: "#c3c2b7", "font-size": "11",
    });
    score.textContent = Math.round(cat.probability * 100);
    svg.appendChild(score);
  });
}
