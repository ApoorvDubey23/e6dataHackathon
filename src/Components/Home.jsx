// src/pages/Home.jsx
import React, { useMemo, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import axios from "axios";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const API = "http://127.0.0.1:8000";

export default function Home() {
  const [datasetId, setDatasetId] = useState(null);
  const [columns, setColumns] = useState([]);
  const [rowCount, setRowCount] = useState(0);

  const [sumCol, setSumCol] = useState("");
  const [avgCol, setAvgCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [stratifyCol, setStratifyCol] = useState("");

  const [samplePercents, setSamplePercents] = useState([5, 7, 10, 15, 20]);

  const [values, setValues] = useState(null);
  const [times, setTimes] = useState(null);
  const [grouped, setGrouped] = useState(null); // backend grouped_example
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // ---- Upload
  const onFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      setLoading(true);
      const res = await axios.post(`${API}/upload_csv`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDatasetId(res.data.dataset_id);
      setColumns(res.data.columns || []);
      setRowCount(res.data.row_count || 0);
      setErrorMsg("");
    } catch (err) {
      console.error(err);
      setErrorMsg(err?.response?.data?.detail || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  // ---- Transform /init payload (raw) -> {values, times, meta}
  const transformInitPayload = (payload) => {
    const outValues = {};
    const outTimes = {};

    const ensure = (metric) => {
      if (!outValues[metric]) outValues[metric] = { Exact: [], Simple: [] };
      if (!outTimes[metric]) outTimes[metric] = { Exact: [], Simple: [] };
    };

    for (const block of payload.results || []) {
      const qmap = Object.fromEntries((block.queries || []).map((q) => [q.name, q]));

      // COUNT + AVG
      const q1 = qmap["COUNT + AVG"];
      if (q1) {
        const exactRows = q1.exact?.rows || [{}];
        const approxRows = q1.approx?.rows || [{}];
        const exactTime = q1.exact?.time_sec || 0;
        const approxTime = q1.approx?.time_sec || 0;

        ensure("total_trips");
        outValues["total_trips"].Exact.push(exactRows[0]?.total_trips ?? null);
        outValues["total_trips"].Simple.push(
          typeof approxRows[0] === "object" ? approxRows[0]?.total_trips_estimate ?? null : null
        );
        outTimes["total_trips"].Exact.push(Math.round(exactTime * 1000));
        outTimes["total_trips"].Simple.push(Math.round(approxTime * 1000));

        ensure("avg_revenue");
        outValues["avg_revenue"].Exact.push(exactRows[0]?.avg_revenue ?? null);
        let approxAvg = null;
        if (typeof approxRows[0] === "object") {
          approxAvg = approxRows[0]?.avg_revenue_estimate ?? null;
          if (approxAvg == null) {
            const s = approxRows[0]?.__v_sum_avg_revenue ?? 0;
            const c = approxRows[0]?.__v_count_avg_revenue ?? 0;
            approxAvg = c ? s / c : null;
          }
        }
        outValues["avg_revenue"].Simple.push(approxAvg);
        outTimes["avg_revenue"].Exact.push(Math.round(exactTime * 1000));
        outTimes["avg_revenue"].Simple.push(Math.round(approxTime * 1000));
      }

      // SUM
      const q2 = qmap["SUM"];
      if (q2) {
        const exactRows = q2.exact?.rows || [{}];
        const approxRows = q2.approx?.rows || [{}];
        const exactTime = q2.exact?.time_sec || 0;
        const approxTime = q2.approx?.time_sec || 0;

        ensure("total_revenue");
        outValues["total_revenue"].Exact.push(exactRows[0]?.total_revenue ?? null);
        outValues["total_revenue"].Simple.push(
          typeof approxRows[0] === "object" ? approxRows[0]?.total_revenue_estimate ?? null : null
        );
        outTimes["total_revenue"].Exact.push(Math.round(exactTime * 1000));
        outTimes["total_revenue"].Simple.push(Math.round(approxTime * 1000));
      }
    }

    return {
      values: outValues,
      times: outTimes,
      meta: {
        sample_percentages: (payload.results || []).map((r) => Math.round((r.ratio || 0) * 100)),
      },
    };
  };

  // ---- Process
  const runProcess = async () => {
    setLoading(true);
    setErrorMsg("");
    setGrouped(null);
    try {
      if (datasetId) {
        // POST /process
        const body = {
          dataset_id: datasetId,
          sample_percentages: samplePercents,
          sum_column: sumCol || null,
          avg_column: avgCol || null,
          group_by_column: groupCol || null,
          stratify_column: stratifyCol || null,
        };
        const res = await axios.post(`${API}/process`, body);
        setValues(res.data.values);
        setTimes(res.data.times);
        setSamplePercents(res.data.meta?.sample_percentages || samplePercents);
        if (res.data.grouped_example) {
          setGrouped(res.data.grouped_example);
        }
      } else {
        // GET /init (no grouped demo unless you pass group_by below)
        const ratiosStr = (samplePercents || [])
          .map((p) => (Math.max(0, Math.min(100, p)) / 100).toFixed(2))
          .join(",");
        const params = {
          dataset_filename: "yellow_tripdata_2015-01.csv",
          table_name: "trips",
          stratify_col: stratifyCol || "RateCodeID",
          hash_col: "payment_type",
          numeric_col: avgCol || sumCol || "total_amount",
          ratios: ratiosStr,
          confidence: 0.95,
          verbose: false,
          group_by: groupCol || undefined,
        };
        const res = await axios.get(`${API}/init`, { params });
        const shaped = transformInitPayload(res.data);
        setValues(shaped.values);
        setTimes(shaped.times);
        setSamplePercents(shaped.meta.sample_percentages || samplePercents);
        if (res.data.grouped_example) setGrouped(res.data.grouped_example);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(err?.response?.data?.detail || "Process failed");
    } finally {
      setLoading(false);
    }
  };

  const numericCols = useMemo(
    () => columns.filter((c) => /int|float|double|number|bool/i.test(c.dtype)),
    [columns]
  );
  const allCols = useMemo(() => columns.map((c) => c.name), [columns]);

  // Chart helpers
  const palette = {
    Exact: "#60a5fa",
    Simple: "#34d399",
    Stratified: "#fbbf24",
    Systematic: "#a78bfa",
    Cluster: "#f87171",
  };
  const fmt = (v) =>
    Math.abs(v) >= 1000
      ? new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(v)
      : `${v}`;

  const labels = useMemo(() => (samplePercents || []).map((p) => `${p}%`), [samplePercents]);
  const seriesStyle = (name) => ({
    label: name,
    borderColor: palette[name],
    backgroundColor: palette[name],
    borderWidth: name === "Exact" ? 2.5 : 2,
    borderDash: name === "Exact" ? [5, 4] : undefined,
    tension: 0.35,
    pointRadius: 2.5,
    pointHoverRadius: 5,
  });
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: {
        position: "top",
        labels: { color: "#e5e7eb", font: { size: 12 } },
      },
      tooltip: {
        backgroundColor: "#111827",
        borderColor: "#374151",
        borderWidth: 1,
        padding: 10,
        titleColor: "#fff",
        bodyColor: "#e5e7eb",
        callbacks: { label: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.parsed.y)}` },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: "#cbd5e1", font: { size: 12, weight: "600" } } },
      y: {
        beginAtZero: false,
        grace: "10%",
        ticks: { color: "#94a3b8", font: { size: 11 }, callback: (v) => fmt(v) },
        grid: { color: "rgba(148,163,184,0.25)", borderDash: [4, 4] },
      },
    },
  };
  const buildChartData = (metricObj) => ({
    labels,
    datasets: Object.entries(metricObj || {}).map(([name, arr]) => ({
      ...seriesStyle(name),
      data: arr || [],
    })),
  });

  const groupedRows = useMemo(() => {
    if (!grouped?.exact?.rows?.length) return [];
    const exactMap = {};
    for (const r of grouped.exact.rows) exactMap[r[Object.keys(r)[0]]] = r; // keyed by group value

    const approx = grouped.approx?.rows || [];
    const out = [];
    for (const a of approx) {
      const keys = Object.keys(a).filter((k) => !k.endsWith("_lower") && !k.endsWith("_upper") && !k.endsWith("_estimate"));
      // Expect structure: { groupCol, trips_estimate, avg_value_estimate, ... }
      const gval = a[groupCol];
      if (gval === undefined) continue;
      const e = exactMap[gval] || {};
      out.push({
        group: gval,
        trips_exact: e.trips ?? null,
        trips_est: a.trips_estimate ?? null,
        avg_exact: e.avg_value ?? null,
        avg_est: a.avg_value_estimate ?? null,
      });
    }
    // fallback: if approx empty, show exact only
    if (!out.length) {
      for (const e of grouped.exact.rows) {
        out.push({
          group: e[groupCol],
          trips_exact: e.trips ?? null,
          trips_est: null,
          avg_exact: e.avg_value ?? null,
          avg_est: null,
        });
      }
    }
    return out;
  }, [grouped, groupCol]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="mx-auto max-w-7xl px-4 md:px-6 py-6 space-y-6">
        <div className="flex items-end justify-between">
          <div>
            <h2 className="text-2xl md:text-3xl font-semibold">AQP Curves</h2>
            <p className="text-sm text-gray-300">
              Upload CSV → pick columns → run AQP → see Value/Time curves for Exact vs Approx
            </p>
          </div>
          <input
            type="file"
            accept=".csv"
            onChange={onFileChange}
            className="block rounded-md bg-gray-800 border border-gray-700 px-3 py-2 text-sm"
            disabled={loading}
          />
        </div>

        {datasetId && (
          <div className="text-xs text-gray-400">Using uploaded dataset. Rows: {rowCount}</div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 bg-gray-800/40 border border-gray-800 rounded-lg p-4">
          <div className="md:col-span-1">
            <div className="text-xs text-gray-400 mb-1">Dataset</div>
            <div className="text-sm break-all">
              {datasetId || "Default: yellow_tripdata_2015-01.csv"}
            </div>
            {datasetId ? (
              <div className="text-xs text-gray-400 mt-1">{rowCount} rows</div>
            ) : (
              <div className="text-xs text-gray-400 mt-1">No upload — will use default</div>
            )}
          </div>

          <div>
            <label className="block text-sm mb-1">SUM column</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-md p-2"
              value={sumCol}
              onChange={(e) => setSumCol(e.target.value)}
              disabled={loading}
            >
              <option value="">(none)</option>
              {numericCols.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">AVG column</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-md p-2"
              value={avgCol}
              onChange={(e) => setAvgCol(e.target.value)}
              disabled={loading}
            >
              <option value="">(none)</option>
              {numericCols.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">GROUP BY column</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-md p-2"
              value={groupCol}
              onChange={(e) => setGroupCol(e.target.value)}
              disabled={loading}
            >
              <option value="">(none)</option>
              {columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
            <div className="text-xs text-gray-400 mt-1">
              If set, backend returns a small grouped preview (COUNT + AVG).
            </div>
          </div>

          <div>
            <label className="block text-sm mb-1">Stratify (optional)</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-md p-2"
              value={stratifyCol}
              onChange={(e) => setStratifyCol(e.target.value)}
              disabled={loading}
            >
              <option value="">(none)</option>
              {columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </div>

          <div className="md:col-span-5 flex items-center gap-3">
            <label className="text-sm">Sample % (comma sep)</label>
            <input
              className="flex-1 bg-gray-800 border border-gray-700 rounded-md p-2 text-sm"
              value={samplePercents.join(",")}
              onChange={(e) => {
                const v = e.target.value
                  .split(",")
                  .map((s) => parseInt(s.trim(), 10))
                  .filter((n) => Number.isFinite(n) && n > 0);
                setSamplePercents(v.length ? v : [5, 7, 10, 15, 20]);
              }}
              disabled={loading}
            />
            <button
              onClick={runProcess}
              disabled={loading}
              className="px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-500 text-sm font-medium disabled:opacity-50"
            >
              {loading ? "Running…" : "Run AQP"}
            </button>
          </div>
        </div>

        {errorMsg && <div className="text-red-400 text-sm">{errorMsg}</div>}

        {/* CHARTS */}
        {values && times && (
          <>
            {Object.keys(values).map((metric) => (
              <div key={metric} className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8">
                <section className="rounded-xl border border-gray-800 bg-gray-800/40 shadow-sm p-4 h-[56vh]">
                  <h3 className="text-base md:text-lg font-medium mb-2 text-gray-100">
                    {metric} — Estimated Value vs Sample %
                  </h3>
                  <div className="h-[calc(100%-2rem)]">
                    <Line data={buildChartData(values[metric])} options={baseOptions} />
                  </div>
                </section>

                <section className="rounded-xl border border-gray-800 bg-gray-800/40 shadow-sm p-4 h-[56vh]">
                  <h3 className="text-base md:text-lg font-medium mb-2 text-gray-100">
                    {metric} — Time (ms) vs Sample %
                  </h3>
                  <div className="h-[calc(100%-2rem)]">
                    <Line data={buildChartData(times[metric])} options={baseOptions} />
                  </div>
                </section>
              </div>
            ))}
          </>
        )}

        {/* GROUPED PREVIEW TABLE */}
        {groupCol && grouped && (
          <section className="rounded-xl border border-gray-800 bg-gray-800/40 shadow-sm p-4">
            <h3 className="text-base md:text-lg font-medium mb-3 text-gray-100">
              Grouped Preview — GROUP BY {groupCol}
            </h3>
            <div className="text-xs text-gray-400 mb-2">
              SQL: <code className="bg-gray-900 px-2 py-1 rounded">{grouped.sql}</code>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-300">
                    <th className="px-2 py-2 border-b border-gray-800">{groupCol}</th>
                    <th className="px-2 py-2 border-b border-gray-800">Trips (Exact)</th>
                    <th className="px-2 py-2 border-b border-gray-800">Trips (Approx)</th>
                    <th className="px-2 py-2 border-b border-gray-800">Avg (Exact)</th>
                    <th className="px-2 py-2 border-b border-gray-800">Avg (Approx)</th>
                  </tr>
                </thead>
                <tbody>
                  {groupedRows.slice(0, 20).map((r, i) => (
                    <tr key={i} className="border-b border-gray-800">
                      <td className="px-2 py-2">{String(r.group)}</td>
                      <td className="px-2 py-2">{r.trips_exact ?? "—"}</td>
                      <td className="px-2 py-2">{r.trips_est != null ? Math.round(r.trips_est) : "—"}</td>
                      <td className="px-2 py-2">{r.avg_exact != null ? r.avg_exact.toFixed(2) : "—"}</td>
                      <td className="px-2 py-2">{r.avg_est != null ? r.avg_est.toFixed(2) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(grouped?.exact?.time_sec != null || grouped?.approx?.time_sec != null) && (
              <div className="text-xs text-gray-400 mt-2">
                Time — Exact: {Math.round((grouped.exact?.time_sec || 0) * 1000)} ms,&nbsp;
                Approx: {Math.round((grouped.approx?.time_sec || 0) * 1000)} ms
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
