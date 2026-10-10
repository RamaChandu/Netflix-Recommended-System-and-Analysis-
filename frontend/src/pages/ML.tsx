import {useEffect, useMemo, useState} from "react";
import {api} from "../services/api";
import Chart from "../components/Chart";

const classificationModels = [
  ["logistic_regression", "Logistic Regression"],
  ["decision_tree", "Decision Tree"],
  ["random_forest", "Random Forest"],
  ["gradient_boosting", "Gradient Boosting"],
] as const;

const regressionModels = [
  ["linear_regression", "Linear Regression"],
  ["ridge", "Ridge Regression"],
  ["lasso", "Lasso Regression"],
  ["random_forest", "Random Forest"],
  ["gradient_boosting", "Gradient Boosting"],
] as const;

export default function ML() {
  const [task, setTask] = useState<"classification" | "regression">("classification");
  const [algorithm, setAlgorithm] = useState("logistic_regression");
  const [result, setResult] = useState<any>(null);
  const [comparison, setComparison] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const models = useMemo(
    () => task === "classification" ? classificationModels : regressionModels,
    [task]
  );

  useEffect(() => {
    setAlgorithm(models[0][0]);
    setResult(null);
    setComparison([]);
    setError("");
  }, [task]);

  const selectedName = models.find(([id]) => id === algorithm)?.[1] ?? algorithm;

  async function trainSelected() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const trained = await api.train({task, algorithm});
      setResult(trained);
      const ranked = await api.comparison(task);
      setComparison(ranked);
    } catch (e: any) {
      setError(e?.message || "Could not train the selected model. Check that the FastAPI backend is running on port 8000.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm text-red-400 font-medium">MACHINE LEARNING LAB</p>
        <h1 className="text-3xl font-bold mt-1">ML Studio</h1>
        <p className="text-zinc-500 mt-1">
          Select a task and model, then train it on the supplied Netflix metadata.
        </p>
      </div>

      <div className="border border-zinc-800 rounded-xl bg-zinc-900/40 p-5 space-y-5">
        <div className="grid md:grid-cols-2 gap-4">
          <label className="block">
            <span className="text-sm text-zinc-400">1. Choose ML task</span>
            <select
              value={task}
              onChange={(e) => setTask(e.target.value as "classification" | "regression")}
              className="mt-2 w-full bg-zinc-950 border border-zinc-700 rounded-lg px-4 py-3 outline-none focus:border-red-500"
            >
              <option value="classification">Classification — predict Movie / TV Show</option>
              <option value="regression">Regression — estimate release year</option>
            </select>
          </label>

          <label className="block">
            <span className="text-sm text-zinc-400">2. Select model</span>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="mt-2 w-full bg-zinc-950 border border-zinc-700 rounded-lg px-4 py-3 outline-none focus:border-red-500"
            >
              {models.map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <div className="text-sm text-zinc-500">Selected model</div>
            <div className="font-semibold text-lg">{selectedName}</div>
          </div>
          <button
            onClick={trainSelected}
            disabled={busy}
            className="bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-medium"
          >
            {busy ? "Training model…" : "Train Selected Model"}
          </button>
        </div>

        {error && (
          <div className="border border-red-900 bg-red-950/40 text-red-300 rounded-lg p-4 text-sm whitespace-pre-wrap">
            <b>Training failed:</b> {error}
          </div>
        )}
      </div>

      {result && (
        <>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {task === "classification" ? (
              <>
                <Metric label="Accuracy" value={result.metrics.accuracy} />
                <Metric label="Precision" value={result.metrics.precision} />
                <Metric label="Recall" value={result.metrics.recall} />
                <Metric label="F1 Score" value={result.metrics.f1} />
              </>
            ) : (
              <>
                <Metric label="R²" value={result.metrics.r2} />
                <Metric label="MAE" value={result.metrics.mae} />
                <Metric label="RMSE" value={result.metrics.rmse} />
                <Metric label="Training time" value={result.training_time_seconds} suffix="s" />
              </>
            )}
          </div>

          {task === "classification" && result.metrics.confusion_matrix && (
            <div className="border border-zinc-800 rounded-xl p-5">
              <h2 className="font-semibold mb-4">Confusion Matrix</h2>
              <div className="overflow-x-auto">
                <table className="border-collapse text-sm">
                  <tbody>
                    {result.metrics.confusion_matrix.map((row: number[], i: number) => (
                      <tr key={i}>
                        {row.map((value, j) => (
                          <td key={j} className="border border-zinc-700 px-8 py-5 text-center bg-zinc-900">
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="text-xs text-zinc-500 mt-3">
                Class order: {result.metrics.labels?.join(" · ")}
              </div>
            </div>
          )}

          {task === "regression" && result.metrics.actual_vs_predicted && (
            <div className="border border-zinc-800 rounded-xl p-3 h-96">
              <h2 className="font-semibold px-2">Actual vs Predicted Release Year</h2>
              <Chart
                data={[{
                  x: result.metrics.actual_vs_predicted.map((x: any) => x.actual),
                  y: result.metrics.actual_vs_predicted.map((x: any) => x.predicted),
                  type: "scatter",
                  mode: "markers",
                  name: selectedName
                }]}
                layout={{xaxis:{title:"Actual release year"}, yaxis:{title:"Predicted release year"}}}
              />
            </div>
          )}
        </>
      )}

      {comparison.length > 0 && (
        <div className="border border-zinc-800 rounded-xl overflow-hidden">
          <div className="p-4 font-semibold">Model Comparison</div>
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-zinc-400">
              <tr>
                <th className="p-3 text-left">Model</th>
                <th className="p-3 text-right">Primary metric</th>
                <th className="p-3 text-right">Training time</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((r: any) => (
                <tr className="border-t border-zinc-800" key={r.algorithm}>
                  <td className="p-3">{models.find(([id]) => id === r.algorithm)?.[1] ?? r.algorithm}</td>
                  <td className="p-3 text-right">
                    {task === "classification" ? r.metrics.f1.toFixed(4) : r.metrics.rmse.toFixed(2)}
                  </td>
                  <td className="p-3 text-right">{r.training_time_seconds}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="border border-zinc-800 rounded-xl p-5 text-sm text-zinc-500">
        <b className="text-zinc-300">Note:</b> These are educational machine-learning experiments using metadata from the supplied dataset. They do not use Netflix user viewing history and are not Netflix's production models.
      </div>
    </div>
  );
}

function Metric({label, value, suffix=""}:{label:string;value:number;suffix?:string}) {
  return (
    <div className="border border-zinc-800 rounded-xl p-4 bg-zinc-900/40">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="text-2xl font-semibold mt-2">{typeof value === "number" ? value.toFixed(4) : value}{suffix}</div>
    </div>
  );
}
