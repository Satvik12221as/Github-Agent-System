"use client";

import { useState, useRef, useEffect } from "react";
import { Search, Terminal, GitPullRequest, ArrowRight, CheckCircle2, XCircle, Loader2, Bot, AlertTriangle } from "lucide-react";

type AgentNode = "code_reader" | "planner" | "code_writer" | "test_writer" | "pr_opener" | null;

interface PipelineEvent {
  node: AgentNode;
  steps: number;
  error: string | null;
  pr_url: string | null;
  test_result: string | null;
  retry_count: number;
  complexity: string;
}

const AGENTS = [
  { id: "code_reader", label: "Context Retrieval" },
  { id: "planner", label: "Action Plan" },
  { id: "code_writer", label: "Code Generation" },
  { id: "test_writer", label: "Sandbox Tests" },
  { id: "pr_opener", label: "Ship PR" },
];

export default function Home() {
  const [issueUrl, setIssueUrl] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [activeNode, setActiveNode] = useState<AgentNode>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<PipelineEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handleFix = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!issueUrl.trim()) return;

    setIsRunning(true);
    setLogs(["System initialized. Connecting to agents..."]);
    setActiveNode(null);
    setResult(null);
    setError(null);

    const eventSource = new EventSource(`http://localhost:8000/stream_fix?issue_url=${encodeURIComponent(issueUrl)}`);

    eventSource.onmessage = (event) => {
      try {
        const data: PipelineEvent = JSON.parse(event.data);
        setActiveNode(data.node);
        
        let logMsg = `> Agent [${data.node}] completed step. (Total steps: ${data.steps})`;
        if (data.node === "test_writer") {
           logMsg += ` | Tests: ${data.test_result?.toUpperCase() || 'UNKNOWN'} (Retries: ${data.retry_count})`;
        }
        
        setLogs((prev) => [...prev, logMsg]);

        if (data.error && !data.error.includes("Tests failed")) {
          // If it's a hard error (not just tests failing which leads to retry)
          setLogs((prev) => [...prev, `[ERROR] ${data.error}`]);
          if (data.node === "pr_opener" || data.error.includes("Circuit breaker")) {
             setError(data.error);
             eventSource.close();
             setIsRunning(false);
          }
        }

        if (data.pr_url || data.node === "pr_opener") {
          setLogs((prev) => [...prev, `Pipeline finished. PR: ${data.pr_url || 'Not created'}`]);
          setResult(data);
          eventSource.close();
          setIsRunning(false);
          setActiveNode(null);
        }
      } catch (err) {
        console.error("Failed to parse event", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("EventSource error", err);
      setError("Connection to API lost or server error occurred.");
      eventSource.close();
      setIsRunning(false);
    };
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-slate-200 selection:bg-blue-500/30">
      
      {/* Navbar */}
      <nav className="border-b border-white/10 bg-[#161b22]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl tracking-tight">
            <Bot className="w-6 h-6 text-emerald-400" />
            <span className="bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">GitFix AI</span>
          </div>
          <div className="flex items-center gap-4 text-sm font-medium text-slate-400">
            <span>Powered by LangGraph</span>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12 grid lg:grid-cols-[1fr_400px] gap-8">
        
        {/* Left Column: Input & Terminal */}
        <div className="space-y-8">
          
          {/* Header & Input */}
          <div className="space-y-4">
            <h1 className="text-4xl font-extrabold tracking-tight">Autonomous Bug Resolution</h1>
            <p className="text-slate-400 text-lg">Paste a GitHub issue. Watch our AI agents write code, run sandbox tests, and open a Pull Request in real-time.</p>
            
            <form onSubmit={handleFix} className="mt-6 flex gap-3 relative">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  type="url"
                  placeholder="https://github.com/owner/repo/issues/42"
                  value={issueUrl}
                  onChange={(e) => setIssueUrl(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-[#161b22] border border-white/10 rounded-xl py-4 pl-12 pr-4 outline-none focus:border-blue-500/50 focus:ring-4 focus:ring-blue-500/10 transition-all disabled:opacity-50"
                  required
                />
              </div>
              <button
                type="submit"
                disabled={isRunning || !issueUrl}
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-8 rounded-xl font-semibold flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Fixing...
                  </>
                ) : (
                  <>
                    Fix Issue <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Terminal View */}
          <div className="rounded-xl border border-white/10 bg-[#0d1117] overflow-hidden shadow-2xl relative group">
            <div className="bg-[#161b22] border-b border-white/5 px-4 py-3 flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
              </div>
              <div className="ml-4 flex items-center gap-2 text-xs font-medium text-slate-500">
                <Terminal className="w-3.5 h-3.5" />
                <span>agent-execution.log</span>
              </div>
            </div>
            
            <div className="p-6 font-mono text-sm h-[400px] overflow-y-auto space-y-2">
              {logs.length === 0 ? (
                <div className="text-slate-600 italic">Waiting for pipeline to start...</div>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className={`
                    ${log.includes("[ERROR]") ? "text-red-400" : ""}
                    ${log.includes("Pipeline finished") ? "text-emerald-400 font-semibold" : "text-slate-300"}
                  `}>
                    {log}
                  </div>
                ))
              )}
              {isRunning && (
                <div className="flex items-center gap-2 text-emerald-500/70 animate-pulse mt-4">
                  <span className="block w-2 h-4 bg-emerald-500/70"></span>
                  Processing...
                </div>
              )}
              <div ref={logsEndRef} />
            </div>
            
            {/* Glass reflection effect */}
            <div className="absolute inset-0 bg-gradient-to-tr from-white/[0.02] to-transparent pointer-events-none"></div>
          </div>
        </div>

        {/* Right Column: Visual Tracker */}
        <div className="space-y-6">
          <div className="rounded-xl border border-white/10 bg-[#161b22]/50 p-6 backdrop-blur-sm">
            <h3 className="font-semibold text-lg flex items-center gap-2 mb-6">
              <Bot className="w-5 h-5 text-blue-400" />
              Agent Swarm
            </h3>
            
            <div className="space-y-0 relative">
              {/* Connecting Line */}
              <div className="absolute left-[19px] top-6 bottom-6 w-0.5 bg-white/5"></div>
              
              {AGENTS.map((agent, index) => {
                const isActive = activeNode === agent.id;
                const isPast = activeNode && AGENTS.findIndex(a => a.id === activeNode) > index;
                const isFinished = result !== null || error !== null;
                const isDone = isPast || (isFinished && !error);
                
                return (
                  <div key={agent.id} className="relative flex items-center gap-4 py-4 z-10">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-500 ${
                      isActive 
                        ? "bg-blue-500/20 border-blue-500 text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.5)]" 
                        : isDone
                        ? "bg-emerald-500/20 border-emerald-500 text-emerald-400"
                        : "bg-[#0d1117] border-white/10 text-slate-600"
                    }`}>
                      {isActive ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : isDone ? (
                        <CheckCircle2 className="w-5 h-5" />
                      ) : (
                        <span className="text-xs font-bold">{index + 1}</span>
                      )}
                    </div>
                    <div>
                      <div className={`font-medium transition-colors ${isActive ? "text-white" : isDone ? "text-slate-300" : "text-slate-600"}`}>
                        {agent.label}
                      </div>
                      <div className="text-xs text-slate-500">
                        {isActive ? "Currently executing..." : isDone ? "Completed successfully" : "Pending"}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Results Card */}
          {result && (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-6 animate-in slide-in-from-bottom-4 fade-in duration-500">
              <div className="flex items-center gap-3 text-emerald-400 mb-4">
                <CheckCircle2 className="w-6 h-6" />
                <h3 className="font-bold text-lg">Fix Successfully Shipped</h3>
              </div>
              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex justify-between border-b border-white/5 pb-2">
                  <span className="text-slate-500">Complexity</span>
                  <span className="font-medium capitalize">{result.complexity}</span>
                </div>
                <div className="flex justify-between border-b border-white/5 pb-2">
                  <span className="text-slate-500">Total Steps</span>
                  <span className="font-medium">{result.steps}</span>
                </div>
                <div className="flex justify-between border-b border-white/5 pb-2">
                  <span className="text-slate-500">Test Retries</span>
                  <span className="font-medium">{result.retry_count}</span>
                </div>
                {result.pr_url && result.pr_url !== "" ? (
                  <a href={result.pr_url} target="_blank" rel="noreferrer" className="mt-4 flex items-center justify-center gap-2 w-full py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg transition-colors font-medium text-emerald-400">
                    <GitPullRequest className="w-4 h-4" />
                    Review Pull Request
                  </a>
                ) : (
                  <div className="mt-4 text-center text-slate-400 p-3 bg-white/5 rounded-lg border border-white/5">
                    Run completed (Dry Run or No PR created)
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Card */}
          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-6 animate-in slide-in-from-bottom-4 fade-in duration-500">
              <div className="flex items-center gap-3 text-red-400 mb-2">
                <AlertTriangle className="w-6 h-6" />
                <h3 className="font-bold text-lg">Pipeline Failed</h3>
              </div>
              <p className="text-sm text-slate-300 break-words">{error}</p>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
