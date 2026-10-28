export const BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE}/api${path}`, { ...options, headers: { ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...options.headers }, signal: options.signal || AbortSignal.timeout(path.startsWith('/ai') ? 60000 : 30000) })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try { const body = await response.json(); message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail) } catch { /* non-JSON error */ }
    throw new Error(message)
  }
  return response.json()
}
export const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })
export function download(content: string, name: string) { const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' })); const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000) }
export async function submission() {
  const r = await fetch(`${BASE}/api/competition/submission`,{signal:AbortSignal.timeout(30000)})
  if (!r.ok) throw new Error('Could not generate competition submission. Please retry.')
  download(await r.text(), 'churniq_submission.csv')
}
export type Feature = { name: string; min: number; max: number; default: number; integer: boolean; options: number[] | null }
export type Prediction = { probability: number; predicted_class: number; risk: string; contributions: { feature: string; contribution: number; value: number }[]; input_summary: Record<string, number> }
export type BatchRow = { row: number; probability: number | null; prediction: number | null; risk: string | null; error: string; [key: string]: unknown }
export type Batch = { rows: BatchRow[]; valid_count: number; invalid_count: number; summary: Record<string, number>; csv: string }
export type Assumptions = { monthly_revenue: number; contribution_margin: number; lifetime_months: number; monthly_discount_rate: number; intervention_cost: number; effectiveness: number; campaign_budget: number; currency: string }
export type FinanceInput = { assumptions: Assumptions; probabilities: number[]; probability_source: 'hypothetical' | 'single_prediction' | 'uploaded_portfolio' }
export const pct = (n: number, digits = 1) => `${(n * 100).toFixed(digits)}%`
export const number = (n: number, digits = 0) => new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(n)
