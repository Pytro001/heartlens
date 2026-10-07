export type TargetKey = 'Cath' | 'LAD' | 'LCX' | 'RCA'
export const TARGETS: TargetKey[] = ['Cath', 'LAD', 'LCX', 'RCA']
export const VESSELS: TargetKey[] = ['LAD', 'LCX', 'RCA']

export interface Stat { mean: number; lo: number; hi: number; sd: number }
export type MetricName = 'roc_auc' | 'pr_auc' | 'accuracy' | 'precision' | 'recall' | 'f1' | 'brier'

export interface FeatureMeta {
  key: string; label: string; kind: 'num' | 'bin' | 'ord'; unit: string
  min: number; max: number; median: number; step: number
}
export type Calibration =
  | { type: 'platt'; a: number; b: number }
  | { type: 'isotonic'; x: number[]; y: number[] }

export interface TargetMeta {
  key: TargetKey; label: string; model: string; model_label: string; params: Record<string, unknown>
  calibration: Calibration; onnx: string; onnx_outputs: string[]; onnx_max_abs_err: number
  shap_base: number; prevalence: number; global_importance: { key: string; value: number }[]
}
export interface Meta {
  dataset: { name: string; source: string; doi: string; license: string; patients: number; features: number }
  protocol: Record<string, string>
  features: FeatureMeta[]
  targets: TargetMeta[]
  cv: Record<TargetKey, Record<string, Record<string, Record<MetricName, Stat>>>>
  selection: Record<TargetKey, { kind: string; cal: string }>
  calibration_curves: Record<TargetKey, Record<'raw' | 'calibrated', { pred: number; obs: number; n: number }[]>>
  confusion: Record<TargetKey, number[][]>
  train_seconds: number
}
export interface Patient {
  id: number; x: number[]
  truth: Record<TargetKey, number>
  pred: Record<TargetKey, number>
  oof: Record<TargetKey, number>
  shap: Record<TargetKey, number[]>
}
export type Probs = Record<TargetKey, number>
